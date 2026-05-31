#!/usr/bin/env python3
import open3d as o3d
import os
import sys
import argparse
import h5py
import numpy as np
import re
from glob import glob
import time
import importlib
import json
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from create_physics_scenes import obj_surface_to_particles, obj_volume_to_particles
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy

# 在创建模型后添加


def print_model_structure(model):
    """Print model structure information in text format"""
    print("\n======= MODEL STRUCTURE SUMMARY =======")

    # Use built-in summary method
    model.summary()

    # Print network layer information and parameters
    print("\nLayer Details:")
    total_params = 0
    trainable_params = 0
    for layer_idx, layer in enumerate(model.layers):
        layer_name = getattr(layer, 'name', f'layer_{layer_idx}')
        print(f"Layer {layer_idx}: {layer_name}")

        # Print parameters for each layer
        layer_params = 0
        if hasattr(layer, 'trainable_variables'):
            for var in layer.trainable_variables:
                params = np.prod(var.shape)
                layer_params += params
                trainable_params += params
                print(f"  - {var.name}: {var.shape} = {params:,} params")

        print(f"  Total params in layer: {layer_params:,}")
        total_params += layer_params

    # Print network architecture specifics if available
    if hasattr(model, 'layer_channels'):
        print(f"\nChannel configuration: {model.layer_channels}")

    if hasattr(model, '_all_convs'):
        print("\nConvolution layers:")
        for name, _ in model._all_convs:
            print(f"  - {name}")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("===================================\n")


def read_pos_vel_from_h5(path):
    """Load h5py data files from specified path."""
    hf = h5py.File(path, 'r')
    data = []
    for dn in ['positions', 'velocities']:
        d = np.array(hf.get(dn))
        # 取最后一帧
        data.append(d[0])
        # data.append(d[-1])
        d[..., 0] *= -1
        d[..., [1, 2]] = d[..., [2, 1]]
    hf.close()
    return data


def write_particles(path_without_ext, pos, vel=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}
    if not vel is None:
        arrs['vel'] = vel
    np.savez(path_without_ext + '.npz', **arrs)

    if options and options.write_ply:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
        o3d.io.write_point_cloud(path_without_ext + '.ply', pcd)

    if options and options.write_bgeo:
        write_bgeo_from_numpy(path_without_ext + '.bgeo', pos, vel)


def run_sim_tf(trainscript_module, weights_path, scene, num_steps, output_dir,
               options):

    # init the network
    model = trainscript_module.create_model()
    model.init()
    # 支持ckpt和h5两种权重格式
    if weights_path.endswith('.ckpt') or weights_path.endswith('.index'):
        checkpoint = tf.train.Checkpoint(model=model)
        restore_path = weights_path
        if restore_path.endswith('.index'):
            restore_path = restore_path[:-6]
        print(f"Restoring from checkpoint: {restore_path}")
        checkpoint.restore(restore_path).expect_partial()
    else:
        model.load_weights(weights_path, by_name=True)

    print_model_structure(model)

    # prepare static particles
    walls = []
    for x in scene['walls']:
        points, normals = obj_surface_to_particles(x['path'])
        if 'invert_normals' in x and x['invert_normals']:
            normals = -normals
        points += np.asarray([x['translation']], dtype=np.float32)
        walls.append((points, normals))
    box = np.concatenate([x[0] for x in walls], axis=0)
    box_normals = np.concatenate([x[1] for x in walls], axis=0)
    print(box.shape, box_normals.shape)

    # exit()
    # export static particles
    write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)

    # compute lowest point for removing out of bounds particles
    min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))

    # prepare fluids
    fluids = []
    for x in scene['fluids']:
        if 'h5_path' in x and os.path.exists(x['h5_path']):
            data = read_pos_vel_from_h5(x['h5_path'])
            points, velocities = data[0], data[1]
        else:
            points = obj_volume_to_particles(x['path'])[0]
            points += np.asarray([x['translation']], dtype=np.float32)
            velocities = np.empty_like(points)
            velocities[:, 0] = x['velocity'][0]
            velocities[:, 1] = x['velocity'][1]
            velocities[:, 2] = x['velocity'][2]
        range_ = range(x['start'], x['stop'], x['step'])
        fluids.append((points, velocities, range_))

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)

    start_time = time.time()
    for step in range(num_steps):
        # add from fluids to pos vel arrays
        for points, velocities, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)

        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            if isinstance(pos, np.ndarray):
                write_particles(fluid_output_path, pos, vel, options)
            else:
                write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
                                options)

            inputs = (pos, vel, None, box, box_normals)
            # print(
            #     pos.shape,
            #     vel.shape,
            #     box.shape,
            #     box_normals.shape,
            #     )
            pos, vel = model(inputs)

        # remove out of bounds particles
        if step % 10 == 0:
            print(step, 'num particles', pos.shape[0])
            mask = pos[:, 1] > min_y
            if np.count_nonzero(mask) < pos.shape[0]:
                pos = pos[mask]
                vel = vel[mask]

    end_time = time.time()
    print('Total time: ', end_time - start_time)
    print('average time: ', (end_time - start_time) / num_steps)


def run_sim_torch(trainscript_module, weights_path, scene, num_steps,
                  output_dir, options):
    import torch
    device = torch.device(options.device)

    # init the network
    model = trainscript_module.create_model()
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    model.to(device)
    model.requires_grad_(False)

    # prepare static particles
    walls = []
    for x in scene['walls']:
        points, normals = obj_surface_to_particles(x['path'])
        if 'invert_normals' in x and x['invert_normals']:
            normals = -normals
        points += np.asarray([x['translation']], dtype=np.float32)
        walls.append((points, normals))
    box = np.concatenate([x[0] for x in walls], axis=0)
    box_normals = np.concatenate([x[1] for x in walls], axis=0)

    # export static particles
    write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)

    # compute lowest point for removing out of bounds particles
    min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))

    box = torch.from_numpy(box).to(device)
    box_normals = torch.from_numpy(box_normals).to(device)

    # prepare fluids
    fluids = []
    for x in scene['fluids']:
        points = obj_volume_to_particles(x['path'])[0]
        points += np.asarray([x['translation']], dtype=np.float32)
        velocities = np.empty_like(points)
        velocities[:, 0] = x['velocity'][0]
        velocities[:, 1] = x['velocity'][1]
        velocities[:, 2] = x['velocity'][2]
        range_ = range(x['start'], x['stop'], x['step'])
        fluids.append(
            (points.astype(np.float32), velocities.astype(np.float32), range_))

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)

    for step in range(num_steps):
        # add from fluids to pos vel arrays
        for points, velocities, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)

        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            if isinstance(pos, np.ndarray):
                write_particles(fluid_output_path, pos, vel, options)
            else:
                write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
                                options)

            inputs = (torch.from_numpy(pos).to(device),
                      torch.from_numpy(vel).to(device), None, box, box_normals)
            pos, vel = model(inputs)
            pos = pos.cpu().numpy()
            vel = vel.cpu().numpy()

        # remove out of bounds particles
        if step % 10 == 0:
            print(step, 'num particles', pos.shape[0])
            mask = pos[:, 1] > min_y
            if np.count_nonzero(mask) < pos.shape[0]:
                pos = pos[mask]
                vel = vel[mask]


def main():
    parser = argparse.ArgumentParser(
        description="Runs a fluid network on the given scene and saves the particle positions as npz sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trainscript",
                        type=str,
                        help="The python training script.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
    )
    
    parser.add_argument("--num_steps",
                        type=int,
                        default=250,
                        help="The number of simulation steps. Default is 250.")
    parser.add_argument("--scene",
                        type=str,
                        required=True,
                        help="A json file which describes the scene.")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory for the particle data.")
    parser.add_argument("--write-ply",
                        action='store_true',
                        help="Export particle data also as .ply sequence")
    parser.add_argument("--write-bgeo",
                        action='store_true',
                        help="Export particle data also as .bgeo sequence")
    parser.add_argument("--device",
                        type=str,
                        default='cuda',
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()
    print(args)

    '''if args.trainscript is /path/to/my_script.py, then module_name is set to my_script
    this is train_network_tf or train_network_torch
    '''
    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    print(module_name)
    '''adds the current directory to the module search path in Python. ensure that Python can find the module in the current directory that named module_name'''
    sys.path.append('.')
    '''use importlib.import_module dynamically imports module named module_name and assigns it to trainscript_module'''
    trainscript_module = importlib.import_module(module_name)

    with open(args.scene, 'r') as f:
        scene = json.load(f)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    return run_sim_tf(trainscript_module, args.weights, scene,
                          args.num_steps, args.output, args)

    # if args.weights.endswith('.h5'):
    #     return run_sim_tf(trainscript_module, args.weights, scene,
    #                       args.num_steps, args.output, args)

    # elif args.weights.endswith('.pt'):
    #     return run_sim_torch(trainscript_module, args.weights, scene,
    #                          args.num_steps, args.output, args)


if __name__ == '__main__':
    sys.exit(main())
