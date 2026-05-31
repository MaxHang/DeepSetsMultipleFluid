#!/usr/bin/env python3
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
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from create_physics_scenes import obj_surface_to_particles, obj_volume_to_particles
import open3d as o3d
import plyfile
import yaml

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


def read_pos_vel_from_h5(path, random_rotation=False):
    """Load h5py data files from specified path."""
    with h5py.File(path, 'r') as h5f:
        box = h5f['box'][:]
        box_normals = h5f['box_normals'][:]
        cd = np.float32(h5f.attrs['cd'])
        cf = np.float32(h5f.attrs['cf'])
        frame_group = h5f['frames/1']  # 取第一帧
        pos = frame_group['pos'][:]
        vel = frame_group['vel'][:]
        phase_fractions = frame_group['phase_fractions'][:] 
    return [box, box_normals, pos, vel, phase_fractions], cd, cf


def write_particles(path_without_ext, pos, vel=None, phase_fractions=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}
    if vel is not None:
        arrs['vel'] = vel
    if phase_fractions is not None:
        arrs['phase_fractions'] = phase_fractions
    np.savez(path_without_ext + '.npz', **arrs)

    if options and options.write_ply:
        # 准备需要写入到PLY的数据
        num_particles = pos.shape[0]
        
        # 准备plyfile所需的数据结构
        vertex_data = []
        
        # 添加位置数据 (x, y, z)
        vertex_data.append(('x', pos[:, 0].astype('float32')))
        vertex_data.append(('y', pos[:, 1].astype('float32')))
        vertex_data.append(('z', pos[:, 2].astype('float32')))
        
        # 如果有速度数据，添加速度 (vx, vy, vz)
        if vel is not None:
            vertex_data.append(('vx', vel[:, 0].astype('float32')))
            vertex_data.append(('vy', vel[:, 1].astype('float32')))
            vertex_data.append(('vz', vel[:, 2].astype('float32')))
        
        # 添加相体积分数
        if phase_fractions is not None:
            # 添加每个相的体积分数
            for i in range(phase_fractions.shape[1]):
                phase_name = f"p{i+1}"
                vertex_data.append((phase_name, phase_fractions[:, i].astype('float32')))
            
            # # 计算最后一个相的体积分数
            # last_phase = np.ones(phase_fractions.shape[0], dtype=np.float32)
            # last_phase -= np.sum(phase_fractions, axis=1)
            # last_phase = np.clip(last_phase, 0.0, 1.0)  # 确保值在[0,1]范围内
            
            # # 添加最后一个相的体积分数
            # last_phase_idx = phase_fractions.shape[1] + 1
            # vertex_data.append((f"p{last_phase_idx}", last_phase.astype('float32')))
            
            # 设置颜色，根据相体积分数
            colors = np.zeros((num_particles, 3), dtype=np.float32)
            
            # 设置红色通道为第一相的体积分数
            colors[:, 0] = phase_fractions[:, 0]
            
            # 如果有第二相，设置绿色通道为第二相的体积分数，蓝色通道为最后一相
            if phase_fractions.shape[1] > 1:
                colors[:, 1] = phase_fractions[:, 1]
                
            # 归一化颜色值
            row_sums = np.sum(colors, axis=1, keepdims=True)
            # 创建一个广播掩码，将(N,1)扩展为(N,3)
            mask = (row_sums > 0).repeat(3, axis=1)
            # 现在掩码的形状是(N,3)，可以用于索引colors
            colors[mask] = (colors / row_sums)[mask]
            
            # 将颜色值转换为0-255范围的整数
            colors = (colors * 255).astype(np.uint8)
        else:
            # 默认颜色 (灰色)
            colors = np.full((num_particles, 3), 128, dtype=np.uint8)
        
        # 添加RGB颜色通道
        vertex_data.append(('red', colors[:, 0]))
        vertex_data.append(('green', colors[:, 1]))
        vertex_data.append(('blue', colors[:, 2]))
        
        # 创建vertex元素
        vertex_element = plyfile.PlyElement.describe(
            np.array(list(zip(*[data for _, data in vertex_data])), 
                     dtype=[(name, data.dtype.str) for name, data in vertex_data]),
            'vertex'
        )
        
        # 创建PLY文件
        ply_data = plyfile.PlyData([vertex_element], text=True)
        
        # 写入PLY文件
        ply_data.write(path_without_ext + '.ply')

    if options and options.write_bgeo:
        # 注意：bgeo格式可能不支持相体积分数，只写入位置和速度
        write_bgeo_from_numpy(path_without_ext + '.bgeo', pos, vel)


def read_pos_normal_from_ply(path):
    """Load ply files from specified path."""
    pos, normals = None, None
    with open(path, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
        
        # 获取顶点数据
        vertices = plydata['vertex'].data
        
        # 提取位置和法向量
        pos = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        if 'nx' in vertices.dtype.names:
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    return pos, normals

def run_sim_tf(trainscript_module, cfg, weights_path, scene, num_steps, output_dir,
               options, gpu='0'):

    # init the network
    model = trainscript_module.create_model(gpu, **cfg.get('model', {}))
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

    print(scene.keys())
    cd, cf = 0.3, 0.7
    fluids = []
    
    if 'h5_path' in scene:
        print(scene['h5_path'])
        data, cd, cf = read_pos_vel_from_h5(scene['h5_path'], random_rotation=True)
        box, box_normals, points, velocities, phase_fractions = data
        x = scene['fluids'][0]
        range_ = range(x['start'], x['stop'], x['step'])
        fluids.append((points, velocities, phase_fractions, cd, cf, range_))
    else:
        # prepare static particles
        walls = []
        for x in scene['walls']:
            if 'ply_path' in x:
                points, normals = read_pos_normal_from_ply(x['ply_path'])
            else:
                points, normals = obj_surface_to_particles(x['path'])
            if 'invert_normals' in x and x['invert_normals']:
                normals = -normals
            points += np.asarray([x['translation']], dtype=np.float32)
            walls.append((points, normals))
        box = np.concatenate([x[0] for x in walls], axis=0)
        box_normals = np.concatenate([x[1] for x in walls], axis=0)
        # prepare fluids
        for x in scene['fluids']:
            if 'h5_path' in x and os.path.exists(x['h5_path']):
                data = read_pos_vel_from_h5(x['h5_path'])
                points, velocities = data[0], data[1]
                # 检查是否读取了相体积分数
                phase_fractions = data[2] if len(data) > 2 else None
            if 'ply_path' in x:
                points, _ = read_pos_normal_from_ply(x['ply_path'])
                velocities = np.empty_like(points)
                velocities[:, 0] = x['velocity'][0]
                velocities[:, 1] = x['velocity'][1]
                velocities[:, 2] = x['velocity'][2]
                phase_fractions = None
            else:
                points = obj_volume_to_particles(x['path'])[0]
                points += np.asarray([x['translation']], dtype=np.float32)
                velocities = np.empty_like(points)
                velocities[:, 0] = x['velocity'][0]
                velocities[:, 1] = x['velocity'][1]
                velocities[:, 2] = x['velocity'][2]
                # 如果配置中指定了相体积分数，使用它
                phase_fractions = None
                if 'phase_fractions' in x:
                    num_phases = len(x['phase_fractions'])
                    phase_fractions = np.zeros((points.shape[0], num_phases), dtype=np.float32)

            range_ = range(x['start'], x['stop'], x['step'])
            fluids.append((points, velocities, phase_fractions, cd, cf, range_))

    output_dir = output_dir + '_cd_' + str(cd) + '_cf_' + str(cf)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # compute lowest point for removing out of bounds particles
    min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))
    # export static particles
    write_particles(os.path.join(output_dir, 'box'), box, box_normals, None, options)

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)
    phase_fractions = np.empty(shape=(0, model.num_phases), dtype=np.float32)

    start_time = time.time()
    for step in range(num_steps):
        # add from fluids to pos vel arrays
        for points, velocities, fluid_phases, cd, cf, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)
                if fluid_phases is None:
                    fluid_phases = np.zeros((points.shape[0], model.num_phases), dtype=np.float32)
                    if step % 8 == 0:
                        fluid_phases[:, 0] = 1
                    else:
                        fluid_phases[:, 1] = 1
                phase_fractions = np.concatenate([phase_fractions, fluid_phases], axis=0)
                print('add', pos.shape, vel.shape, phase_fractions.shape)

        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            if isinstance(pos, np.ndarray):
                write_particles(fluid_output_path, pos, vel, phase_fractions, options)
            else:
                write_particles(fluid_output_path, pos.numpy(), vel.numpy(), phase_fractions.numpy(), options)

            # 准备输入，包含相体积分数
            inputs = (pos, vel, phase_fractions, box, box_normals)
            
            # 调用模型执行一步仿真，根据模型输出处理返回结果
            if phase_fractions is not None and hasattr(model, 'num_phases') and model.num_phases > 1:
                # 执行多相流体模拟
                pos, vel, phase_fractions = model(inputs, cd=cd, cf=cf)
            else:
                # 执行单相流体模拟
                pos, vel = model(inputs)

        # remove out of bounds particles
        if step % 10 == 0:
            print(step, 'num particles', pos.shape[0])
            mask = pos[:, 1] > min_y
            if np.count_nonzero(mask) < pos.shape[0]:
                pos = pos[mask]
                vel = vel[mask]
                # 同样过滤相体积分数
                if phase_fractions is not None:
                    phase_fractions = phase_fractions[mask]

    end_time = time.time()  
    print('Total time: ', end_time - start_time)
    print('average time: ', (end_time - start_time) / num_steps)


def main():
    parser = argparse.ArgumentParser(
        description=
        "Runs a fluid network on the given scene and saves the particle positions as npz sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trainscript",
                        type=str,
                        help="The python training script.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=
        "The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
    )
    parser.add_argument("--num_steps",
                        type=int,
                        default=250,
                        help="The number of simulation steps. Default is 250.")
    parser.add_argument("--cfg",
                        type=str,
                        help="The path to the yaml config file")
    parser.add_argument('--gpu',
                        help='指定使用的GPU ID，例如：0,1,2',
                        type=str,
                        default='0')
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

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

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

    gpu_id = int(args.gpu)
    return run_sim_tf(trainscript_module, cfg, args.weights, scene,
                          args.num_steps, args.output, args, gpu=gpu_id)


if __name__ == '__main__':
    sys.exit(main())
