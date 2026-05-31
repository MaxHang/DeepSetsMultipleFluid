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
    """从h5py数据文件中加载数据，并尝试加载场景属性。"""
    with h5py.File(path, 'r') as h5f:
        box = h5f['box'][:]
        box_normals = h5f['box_normals'][:]
        cd = np.float32(h5f.attrs.get('cd', 0.5))
        cf = np.float32(h5f.attrs.get('cf', 0.5))
        
        # 尝试从 h5 属性中读取场景信息
        num_phases = int(h5f.attrs.get('num_phases', 2))
        # 如果 'density' 存在，则读取，否则根据num_phases创建默认值
        if 'density' in h5f.attrs:
            density = np.array(h5f.attrs['density'], dtype=np.float32)
        else:
            density = np.full(shape=(num_phases,), fill_value=1000.0, dtype=np.float32)

        frame_group = h5f['frames/1']
        pos = frame_group['pos'][:]
        vel = frame_group['vel'][:]
        phase_fractions = frame_group['phase_fractions'][:]
        
    scene_props = {'num_phases': num_phases, 'density': density}
    return [box, box_normals, pos, vel, phase_fractions], cd, cf, scene_props

import numpy as np
import plyfile

def write_particles(path_without_ext, pos, vel=None, phase_fractions=None, options=None, cd=None, cf=None, densities=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """

    if options and options.write_ply:
        num_particles = pos.shape[0]
        vertex_data = []
        
        # 位置
        vertex_data.append(('x', pos[:, 0].astype('float32')))
        vertex_data.append(('y', pos[:, 1].astype('float32')))
        vertex_data.append(('z', pos[:, 2].astype('float32')))
        
        # 速度
        if vel is not None:
            vertex_data.append(('vx', vel[:, 0].astype('float32')))
            vertex_data.append(('vy', vel[:, 1].astype('float32')))
            vertex_data.append(('vz', vel[:, 2].astype('float32')))
        
        # 相体积分数
        if phase_fractions is not None:
            n_phases = phase_fractions.shape[1]
            for i in range(n_phases):
                phase_name = f"p{i+1}"
                vertex_data.append((phase_name, phase_fractions[:, i].astype('float32')))
            
            # ===================== 修复：正确支持 1/2/3 相 =====================
            colors = np.zeros((num_particles, 3), dtype=np.float32)
            
            # 第1相 → 红
            if n_phases >= 1:
                colors[:, 0] = phase_fractions[:, 0]
            # 第2相 → 绿
            if n_phases >= 2:
                colors[:, 1] = phase_fractions[:, 1]
            # 第3相 → 蓝（关键修复！）
            if n_phases >= 3:
                colors[:, 2] = phase_fractions[:, 2]
            
            # 归一化颜色
            row_sums = colors.sum(axis=1, keepdims=True)
            mask = row_sums > 0
            colors[mask.squeeze()] = colors[mask.squeeze()] / row_sums[mask.squeeze()]
            
            # 转 0-255
            colors = (colors * 255).astype(np.uint8)
            # =================================================================
        else:
            colors = np.full((num_particles, 3), 128, dtype=np.uint8)
        
        # 添加颜色通道
        vertex_data.append(('red', colors[:, 0]))
        vertex_data.append(('green', colors[:, 1]))
        vertex_data.append(('blue', colors[:, 2]))
        
        # 创建 PLY 元素
        dtype_list = [(name, data.dtype.str) for name, data in vertex_data]
        vertex_array = np.core.records.fromarrays([data for _, data in vertex_data], dtype=dtype_list)
        vertex_element = plyfile.PlyElement.describe(vertex_array, 'vertex')
        
        # 注释
        comments = []
        if cd is not None:
            comments.append(f"cd: {float(cd):.6g}")
        if cf is not None:
            comments.append(f"cf: {float(cf):.6g}")
        if densities is not None:
            dens_arr = np.array(densities, dtype=np.float32).ravel()
            comments.append("dens: " + ",".join([f"{x:.6g}" for x in dens_arr]))
        
        # 写入 PLY
        ply_data = plyfile.PlyData([vertex_element], comments=comments, text=True)
        ply_data.write(path_without_ext + '.ply')

    if options and options.write_bgeo:
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

def read_fluid(path):
    """
    从PLY文件中读取位置(pos)和p参数(所有pi字段)
    
    参数:
        path: PLY文件路径（包含.ply后缀）
    
    返回:
        pos: 点云位置数组，shape为(N, 3)
        p: 所有p参数的数组，shape为(N, M)，M是p参数的数量（p1-pn）
           如果没有p参数，返回None
    """
    pos, p = None, None
    
    # 读取PLY文件
    with open(path, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
        
        # 获取顶点数据
        vertices = plydata['vertex'].data
        
        # 提取位置信息（x/y/z）
        pos = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        
        # 提取所有p开头的字段（p1/p2/p3...pn）
        p_fields = [field for field in vertices.dtype.names if field.startswith('p')]
        if p_fields:
            # 按字段名排序（确保p1在前，p2次之，...pn最后）
            p_fields.sort(key=lambda x: int(x[1:]))
            # 拼接所有p字段为二维数组
            p = np.vstack([vertices[field] for field in p_fields]).T
    
    return pos, p

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

    # print_model_structure(model)

    fluids = []
    cd, cf = None, None
    scene_num_phases, scene_phase_densities = None, None
    print(scene.keys())
    if 'h5_path' in scene:
        print(scene['h5_path'])
        data, cd, cf, h5_scene_props = read_pos_vel_from_h5(scene['h5_path'], random_rotation=True)
        box, box_normals, points, velocities, phase_fractions = data
        # h5文件中的属性优先
        scene_num_phases = h5_scene_props['num_phases']
        scene_phase_densities = h5_scene_props['density']
        print(f"Scene properties loaded from H5: {scene_num_phases} phases with densities {scene_phase_densities}")
        print(f"Diffusion coefficient (cd): {cd}, Convection coefficient (cf): {cf}")
        x = scene['fluids'][0]
        range_ = range(x['start'], x['stop'], x['step'])
        fluids.append((points, velocities, phase_fractions, cd, cf, range_))
    else:
        ## V4-MOD: 从场景定义中获取场景属性
        props = scene['scene_properties']
        scene_num_phases = int(props.get('num_phases', 1))
        scene_phase_densities = np.array(props.get('density', [1000.0]*scene_num_phases), dtype=np.float32)
        print(f"Scene properties loaded: {scene_num_phases} phases with densities {scene_phase_densities}")
        
        # 获取扩散和交换系数
        cd = props['cd']
        cf = props['cf']
        print(f"Diffusion coefficient (cd): {cd}, Convection coefficient (cf): {cf}")

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
        print(f"Box shape: {box.shape}, Normals shape: {box_normals.shape}")
        # prepare fluids
        for x in scene['fluids']:
            if 'h5_path' in x and os.path.exists(x['h5_path']):
                data = read_pos_vel_from_h5(x['h5_path'])
                points, velocities = data[0], data[1]
                # 检查是否读取了相体积分数
                phase_fractions = data[2] if len(data) > 2 else None
            if 'ply_path' in x:
                points, phase_fractions = read_fluid(x['ply_path'])
                velocities = np.empty_like(points)
                velocities[:, 0] = x['velocity'][0]
                velocities[:, 1] = x['velocity'][1]
                velocities[:, 2] = x['velocity'][2]
                if 'phase' in x:
                    num_phases = scene_num_phases
                    phase_fractions = np.zeros((points.shape[0], num_phases), dtype=np.float32)
                    phase_fractions[:, x['phase']] = 1.0

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

    ## V4-MOD: 初始化粒子状态数组时使用从场景中读取的相数
    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)
    phase_fractions = np.empty(shape=(0, scene_num_phases), dtype=np.float32)

    start_time = time.time()
    for step in range(num_steps):
        # 注入新的流体粒子
        for points, velocities, fluid_phases, cd, cf, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)
                phase_fractions = np.concatenate([phase_fractions, fluid_phases], axis=0)
                print('add', pos.shape, vel.shape, phase_fractions.shape)

        if pos.shape[0] > 0:
            fluid_output_path = os.path.join(output_dir, f'fluid_{step:04d}')
            write_particles(fluid_output_path, pos, vel, phase_fractions, options, cd=cd, cf=cf, densities=scene_phase_densities)

            # 准备模型输入
            inputs = (tf.constant(pos), tf.constant(vel), tf.constant(phase_fractions), 
                      tf.constant(box), tf.constant(box_normals))
            
            ## V4-MOD: 使用统一的、新的模型调用签名
            pos_tensor, vel_tensor, phase_fractions_tensor = model(
                inputs,
                current_num_phases=tf.constant(scene_num_phases, dtype=tf.int32),
                phase_densities=tf.constant(scene_phase_densities, dtype=tf.float32),
                cd=tf.constant(cd, dtype=tf.float32),
                cf=tf.constant(cf, dtype=tf.float32),
                training=False
            )
            # 将输出转换回 numpy 数组以进行下一步处理
            pos, vel, phase_fractions = pos_tensor.numpy(), vel_tensor.numpy(), phase_fractions_tensor.numpy()

        # remove out of bounds particles
        if step % 10 == 0:
            print(f'Step {step}, Num particles: {pos.shape[0]}')
            mask = pos[:, 1] > min_y
            if np.count_nonzero(mask) < pos.shape[0]:
                pos = pos[mask]
                vel = vel[mask]
                if phase_fractions is not None:
                    phase_fractions = phase_fractions[mask]

    end_time = time.time()  
    total_time = end_time - start_time
    avg_time = total_time / num_steps if num_steps > 0 else 0
    print(f'Total time: {total_time:.2f}s')
    print(f'Average time per step: {avg_time:.4f}s')


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

    with open(args.cfg, 'r', encoding='utf-8') as f:
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

    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)

    gpu_id = int(args.gpu)
    return run_sim_tf(trainscript_module, cfg, args.weights, scene,
                          args.num_steps, args.output, args, gpu=gpu_id)


if __name__ == '__main__':
    sys.exit(main())
