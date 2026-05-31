import os
import numpy as np
import h5py
from glob import glob
import time
from plyfile import PlyData
import argparse
from tqdm import tqdm
import dataflow
import json

def read_boundary_from_ply(boundary_file):
    """从指定的PLY文件中读取边界粒子
    
    Args:
        boundary_file: 边界PLY文件路径
        
    Returns:
        tuple: (边界粒子位置, 边界粒子法线)
    """
    if not boundary_file or not os.path.exists(boundary_file):
        print(f"边界文件不存在: {boundary_file}")
        return None, None
        
    try:
        print(f"读取边界文件: {boundary_file}")
        plydata = PlyData.read(boundary_file)
        vertex = plydata['vertex']
        
        # 提取基本属性
        x = np.array(vertex['x'])
        y = np.array(vertex['y'])
        z = np.array(vertex['z'])
        
        # 构建位置数组
        pos = np.column_stack((x, y, z))
        
        # 尝试读取法线信息
        if all(attr in vertex for attr in ['nx', 'ny', 'nz']):
            nx = np.array(vertex['nx'])
            ny = np.array(vertex['ny'])
            nz = np.array(vertex['nz'])
            normals = np.column_stack((nx, ny, nz))
        else:
            # 如果没有法线信息，创建默认法线
            normals = pos.copy()
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # 防止除零错误
            normals = -normals / norms  # 假设法线指向内部
        
        return pos.astype(np.float32), normals.astype(np.float32)
        
    except Exception as e:
        print(f"处理边界文件 {boundary_file} 时出错: {e}")
        return None, None

def read_ply_with_phases(ply_file):
    """读取包含多相流体信息的PLY文件
    
    Args:
        ply_file: PLY文件路径
        
    Returns:
        dict: 包含位置、速度、相分数等信息的字典
    """
    try:
        plydata = PlyData.read(ply_file)
        vertex = plydata['vertex']
        
        # 提取基本属性
        x = np.array(vertex['x'])
        y = np.array(vertex['y'])
        z = np.array(vertex['z'])
        
        # 提取速度
        vx = np.array(vertex['vx'])
        vy = np.array(vertex['vy'])
        vz = np.array(vertex['vz'])
        
        # 提取相分数 - 提取所有相分数
        phase_properties = []
        phase_data = []
        
        for prop in vertex.properties:
            if prop.name.startswith('p') and len(prop.name) > 1:
                try:
                    # 尝试将属性名称解析为p1, p2等
                    phase_idx = int(prop.name[1:])
                    phase_properties.append((phase_idx, prop.name))
                except:
                    pass
        
        # 按相索引排序
        phase_properties.sort()
        
        # 提取所有相分数
        for _, prop_name in phase_properties:
            phase_data.append(np.array(vertex[prop_name]))
        
        # 提取CD和CF参数从注释中
        cd = 0.5
        cf = 0.5
        for comment in plydata.comments:
            if comment.startswith('cd:'):
                cd = float(comment.split(':')[1].strip())
            elif comment.startswith('cf:'):
                cf = float(comment.split(':')[1].strip())
        
        # 构建位置和速度数组
        pos = np.column_stack((x, y, z))
        vel = np.column_stack((vx, vy, vz))
        
        # 构建相分数数组 - 包括所有相
        phase_fractions = np.column_stack(phase_data) if phase_data else None
        
        frame_id = int(os.path.basename(ply_file).split('.')[0])
        
        return {
            'pos': pos.astype(np.float32),
            'vel': vel.astype(np.float32),
            'phase_fractions': phase_fractions.astype(np.float32) if phase_fractions is not None else None,
            'cd': cd,
            'cf': cf,
            'frame_id': frame_id,
            'scene_id': os.path.dirname(ply_file),
            'm': np.ones(len(pos), dtype=np.float32),  # 质量默认为1
            'viscosity': np.ones(len(pos), dtype=np.float32) * 0.01  # 粘度默认值
        }
    except Exception as e:
        print(f"处理文件 {ply_file} 时出错: {e}")
        return None

def convert_ply_sequence_to_h5(input_dir, output_dir, boundary_file=None, scene_name=None):
    """将PLY文件序列转换为H5格式的训练数据
    
    Args:
        input_dir: 包含PLY文件的输入目录
        output_dir: 输出目录
        boundary_file: 边界粒子PLY文件路径
        scene_name: 场景名称，用于命名输出文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有提供场景名称，使用输入目录的最后一部分
    if scene_name is None:
        scene_name = os.path.basename(os.path.normpath(input_dir))
    
    # 读取边界粒子数据
    box, box_normals = None, None
    if boundary_file:
        box, box_normals = read_boundary_from_ply(boundary_file)
        
    if box is None:
        print("警告: 未加载边界数据，将创建没有边界的数据集")
    else:
        print(f"成功加载 {len(box)} 个边界粒子")
    
    # 查找所有PLY文件
    ply_files = sorted(glob(os.path.join(input_dir, "*.ply")))
    if not ply_files:
        print(f"在 {input_dir} 中没有找到PLY文件")
        return
    
    print(f"找到 {len(ply_files)} 个PLY文件")
    
    # 创建输出文件
    output_file = os.path.join(output_dir, f"{scene_name}.h5")
    
    # 读取第一个文件以获取CD和CF参数以及相数量
    first_frame = read_ply_with_phases(ply_files[0])
    if not first_frame:
        print("无法读取第一帧数据，退出")
        return
    
    cd = first_frame['cd']
    cf = first_frame['cf']
    num_phases = first_frame['phase_fractions'].shape[1] if first_frame['phase_fractions'] is not None else 0
    
    # 创建H5文件
    with h5py.File(output_file, 'w') as h5f:
        # 存储全局属性
        h5f.attrs['cd'] = cd
        h5f.attrs['cf'] = cf
        h5f.attrs['num_phases'] = num_phases
        h5f.attrs['num_frames'] = len(ply_files)
        h5f.attrs['scene_name'] = scene_name
        
        # 存储边界数据
        if box is not None:
            h5f.create_dataset('box', data=box)
            h5f.create_dataset('box_normals', data=box_normals)
        
        # 创建帧组
        frames_group = h5f.create_group('frames')
        
        # 处理每一帧
        for i, ply_file in enumerate(tqdm(ply_files, desc=f"处理 {scene_name} 帧")):
            frame_data = read_ply_with_phases(ply_file)
            if not frame_data:
                print(f"跳过无效帧 {ply_file}")
                continue
                
            # 创建帧子组
            frame_id = frame_data['frame_id']
            frame_group = frames_group.create_group(f"{frame_id}")
            
            # 存储帧数据
            frame_group.create_dataset('pos', data=frame_data['pos'])
            frame_group.create_dataset('vel', data=frame_data['vel'])
            
            if frame_data['phase_fractions'] is not None:
                frame_group.create_dataset('phase_fractions', data=frame_data['phase_fractions'])
                
            frame_group.create_dataset('m', data=frame_data['m'])
            frame_group.create_dataset('viscosity', data=frame_data['viscosity'])
            
            # 存储帧元数据
            frame_group.attrs['frame_id'] = frame_id
            frame_group.attrs['scene_id'] = os.path.basename(input_dir)
    
    print(f"成功将 {len(ply_files)} 个帧保存到 {output_file}")
    
    # 创建索引文件
    index_file = os.path.join(output_dir, "dataset_index.json")
    dataset_info = {
        "files": [output_file],
        "scenes": [{
            "name": scene_name,
            "file": os.path.basename(output_file),
            "cd": cd,
            "cf": cf,
            "num_phases": num_phases,
            "num_frames": len(ply_files)
        }]
    }
    
    # 如果索引文件已存在，更新它
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            try:
                existing_index = json.load(f)
                
                # 更新文件列表，避免重复
                if output_file not in existing_index.get("files", []):
                    existing_index.setdefault("files", []).append(output_file)
                
                # 更新场景信息
                scene_names = [s["name"] for s in existing_index.get("scenes", [])]
                if scene_name not in scene_names:
                    existing_index.setdefault("scenes", []).append(dataset_info["scenes"][0])
                
                dataset_info = existing_index
            except:
                pass
    
    with open(index_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"更新了数据集索引: {index_file}")
    
    return output_file





def batch_process_directories(base_dir, output_base_dir, boundary_file=None):
    """批处理所有子目录中的PLY文件
    
    Args:
        base_dir: 包含多个场景子目录的基础目录
        output_base_dir: 输出基础目录
        boundary_file: 边界文件路径
    """
    # 找到所有子目录
    scene_dirs = [d for d in glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    if not scene_dirs:
        print(f"在 {base_dir} 中没有找到场景子目录")
        return
    
    print(f"找到 {len(scene_dirs)} 个场景目录")
    
    # 处理每个场景
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        print(f"\n处理场景: {scene_name}")
        
        # 检查是否包含PLY文件
        ply_files = glob(os.path.join(scene_dir, "*.ply"))
        if not ply_files:
            print(f"场景 {scene_name} 中没有找到PLY文件，跳过")
            continue
        
        # 转换该场景
        convert_ply_sequence_to_h5(
            input_dir=scene_dir,
            output_dir=output_base_dir,
            boundary_file=boundary_file,
            scene_name=scene_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将PLY文件转换为HDF5训练数据格式')
    parser.add_argument('input', type=str, help='输入目录 - 可以是单个场景目录或包含多个场景子目录的基础目录')
    parser.add_argument('output', type=str, help='输出目录')
    parser.add_argument('--boundary', type=str, default=None, help='边界粒子PLY文件路径')
    parser.add_argument('--batch', action='store_true', help='批处理模式 - 处理输入目录下的所有子目录')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_process_directories(args.input, args.output, args.boundary)
    else:
        convert_ply_sequence_to_h5(args.input, args.output, args.boundary)