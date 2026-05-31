import os
import numpy as np
import zstandard as zstd
import msgpack
import msgpack_numpy
from glob import glob
import time
from plyfile import PlyData
import argparse

# 初始化msgpack_numpy
msgpack_numpy.patch()

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
            # 如果没有法线信息，创建默认法线 (指向原点的单位向量)
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
        
        # 提取相分数
        p1 = np.array(vertex['p1'])
        p2 = np.array(vertex['p2'])
        
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
        
        # 构建相分数数组 - 只包含n-1个相（最后一个可以计算得出）
        phase_fractions = np.column_stack((p1,))  # 假设p2 = 1-p1
        
        # 假设所有粒子都是流体粒子（边界从单独文件读取）
        fluid_pos = pos
        fluid_vel = vel
        fluid_phase_fractions = phase_fractions
        
        return {
            'pos': fluid_pos.astype(np.float32),
            'vel': fluid_vel.astype(np.float32),
            'phase_fractions': fluid_phase_fractions.astype(np.float32),
            'cd': cd,
            'cf': cf,
            'frame_id': int(os.path.basename(ply_file).split('.')[0]),
            'scene_id': os.path.dirname(ply_file),
            'm': np.ones(len(fluid_pos), dtype=np.float32),  # 质量默认为1
            'viscosity': np.ones(len(fluid_pos), dtype=np.float32) * 0.01  # 粘度默认值
        }
    except Exception as e:
        print(f"处理文件 {ply_file} 时出错: {e}")
        return None

def create_window_data(frames, window_size=2):
    """从连续帧创建训练窗口数据
    
    Args:
        frames: 排序后的帧列表
        window_size: 窗口大小（默认为2，即t和t+1）
        
    Returns:
        list: 包含窗口数据的列表
    """
    if len(frames) < window_size:
        return []
    
    window_data = []
    for i in range(len(frames) - window_size + 1):
        window = []
        for j in range(window_size):
            window.append(frames[i+j])
        window_data.append(window)
    
    return window_data

def convert_ply_sequence_to_zstd(input_dir, output_dir, boundary_file=None, window_size=2):
    """将PLY文件序列转换为zstd压缩的训练数据
    
    Args:
        input_dir: 包含PLY文件的输入目录
        output_dir: 输出目录
        boundary_file: 边界粒子PLY文件路径
        window_size: 窗口大小
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 读取所有帧
    frames = []
    for ply_file in ply_files:
        print(f"处理 {ply_file}...")
        frame_data = read_ply_with_phases(ply_file)
        if frame_data:
            frames.append(frame_data)
    
    if not frames:
        print("没有有效的帧数据")
        return
    
    # 创建训练窗口
    window_data = create_window_data(frames, window_size)
    if not window_data:
        print(f"无法创建长度为 {window_size} 的窗口")
        return
    
    # 将窗口数据转换为训练样本格式
    dataset = []
    for window in window_data:
        sample = {}
        
        # 添加边界框信息（在所有帧中相同）
        sample['box'] = box
        sample['box_normals'] = box_normals
        
        # 添加每个时间步的数据
        for i, frame in enumerate(window):
            sample[f'pos{i}'] = frame['pos']
            sample[f'vel{i}'] = frame['vel']
            sample[f'm{i}'] = frame['m']
            sample[f'viscosity{i}'] = frame['viscosity']
            sample[f'frame_id{i}'] = frame['frame_id']
            sample[f'scene_id{i}'] = frame['scene_id']
            
            # 添加相分数信息
            if i == 0:  # 只在第一帧添加相参数
                sample['cd'] = frame['cd']
                sample['cf'] = frame['cf']
                sample['phase_fractions'] = frame['phase_fractions']
        
        dataset.append(sample)
    
    # 计算文件名
    cd = frames[0]['cd']
    cf = frames[0]['cf']
    output_file = os.path.join(output_dir, f"multi_phase_cd{cd:.1f}_cf{cf:.1f}.msgpack.zst")
    
    # 压缩并保存数据
    print(f"压缩并保存数据到 {output_file}...")
    compressor = zstd.ZstdCompressor(level=10)  # 较高的压缩级别
    compressed = compressor.compress(msgpack.packb(dataset, use_bin_type=True))
    
    with open(output_file, 'wb') as f:
        f.write(compressed)
    
    print(f"成功将 {len(dataset)} 个训练样本保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将PLY文件转换为训练数据格式')
    parser.add_argument('input_dir', type=str, help='包含PLY文件的输入目录')
    parser.add_argument('output_dir', type=str, help='输出目录')
    parser.add_argument('--boundary', type=str, default=None, help='边界粒子PLY文件路径')
    parser.add_argument('--window', type=int, default=2, help='窗口大小 (默认: 2)')
    
    args = parser.parse_args()
    
    convert_ply_sequence_to_zstd(args.input_dir, args.output_dir, args.boundary, args.window)