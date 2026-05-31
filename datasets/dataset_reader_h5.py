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

class H5PhysicsDataFlow(dataflow.RNGDataFlow):
    """H5格式物理模拟数据的数据流
    """
    
    def __init__(self, files, random_rotation=False, shuffle=False, window=2):
        """初始化数据流
        
        Args:
            files: H5文件路径列表
            random_rotation: 是否随机旋转数据
            shuffle: 是否打乱数据
            window: 时间窗口大小
        """
        if not len(files):
            raise Exception("The file list cannot be empty")
        if window < 1:
            raise Exception(f"window must >=1, but get {window}")
        
        self.files = files
        self.random_rotation = random_rotation
        self.shuffle = shuffle
        self.window = window
        
        # 收集每个文件中的可用窗口
        self.all_windows = []
        
        for file_path in files:
            with h5py.File(file_path, 'r') as h5f:
                frames_group = h5f['frames']
                frame_ids = sorted([int(k) for k in frames_group.keys()])
                
                # 检查是否有足够的连续帧
                if len(frame_ids) < window:
                    continue
                
                # 收集所有可能的窗口
                for start_idx in range(len(frame_ids) - window + 1):
                    window_frames = frame_ids[start_idx:start_idx+window]
                    self.all_windows.append({
                        'file': file_path,
                        'frames': window_frames
                    })
    
    def __len__(self):
        return len(self.all_windows)
    
    def __iter__(self):
        """迭代数据集"""
        window_indices = np.arange(len(self.all_windows))
        if self.shuffle:
            self.rng.shuffle(window_indices)
        
        for win_idx in window_indices:
            window_info = self.all_windows[win_idx]
            file_path = window_info['file']
            frame_ids = window_info['frames']
            
            with h5py.File(file_path, 'r') as h5f:
                # 获取边界数据
                if 'box' in h5f:
                    box = h5f['box'][:]
                    box_normals = h5f['box_normals'][:]
                else:
                    # 创建空边界
                    box = np.zeros((0, 3), dtype=np.float32)
                    box_normals = np.zeros((0, 3), dtype=np.float32)
                
                # 准备随机旋转矩阵
                if self.random_rotation:
                    angle_rad = self.rng.uniform(0, 2 * np.pi)
                    s = np.sin(angle_rad)
                    c = np.cos(angle_rad)
                    rand_R = np.array([c, 0, s, 0, 1, 0, -s, 0, c],
                                    dtype=np.float32).reshape((3, 3))
                else:
                    rand_R = None
                
                # 创建样本
                if self.random_rotation:
                    sample = {
                        'box': np.matmul(box, rand_R),
                        'box_normals': np.matmul(box_normals, rand_R)
                    }
                else:
                    sample = {'box': box, 'box_normals': box_normals}
                
                # 读取CD和CF参数
                if 'cd' in h5f.attrs:
                    sample['cd'] = np.float32(h5f.attrs['cd'])
                else:
                    sample['cd'] = np.float32(0.5)
                    
                if 'cf' in h5f.attrs:
                    sample['cf'] = np.float32(h5f.attrs['cf'])
                else:
                    sample['cf'] = np.float32(0.5)
                
                # 读取每个时间步的数据
                for time_i, frame_id in enumerate(frame_ids):
                    frame_group = h5f[f'frames/{frame_id}']
                    
                    # 读取位置和速度、相分数
                    pos = frame_group['pos'][:]
                    vel = frame_group['vel'][:]

                    # 读取并转换相分数数据
                    if 'phase_fractions' in frame_group:
                        full_phase_fractions = frame_group['phase_fractions'][:]
                        
                        # 检查数据维度
                        if len(full_phase_fractions.shape) == 2:
                            # 获取相数量
                            num_phases = full_phase_fractions.shape[1]
                            
                            if num_phases > 1:
                                # 只保留前n-1个相的数据
                                phase_fractions = full_phase_fractions[:, :-1]
                            else:
                                # 单相流体情况，保持原样
                                phase_fractions = full_phase_fractions
                        else:
                            # 如果是一维数组，说明只有一个相，保持原样
                            phase_fractions = full_phase_fractions
                    else:
                        # 如果没有相分数数据，默认为单相(全为1)
                        phase_fractions = np.ones((len(pos), 1), dtype=np.float32)
                    
                    # 如果需要旋转
                    if self.random_rotation:
                        sample[f'pos{time_i}'] = np.matmul(pos, rand_R)
                        sample[f'vel{time_i}'] = np.matmul(vel, rand_R)
                        # 相分数是标量，不需要旋转，直接赋值
                        sample[f'phase_fractions{time_i}'] = phase_fractions
                    else:
                        sample[f'pos{time_i}'] = pos
                        sample[f'vel{time_i}'] = vel
                        sample[f'phase_fractions{time_i}'] = phase_fractions
                    
                    # 读取其他属性
                    for k in ('m', 'viscosity'):
                        if k in frame_group:
                            sample[f'{k}{time_i}'] = frame_group[k][:]
                        else:
                            # 创建默认值
                            if k == 'm':
                                sample[f'{k}{time_i}'] = np.ones(len(pos), dtype=np.float32)
                            elif k == 'viscosity':
                                sample[f'{k}{time_i}'] = np.ones(len(pos), dtype=np.float32) * 0.01
                    
                    # 读取元数据
                    sample[f'frame_id{time_i}'] = frame_group.attrs['frame_id']
                    sample[f'scene_id{time_i}'] = frame_group.attrs['scene_id']
                
                yield sample


def read_data(files=None,
              batch_size=1,
              window=2,
              random_rotation=False,
              repeat=False,
              shuffle_buffer=None,
              num_workers=1,
              cache_data=False):
    """创建H5数据读取管道
    
    兼容原始PhysicsSimDataFlow接口
    """
    print(f"read data files: {files[0:5]}" + ('...' if len(files) > 5 else ''))

    # 创建数据流
    df = H5PhysicsDataFlow(
        files=files,
        random_rotation=random_rotation,
        shuffle=shuffle_buffer is not None,
        window=window,
    )

    if repeat:
        df = dataflow.RepeatedData(df, -1)

    if shuffle_buffer:
        df = dataflow.LocallyShuffleData(df, shuffle_buffer)

    if num_workers > 1:
        df = dataflow.MultiProcessRunnerZMQ(df, num_proc=num_workers)

    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)

    if cache_data:
        df = dataflow.CacheData(df)

    df.reset_state()
    return df


def read_data_train(files, batch_size, random_rotation=True, **kwargs):
    """训练数据读取
    
    兼容原始read_data_train接口
    """
    return read_data(
        files=files,
        batch_size=batch_size,
        random_rotation=random_rotation,
        repeat=True,
        shuffle_buffer=512,
        **kwargs
    )


def read_data_val(files, **kwargs):
    """验证数据读取
    
    兼容原始read_data_val接口
    """
    return read_data(
        files=files,
        batch_size=1,
        repeat=False,
        shuffle_buffer=None,
        num_workers=1,
        **kwargs
    )
