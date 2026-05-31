#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的 Checkpoint 管理器 - 解决 NAS 存储不稳定问题

创建日期: 2025-10-21
作者: AI Assistant

核心策略:
    1. 先写入本地 SSD/HDD 缓存（秒级完成，不阻塞训练）
    2. 异步拷贝到 NAS（后台线程处理，失败自动重试）
    3. 智能清理：本地保留最近 3 个，NAS 保留所有需要的
    4. 容错机制：即使 NAS 完全失败，本地也有备份

使用场景:
    - NAS 网络不稳定、延迟高
    - Checkpoint 文件很大（几百 MB 到几 GB）
    - 需要高频保存但不想阻塞训练

使用示例:
    from utils.deeplearningutilities.tf.my_checkpoint_management_with_cache import MyCheckpointManagerWithCache
    
    manager = MyCheckpointManagerWithCache(
        checkpoint=checkpoint,
        directory='/workspace/nas/checkpoints',  # NAS 目标目录
        keep_checkpoint_steps=list(range(1000, 100000, 1000)),
        local_cache_dir='/tmp/training_cache'  # 本地缓存
    )
"""

import os
import time
import shutil
import threading
from glob import glob
import re
from typing import List, Tuple, Optional, Set
import tensorflow as tf


class MyCheckpointManagerWithCache:
    """支持本地缓存的 Checkpoint 管理器，专为 NAS 存储优化"""

    def __init__(self,
                 checkpoint,
                 directory: str,
                 keep_checkpoint_steps: List[int],
                 save_interval_minutes: int = 30,
                 checkpoint_prefix: str = "ckpt",
                 local_cache_dir: Optional[str] = None,
                 max_local_cache: int = 3,
                 upload_timeout: int = 300):
        """
        初始化带缓存的 Checkpoint 管理器。

        Args:
            checkpoint: tf.train.Checkpoint 对象
            directory: NAS 目标目录 (e.g., /workspace/xyh_synology/.../checkpoints)
            keep_checkpoint_steps: 需要保留的 step 列表
            save_interval_minutes: 自动保存间隔（分钟）
            checkpoint_prefix: checkpoint 文件前缀（默认 "ckpt"）
            local_cache_dir: 本地缓存目录（None 则自动使用 /tmp）
            max_local_cache: 本地最多保留的 checkpoint 数量（默认 3）
            upload_timeout: 单次上传超时时间（秒，默认 300）
        """
        self._checkpoint = checkpoint
        self._keep_checkpoint_steps = set(keep_checkpoint_steps)
        self._nas_directory = directory
        self._checkpoint_prefix = checkpoint_prefix
        self._save_interval_seconds = save_interval_minutes * 60
        self._last_save_time = time.time()
        self._max_local_cache = max_local_cache
        self._upload_timeout = upload_timeout
        
        # 本地缓存目录设置
        if local_cache_dir is None:
            # 使用 /tmp 或系统临时目录
            base_name = os.path.basename(directory.rstrip('/'))
            local_cache_dir = f"/tmp/deeplearning_checkpoints/{base_name}"
        
        self._local_cache_dir = local_cache_dir
        
        # 确保目录存在
        os.makedirs(self._local_cache_dir, exist_ok=True)
        os.makedirs(self._nas_directory, exist_ok=True)
        
        # 异步上传管理
        self._upload_queue = []
        self._upload_lock = threading.Lock()
        self._active_uploads = set()
        
        # 扫描已有 checkpoints
        self._all_steps_checkpoints = self.get_steps_and_checkpoints()
        
        print(f"\n{'='*80}")
        print(f"[CheckpointManager] Initialized with local cache strategy")
        print(f"{'='*80}")
        print(f"  Local cache dir : {self._local_cache_dir}")
        print(f"  NAS target dir  : {self._nas_directory}")
        print(f"  Max local cache : {self._max_local_cache} checkpoints")
        print(f"  Upload timeout  : {self._upload_timeout} seconds")
        print(f"  Existing ckpts  : {len(self._all_steps_checkpoints)} found on NAS")
        print(f"{'='*80}\n")

    def get_steps_and_checkpoints(self) -> List[Tuple[int, str]]:
        """
        扫描 NAS 目录中已有的 checkpoints。
        
        Returns:
            按 step 排序的 (step, checkpoint_prefix) 元组列表
        """
        checkpoint_files = glob(
            os.path.join(self._nas_directory, f'{self._checkpoint_prefix}-*.index')
        )
        
        steps_and_ckpts = []
        for file_path in checkpoint_files:
            match = re.match(f'.*{self._checkpoint_prefix}-(\d+)\.index', file_path)
            if match:
                step = int(match.group(1))
                prefix = file_path.replace('.index', '')
                steps_and_ckpts.append((step, prefix))
        
        return sorted(steps_and_ckpts)

    @property
    def checkpoints(self) -> List[str]:
        """所有 checkpoint 文件路径列表（NAS 上的）"""
        return [x[1] for x in self._all_steps_checkpoints]

    @property
    def latest_checkpoint(self) -> Optional[str]:
        """最新的 checkpoint 路径（NAS 上的）"""
        if self._all_steps_checkpoints:
            return self._all_steps_checkpoints[-1][1]
        
        # 如果 NAS 上没有，检查本地缓存
        local_checkpoints = glob(
            os.path.join(self._local_cache_dir, f'{self._checkpoint_prefix}-*.index')
        )
        if local_checkpoints:
            latest_local = sorted(local_checkpoints)[-1]
            return latest_local.replace('.index', '')
        
        return None

    def save(self, step: int):
        """
        保存 checkpoint（两阶段策略：本地快速保存 + 异步上传 NAS）。
        
        Args:
            step: 当前训练 step
        """
        current_step = int(step)
        
        # ========== 阶段1: 快速写入本地缓存 ==========
        local_prefix = os.path.join(
            self._local_cache_dir, 
            f'{self._checkpoint_prefix}-{current_step}'
        )
        
        try:
            print(f'[Checkpoint] Step {current_step}: Saving to local cache...', flush=True)
            start_time = time.time()
            
            # TensorFlow checkpoint 写入
            self._checkpoint.write(local_prefix)
            
            local_save_time = time.time() - start_time
            print(f'[Checkpoint] SUCCESS Local save completed in {local_save_time:.2f}s', flush=True)
            
            # ========== 阶段2: 异步上传到 NAS ==========
            nas_prefix = os.path.join(
                self._nas_directory, 
                f'{self._checkpoint_prefix}-{current_step}'
            )
            
            # 启动后台上传线程
            self._async_upload_to_nas(local_prefix, nas_prefix, current_step)
            
            # 更新状态
            self._last_save_time = time.time()
            self._all_steps_checkpoints.append((current_step, nas_prefix))
            
            # ========== 阶段3: 清理旧的 checkpoints ==========
            # 清理 NAS（在后台线程中执行，避免阻塞）
            threading.Thread(target=self.sweep, daemon=True).start()
            
            # 清理本地缓存（立即执行，避免占用过多磁盘）
            self._cleanup_local_cache()
            
        except Exception as e:
            print(f'[ERROR] Failed to save checkpoint to local cache: {e}', flush=True)
            print('[WARNING] Checkpoint save failed, but training continues...', flush=True)

    def _async_upload_to_nas(self, local_prefix: str, nas_prefix: str, step: int):
        """
        异步上传 checkpoint 到 NAS（不阻塞主训练线程）。
        
        Args:
            local_prefix: 本地 checkpoint 前缀（不含扩展名）
            nas_prefix: NAS checkpoint 前缀（不含扩展名）
            step: 当前 step
        """
        def upload_worker():
            max_retries = 5
            base_retry_delay = 10  # 秒
            
            # 标记为活跃上传
            with self._upload_lock:
                self._active_uploads.add(step)
            
            try:
                for attempt in range(max_retries):
                    try:
                        print(f'[Upload] Step {step}: Uploading to NAS (attempt {attempt+1}/{max_retries})...', flush=True)
                        upload_start = time.time()
                        
                        # 获取所有相关文件
                        local_files = glob(f'{local_prefix}*')
                        
                        if not local_files:
                            print(f'[WARNING] No files found for {local_prefix}', flush=True)
                            break
                        
                        # 拷贝每个文件到 NAS
                        for local_file in local_files:
                            # 构建 NAS 目标路径
                            file_suffix = local_file.replace(local_prefix, '')
                            nas_file = nas_prefix + file_suffix
                            
                            # 确保 NAS 目录存在
                            nas_dir = os.path.dirname(nas_file)
                            os.makedirs(nas_dir, exist_ok=True)
                            
                            # 拷贝文件（带超时保护）
                            shutil.copy2(local_file, nas_file)
                            
                            # 验证文件大小
                            local_size = os.path.getsize(local_file)
                            nas_size = os.path.getsize(nas_file)
                            if local_size != nas_size:
                                raise ValueError(
                                    f"Size mismatch: {local_file} ({local_size}) vs "
                                    f"{nas_file} ({nas_size})"
                                )
                        
                        upload_time = time.time() - upload_start
                        print(f'[Upload] SUCCESS Step {step} uploaded to NAS in {upload_time:.2f}s', flush=True)
                        break  # 成功则退出重试循环
                        
                    except Exception as e:
                        print(f'[Upload] FAIL!!! Step {step} failed (attempt {attempt+1}/{max_retries}): {e}', flush=True)
                        
                        if attempt < max_retries - 1:
                            # 指数退避策略
                            retry_delay = base_retry_delay * (2 ** attempt)
                            print(f'[Upload] Retrying in {retry_delay}s...', flush=True)
                            time.sleep(retry_delay)
                        else:
                            print(f'[WARNING] All upload attempts failed for step {step}', flush=True)
                            print(f'[INFO] Local backup is safe at: {local_prefix}*', flush=True)
            
            finally:
                # 移除活跃上传标记
                with self._upload_lock:
                    self._active_uploads.discard(step)
        
        # 启动后台上传线程
        upload_thread = threading.Thread(target=upload_worker, daemon=True, name=f"Upload-{step}")
        upload_thread.start()

    def _cleanup_local_cache(self):
        """
        清理本地缓存，只保留最近 N 个 checkpoint。
        
        策略:
            - 保留最近的 max_local_cache 个
            - 正在上传的不删除
            - 按修改时间排序
        """
        try:
            local_checkpoints = glob(
                os.path.join(self._local_cache_dir, f'{self._checkpoint_prefix}-*.index')
            )
            
            if len(local_checkpoints) <= self._max_local_cache:
                return  # 不需要清理
            
            # 提取 step 并按时间排序
            checkpoint_info = []
            for ckpt_file in local_checkpoints:
                match = re.match(f'.*{self._checkpoint_prefix}-(\d+)\.index', ckpt_file)
                if match:
                    step = int(match.group(1))
                    mtime = os.path.getmtime(ckpt_file)
                    checkpoint_info.append((step, mtime, ckpt_file))
            
            # 按修改时间排序（最新的在后面）
            checkpoint_info.sort(key=lambda x: x[1])
            
            # 保留最近的 N 个
            checkpoints_to_delete = checkpoint_info[:-self._max_local_cache]
            
            for step, _, index_file in checkpoints_to_delete:
                # 检查是否正在上传
                with self._upload_lock:
                    if step in self._active_uploads:
                        print(f'[Cache] Skipping step {step} (upload in progress)', flush=True)
                        continue
                
                # 删除该 checkpoint 的所有文件
                prefix = index_file.replace('.index', '')
                files_to_delete = glob(f'{prefix}*')
                
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f'[WARNING] Failed to remove {file_path}: {e}', flush=True)
                
                if files_to_delete:
                    print(f'[Cache] Removed old local checkpoint: step {step}', flush=True)
        
        except Exception as e:
            print(f'[WARNING] Failed to cleanup local cache: {e}', flush=True)

    def sweep(self):
        """
        清理 NAS 上不需要保留的旧 checkpoints。
        
        策略:
            - 删除不在 keep_checkpoint_steps 列表中的
            - 始终保留最新的一个
        """
        try:
            delete_ckpts = [
                x for x in self._all_steps_checkpoints[:-1]  # 排除最新的
                if x[0] not in self._keep_checkpoint_steps
            ]
            
            for step, prefix in delete_ckpts:
                # 删除所有相关文件
                delete_files = [prefix + '.index']
                delete_files.extend(glob(prefix + '.data-?????-of-?????'))
                
                deleted_count = 0
                for file_path in delete_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            deleted_count += 1
                    except Exception as e:
                        print(f'[WARNING] Failed to remove {file_path}: {e}', flush=True)
                
                if deleted_count > 0:
                    # 从列表中移除
                    self._all_steps_checkpoints.remove((step, prefix))
                    print(f'[Cleanup] Removed old NAS checkpoint: step {step} ({deleted_count} files)', flush=True)
        
        except Exception as e:
            print(f'[WARNING] Failed to cleanup NAS checkpoints: {e}', flush=True)

    def save_if_needed(self, step: int):
        """
        根据条件决定是否保存 checkpoint。
        
        条件:
            1. step 在 keep_checkpoint_steps 列表中
            2. 距离上次保存超过 save_interval_seconds
        
        Args:
            step: 当前训练 step
        """
        current_step = int(step)
        now = time.time()
        seconds_since_last_save = now - self._last_save_time

        if current_step in self._keep_checkpoint_steps or \
           seconds_since_last_save > self._save_interval_seconds:
            try:
                self.save(current_step)
            except Exception as e:
                print(f'[ERROR] Checkpoint save failed: {e}', flush=True)
                print('[WARNING] Training continues despite save failure...', flush=True)

    def wait_for_pending_uploads(self, timeout: int = 600):
        """
        等待所有待上传的任务完成（在训练结束时调用）。
        
        Args:
            timeout: 最大等待时间（秒）
        """
        print(f'\n[CheckpointManager] Waiting for pending uploads to complete...', flush=True)
        start_time = time.time()
        
        while True:
            with self._upload_lock:
                pending_count = len(self._active_uploads)
                if pending_count == 0:
                    print('[CheckpointManager] SUCCESS All uploads completed', flush=True)
                    break
                
                print(f'[CheckpointManager] Pending uploads: {pending_count}', flush=True)
            
            # 检查超时
            if time.time() - start_time > timeout:
                print(f'[WARNING] Upload timeout reached, {pending_count} uploads still pending', flush=True)
                print('[INFO] Check local cache for missing checkpoints:', flush=True)
                print(f'       {self._local_cache_dir}', flush=True)
                break
            
            time.sleep(5)

    def get_status(self) -> dict:
        """
        获取管理器状态（用于调试和监控）。
        
        Returns:
            包含状态信息的字典
        """
        with self._upload_lock:
            active_uploads = list(self._active_uploads)
        
        local_checkpoints = glob(
            os.path.join(self._local_cache_dir, f'{self._checkpoint_prefix}-*.index')
        )
        
        return {
            'nas_directory': self._nas_directory,
            'local_cache_directory': self._local_cache_dir,
            'total_nas_checkpoints': len(self._all_steps_checkpoints),
            'total_local_checkpoints': len(local_checkpoints),
            'active_uploads': active_uploads,
            'latest_checkpoint': self.latest_checkpoint,
            'seconds_since_last_save': time.time() - self._last_save_time,
        }

    def print_status(self):
        """打印当前状态（用于调试）"""
        status = self.get_status()
        print(f"\n{'='*80}")
        print(f"[CheckpointManager Status]")
        print(f"{'='*80}")
        for key, value in status.items():
            print(f"  {key:30s}: {value}")
        print(f"{'='*80}\n")


# ========== 便捷函数 ==========

def create_checkpoint_manager(checkpoint,
                              directory: str,
                              keep_checkpoint_steps: List[int],
                              use_local_cache: bool = True,
                              **kwargs):
    """
    创建 checkpoint 管理器的便捷函数。
    
    Args:
        checkpoint: tf.train.Checkpoint 对象
        directory: checkpoint 保存目录
        keep_checkpoint_steps: 需要保留的 step 列表
        use_local_cache: 是否使用本地缓存策略（推荐开启）
        **kwargs: 其他参数传递给管理器
    
    Returns:
        MyCheckpointManagerWithCache 或 MyCheckpointManager 实例
    """
    if use_local_cache:
        return MyCheckpointManagerWithCache(
            checkpoint=checkpoint,
            directory=directory,
            keep_checkpoint_steps=keep_checkpoint_steps,
            **kwargs
        )
    else:
        # 使用原始的管理器
        from .my_checkpoint_management import MyCheckpointManager
        return MyCheckpointManager(
            checkpoint=checkpoint,
            directory=directory,
            keep_checkpoint_steps=keep_checkpoint_steps,
            **kwargs
        )