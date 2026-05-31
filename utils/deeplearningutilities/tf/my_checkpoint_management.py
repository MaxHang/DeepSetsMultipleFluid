import os
import tensorflow as tf
import time
from glob import glob
import re


class MyCheckpointManager:

    def __init__(self,
                 checkpoint,
                 directory,
                 keep_checkpoint_steps,
                 save_interval_minutes=30,
                 checkpoint_prefix="ckpt"):
        """Creates the checkpoint manager.

        This checkpoint manager creates and keeps checkpoints for a specific set of steps
        and creates checkpoints at specific time intervals.
        This manager removes old checkpoints that do not correspond to the set of
        checkpoint steps that shall be kept.

        checkpoint            : tf.train.Checkpoint
                                The checkpoint object

        directory             : The directory for storing checkpoints.

        keep_checkpoint_steps : list or set of integers
                                A set of steps for which checkpoints are kept.

        save_interval_minutes : int
                                The save interval in minutes. If the latest checkpoint is
                                older than this interval then a new checkpoint will be
                                created by save_if_needed()

        checkpoint_prefix     : str
                                Filename prefix for the checkpoints.
                                
        """
        self._checkpoint = checkpoint
        self._keep_checkpoint_steps = keep_checkpoint_steps
        self._directory = directory
        self._checkpoint_prefix = os.path.join(directory, checkpoint_prefix)
        self._save_interval_seconds = save_interval_minutes * 30
        self._last_save_time = time.time()
        self._all_steps_checkpoints = self.get_steps_and_checkpoints()


#         print(self._all_steps_checkpoints)

    def get_steps_and_checkpoints(self):
        """Returns a list of tuples with steps and checkpoint prefixes.
        E.g. [(100, 'train_dir/checkpoints/ckpt-100'), ..]
        """

        all_index_files = glob(self._checkpoint_prefix + '-*.index')
        step_prefix = []
        for x in all_index_files:
            match = re.match('(.*-(\d+))\.index', x)
            prefix = match.group(1)
            step = int(match.group(2))
            step_prefix.append((step, prefix))
        step_prefix.sort()
        return step_prefix

    @property
    def checkpoints(self):
        """A list of all checkpoint prefixes"""
        return [x[1] for x in self._all_steps_checkpoints]

    @property
    def latest_checkpoint(self):
        "The prefix to the latest checkpoint or None if there are no checkpoints"
        all_checkpoints = self._all_steps_checkpoints
        if all_checkpoints:
            return all_checkpoints[-1][1]
        else:
            return None

    def sweep(self):
        """Removes checkpoints that are not preserved by the keep_checkpoint_every_n_steps rule.
        Never removes the latest checkpoint.
        """
        delete_ckpts = [
            x for x in self._all_steps_checkpoints[:-1]
            if not x[0] in self._keep_checkpoint_steps
        ]
        for x in delete_ckpts:
            delete_files = [x[1] + '.index']
            delete_files.extend(glob(x[1] + '.data-?????-of-?????'))
            self._all_steps_checkpoints.remove(x)
            #print(delete_files)
            for x in delete_files:
                #print('removing', x)
                try:
                    os.remove(x)
                except:
                    print('Failed to remove file', x, flush=True)

    # def save(self, step):
    #     """Always writes a checkpoint and cleans up old checkpoints."""
    #     current_step = int(step)

    #     prefix = '{0}-{1}'.format(self._checkpoint_prefix, current_step)
    #     print('saving', prefix, flush=True)
    #     self._checkpoint.write(prefix)
    #     self._last_save_time = time.time()
    #     self._all_steps_checkpoints.append((current_step, prefix))
    #     self.sweep()

    # def save_if_needed(self, step):
    #     """Writes a checkpoint according to the parameters passed to the object ctor.

    #     This function saves a snapshot if step is in the list of checkpoints 
    #     that we want to keep or if the time passed since the last save is greater
    #     than the save interval.

    #     This function is intended to be called inside a training loop.
    #     """
    #     current_step = int(step)

    #     now = time.time()
    #     seconds_since_last_save = now - self._last_save_time

    #     if current_step in self._keep_checkpoint_steps or seconds_since_last_save > self._save_interval_seconds:
    #         self.save(current_step)

    def save(self, step):
        """Always writes a checkpoint and cleans up old checkpoints."""
        current_step = int(step)
        prefix = '{0}-{1}'.format(self._checkpoint_prefix, current_step)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print('saving', prefix, flush=True)
                self._checkpoint.write(prefix)
                self._last_save_time = time.time()
                self._all_steps_checkpoints.append((current_step, prefix))
                self.sweep()
                print(f'Checkpoint saved successfully: {prefix}', flush=True)
                break  # 成功则跳出重试循环
                
            except Exception as e:
                print(f'Failed to save checkpoint (attempt {attempt + 1}/{max_retries}')
                
                if attempt < max_retries - 1:
                    # 删除可能损坏的文件
                    corrupted_files = glob(f'{prefix}*')
                    for corrupted_file in corrupted_files:
                        try:
                            os.remove(corrupted_file)
                            print(f'Removed potentially corrupted file: {corrupted_file}')
                        except Exception as rm_err:
                            print(f'Could not remove {corrupted_file}: {rm_err}')
                    
                    time.sleep(2)  # 等待文件系统同步
                    print(f'Retrying checkpoint save...')
                else:
                    # 所有重试都失败，记录错误但不中断训练
                    print(f'All {max_retries} attempts failed for step {current_step}. Skipping this checkpoint.')
                    print(f'WARNING: Checkpoint save failed at step {current_step}, training continues...', flush=True)

    def save_if_needed(self, step):
        """Writes a checkpoint according to the parameters passed to the object ctor.

        This function saves a snapshot if step is in the list of checkpoints 
        that we want to keep or if the time passed since the last save is greater
        than the save interval.

        This function is intended to be called inside a training loop.
        """
        current_step = int(step)

        now = time.time()
        seconds_since_last_save = now - self._last_save_time

        if current_step in self._keep_checkpoint_steps or seconds_since_last_save > self._save_interval_seconds:
            try:
                self.save(current_step)
            except Exception as e:
                # save() 内部已经处理了异常，这里再捕获一次作为最后的防线
                print(f'Checkpoint save failed but training continues...', flush=True)
