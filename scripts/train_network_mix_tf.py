#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Assuming evaluate_mix_network is adapted for the new model output
# 假设 evaluate_mix_network 已经适配了新的模型输出
from evaluate_mix_network import evaluate_tf as evaluate
from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
import tensorflow as tf
from datetime import date
import time
from glob import glob
from collections import namedtuple
# Ensure these functions can return phase_fractions and handle num_phases
# 确保这些函数能返回 phase_fractions 并处理 num_phases
from datasets.dataset_reader_h5_mix import read_data_train, read_data_val
import numpy as np
import argparse
import yaml

_k = 1000

TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
# Default values, can be overridden by cfg
# 默认值，可以被cfg覆盖
# train_params = TrainParams(50 * _k, 0.001, 16)
train_params = TrainParams(50 * _k, 0.001, 64)


def create_model(gpu_id=0, **kwargs): # Receives model_config # 接收 model_config
    if gpu_id is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                print(f"Using GPU {gpu_id}")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
                # Fallback or exit if GPU setup fails
                # 如果GPU设置失败，则回退或退出
                # For now, just print and continue (hoping CPU works or user notices)
                # 目前，仅打印并继续（希望CPU能工作或用户注意到）
    # Ensure the import path is correct
    # 确保导入路径正确
    from models.default_tf_mix import MultiPhaseParticleNetwork
    """Returns an instance of the network for training and evaluation"""
    model = MultiPhaseParticleNetwork(**kwargs)
    return model


def main():
    parser = argparse.ArgumentParser(description="Training script for Multi-Phase Fluid Network")
    parser.add_argument("cfg", type=str, help="The path to the yaml config file")
    # Changed to int
    # 改为整型
    parser.add_argument('--gpu', help='Specify GPU ID (e.g., 0, 1)', type=int, default=0)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    print(f"Training with config file: {args.cfg}")
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override train_params if specified in cfg
    # 如果在cfg中指定，则覆盖train_params
    global train_params
    if 'train_params' in cfg:
        tp_cfg = cfg['train_params']
        train_params = TrainParams(
            tp_cfg.get('max_iter', train_params.max_iter),
            tp_cfg.get('base_lr', train_params.base_lr),
            tp_cfg.get('batch_size', train_params.batch_size)
        )
    print(f"Training Parameters: {train_params}")


    train_dir_base_name = os.path.splitext(os.path.basename(__file__))[0] + \
                          '_' + os.path.splitext(os.path.basename(args.cfg))[0]
    # Shorter date format
    # 更短的日期格式
    train_dir = os.path.join(cfg['train_dir'],
                             train_dir_base_name + date.today().strftime("_%Y%m%d"))

    print(f"Train directory: {train_dir}")

    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.h5')))
    train_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'train', '*.h5')))

    if not train_files:
        print(f"Error: No training files found in {os.path.join(cfg['dataset_dir'], 'train')}")
        sys.exit(1)
    if not val_files:
        print(f"Warning: No validation files found in {os.path.join(cfg['dataset_dir'], 'valid')}")


    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    print(cfg.get('train_data'))

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=3, # For 2-step prediction # 用于2步预测
                              num_workers=cfg.get('num_workers', 2),
                              **cfg.get('train_data', {}))
    data_iter = iter(dataset)

    trainer = Trainer(train_dir)
    # Get model config from YAML
    # 从YAML获取模型配置
    model = create_model(gpu_id=args.gpu, **cfg.get('model', {}))

    try:
        print("Attempting to initialize model for summary...")
        # Example values
        # 示例值
        model.init()
    except Exception as e:
        print(f"Could not explicitly initialize model, will build on first data pass. Error: {e}")


    # Learning rate schedule
    # 学习率调度
    lr_boundaries = cfg.get('optimizer', {}).get('boundaries', [10*_k, 20*_k, 25*_k, 30*_k, 35*_k])
    lr_values_factors = cfg.get('optimizer', {}).get('lr_value_factors', [1.0, 0.5, 0.25, 0.125, 0.5 * 0.125, 0.25 * 0.125])
    lr_values_actual = [train_params.base_lr * factor for factor in lr_values_factors]

    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        lr_boundaries, lr_values_actual)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,
                                         epsilon=cfg.get('optimizer', {}).get('epsilon', 1e-6))
    # Ensure step is int64
    # 确保步骤是int64
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                     model=model,
                                     optimizer=optimizer)

    manager = MyCheckpointManager(checkpoint,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1,
                                            1 * _k)))


    def euclidean_distance(a, b, epsilon=1e-9):
        # Ensure a and b are float32 for stability with sqrt
        # 确保a和b是float32以保证sqrt的稳定性
        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1) + epsilon)

    # Allow choosing loss type
    # 允许选择损失类型
    def volume_fraction_loss(pr_vol, gt_vol, importance=None, loss_type='mse'):
        """Calculates volume fraction loss."""
        # Ensure inputs are float32
        # 确保输入是float32
        pr_vol = tf.cast(pr_vol, tf.float32)
        gt_vol = tf.cast(gt_vol, tf.float32)
        # Increased epsilon for KL
        # 为KL增加epsilon
        epsilon = 1e-7

        if loss_type == 'kl_divergence':
            # KLD expects y_true, y_pred. Reshape if necessary.
            # KLD期望y_true, y_pred。如果需要，进行重塑。
            # Assuming pr_vol and gt_vol are [batch, particles, num_phases]
            # 假设pr_vol和gt_vol是[batch, particles, num_phases]
            # We want to calculate per-particle KL divergence then average
            # 我们想要计算每个粒子的KL散度然后取平均
            kl_div = tf.keras.losses.KLDivergence()
            # Flatten batch & particles
            # 展平批次和粒子维度
            num_particles_dim = tf.shape(gt_vol)[-2] # Assuming second to last is num_particles # 假设倒数第二个是num_particles
            error_per_particle = kl_div(
                tf.reshape(gt_vol, [-1, model.num_phases]),
                tf.reshape(pr_vol, [-1, model.num_phases])
            )
            # Reshape back
            # 重塑回去
            error = tf.reshape(error_per_particle, [-1, num_particles_dim])

        elif loss_type == 'mse':
            error = tf.reduce_mean(tf.square(pr_vol - gt_vol), axis=-1)
        elif loss_type == 'mae':
            error = tf.reduce_mean(tf.abs(pr_vol - gt_vol), axis=-1)
        else: # Your combined loss # 你的组合损失
            l1_loss = tf.abs(pr_vol - gt_vol)
            gt_safe = tf.clip_by_value(gt_vol, epsilon, 1.0 - epsilon)
            pr_safe = tf.clip_by_value(pr_vol, epsilon, 1.0 - epsilon)
            # Careful with log(0) or log(>1) if not properly clipped for (1-p) terms
            # 小心log(0)或log(>1)，如果(1-p)项没有正确裁剪
            term1 = gt_safe * tf.math.log(gt_safe / pr_safe)
            term2 = (1.0 - gt_safe) * tf.math.log((1.0 - gt_safe) / (1.0 - pr_safe))
            # This is per-phase, then sum
            # 这是每相的，然后求和
            kl_like_loss = term1 + term2
            
            # Sum over phases
            # 对各相求和
            combined_loss = tf.reduce_sum(l1_loss + 0.1 * kl_like_loss, axis=-1)
            # error is now [batch, particles]
            # error现在是[batch, particles]
            error = combined_loss

        if importance is not None:
            importance = tf.cast(importance, tf.float32)
            return tf.reduce_mean(importance * error)
        return tf.reduce_mean(error)

    # Get loss weights from config
    # 从配置中获取损失权重
    loss_weights = cfg.get('loss_weights', {'pos': 1.0, 'vol': 1.0, 'gamma': 0.5})
    # e.g., 'mse', 'kl_divergence', 'combined'
    # 例如 'mse', 'kl_divergence', 'combined'
    vf_loss_type = cfg.get('loss_vf_type', 'mse')

    def loss_fn(pr_pos, gt_pos, pr_vol=None, gt_vol=None, num_fluid_neighbors=None):
        gamma = tf.cast(loss_weights.get('gamma', 0.5), tf.float32)
        # Default neighbor_scale if num_fluid_neighbors is None or not effective
        # 如果num_fluid_neighbors为None或无效，则使用默认的neighbor_scale
        neighbor_scale_val = 1.0 / 40.0
        
        if num_fluid_neighbors is not None and tf.size(num_fluid_neighbors) > 0 :
            importance = tf.exp(-neighbor_scale_val * tf.cast(num_fluid_neighbors, tf.float32))
        else:
            # If num_fluid_neighbors is None, create importance of ones.
            # 如果num_fluid_neighbors为None，则创建全为1的重要性因子。
            # The per-batch-item processing in train() means num_fluid_neighbors is [num_particles]
            # train()中的每批次项处理意味着num_fluid_neighbors是[num_particles]
            # So importance should also be [num_particles]
            # 所以importance也应该是[num_particles]
             dummy_particle_dim_shape = tf.shape(pr_pos)[0] # Assuming pr_pos is [particles_in_sample, 3] # 假设pr_pos是[样本中的粒子数, 3]
             importance = tf.ones(shape=(dummy_particle_dim_shape,), dtype=tf.float32)


        pos_loss_val = tf.reduce_mean(importance * tf.pow(euclidean_distance(pr_pos, gt_pos), gamma))

        total_loss = loss_weights.get('pos', 1.0) * pos_loss_val

        if model.num_phases > 1 and pr_vol is not None and gt_vol is not None:
            # For now, assume gt_vol is also [particles, num_phases]
            # 目前，假设gt_vol也是[particles, num_phases]
            vol_loss_val = volume_fraction_loss(pr_vol, gt_vol, importance, loss_type=vf_loss_type)
            total_loss += loss_weights.get('vol', 1.0) * vol_loss_val
        
        # Add stability loss here if implemented
        # 如果实现，在此处添加稳定性损失
        # stability_loss_val = calculate_stability_loss(...)
        # total_loss += loss_weights.get('stability', 0.1) * stability_loss_val

        return total_loss


    @tf.function(experimental_relax_shapes=True)
    # Renamed for clarity
    # 为清晰起见重命名
    def train_step(model_instance, optimizer_instance, current_batch):
        with tf.GradientTape() as tape:
            # Accumulate loss for each item in the batch
            # 累积批次中每个项目的损失
            accumulated_losses = []

            # Iterate over each sample in the batch (as loaded by dataset reader)
            # 遍历批次中的每个样本（由数据集读取器加载）
            for i in range(train_params.batch_size):
                pos0 = current_batch['pos0'][i]
                vel0 = current_batch['vel0'][i]
                box_pos_sample = current_batch['box'][i]
                box_normals_sample = current_batch['box_normals'][i]

                gt_pos1 = current_batch['pos1'][i]
                # For 2-step prediction
                # 用于2步预测
                gt_pos2 = current_batch['pos2'][i]

                # Phase fractions - ensure shape is [particles, num_phases]
                # 相分数 - 确保形状为[particles, num_phases]
                current_vf0 = None
                gt_vf1 = None
                gt_vf2 = None
                if model_instance.num_phases > 1:
                    if 'phase_fractions0' in current_batch and current_batch['phase_fractions0']:
                        current_vf0 = current_batch['phase_fractions0'][i]
                    if 'phase_fractions1' in current_batch and current_batch['phase_fractions1']:
                        gt_vf1 = current_batch['phase_fractions1'][i]
                    if 'phase_fractions2' in current_batch and current_batch['phase_fractions2']:
                        gt_vf2 = current_batch['phase_fractions2'][i]
                    
                    # Fallback if GT VFs for next steps are missing (e.g. for stability loss later)
                    # 如果后续步骤的GT VF缺失，则回退（例如，用于稍后的稳定性损失）
                    if gt_vf1 is None and current_vf0 is not None: gt_vf1 = current_vf0 
                    if gt_vf2 is None and gt_vf1 is not None: gt_vf2 = gt_vf1


                # Cd and Cf for this sample
                # 此样本的Cd和Cf
                # Default if not in batch
                # 如果不在批次中则使用默认值
                cd_sample = current_batch.get('cd', 0.5)
                if isinstance(cd_sample, (list, tf.Tensor, np.ndarray)):
                    cd_val = tf.cast(cd_sample[i], dtype=tf.float32) if len(cd_sample) > i else tf.constant(0.5, dtype=tf.float32)
                else:
                    cd_val = tf.cast(cd_sample, dtype=tf.float32)

                cf_sample = current_batch.get('cf', 0.5) # Default if not in batch
                if isinstance(cf_sample, (list, tf.Tensor, np.ndarray)):
                    cf_val = tf.cast(cf_sample[i], dtype=tf.float32) if len(cf_sample) > i else tf.constant(0.5, dtype=tf.float32)
                else:
                    cf_val = tf.cast(cf_sample, dtype=tf.float32)


                # --- First prediction step ---
                # --- 第一个预测步骤 ---
                inputs1 = (pos0, vel0, current_vf0, box_pos_sample, box_normals_sample)
                pr_pos1, pr_vel1, pr_vf1 = model_instance(inputs1, training=True, cd=cd_val, cf=cf_val)
                
                loss1 = loss_fn(pr_pos1, gt_pos1, pr_vf1, gt_vf1, model_instance.num_fluid_neighbors)
                
                # --- Second prediction step ---
                # --- 第二个预测步骤 ---
                # Use predicted as input
                # 使用预测值作为输入
                inputs2 = (pr_pos1, pr_vel1, pr_vf1, box_pos_sample, box_normals_sample)
                pr_pos2, pr_vel2, pr_vf2 = model_instance(inputs2, training=True, cd=cd_val, cf=cf_val)

                loss2 = loss_fn(pr_pos2, gt_pos2, pr_vf2, gt_vf2, model_instance.num_fluid_neighbors)
                
                accumulated_losses.append(0.5 * loss1 + 0.5 * loss2)

            accumulated_losses.extend(model_instance.losses)

            # Average loss over the batch
            # 对批次损失取平均
            # Loss scaling factor, can be tuned or made a config parameter
            # 损失缩放因子，可以调整或设为配置参数
            loss_scaling_factor = cfg.get('loss_scaling_factor', 1.0)
            batch_total_loss = loss_scaling_factor * tf.add_n(accumulated_losses) / float(train_params.batch_size)


            grads = tape.gradient(batch_total_loss, model_instance.trainable_variables)
            optimizer_instance.apply_gradients(zip(grads, model_instance.trainable_variables))
        
        return batch_total_loss


    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        checkpoint.restore(manager.latest_checkpoint)


    display_str_list = []
    # Main training loop
    # 主训练循环
    while trainer.keep_training(checkpoint.step,
                                train_params.max_iter,
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):
        data_fetch_start = time.time()
        batch_from_dataset = next(data_iter)
        
        # Convert numpy arrays from dataset to TensorFlow tensors
        # 将数据集中的NumPy数组转换为TensorFlow张量
        batch_tf = {}
        # Standard features
        # 标准特征
        for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals'):
            if k in batch_from_dataset:
                batch_tf[k] = [tf.convert_to_tensor(x, dtype=tf.float32) for x in batch_from_dataset[k]]
        
        # Phase fractions
        # 相分数
        # Ensure phase_fractionsX has shape [particles, num_phases] for each item in batch list
        # 确保批次列表中每个项目的phase_fractionsX形状为[particles, num_phases]
        num_model_phases = model.num_phases
        for k_vf_idx in range(3): # For phase_fractions0, 1, 2 # 对于phase_fractions0, 1, 2
            k_vf = f'phase_fractions{k_vf_idx}'
            if k_vf in batch_from_dataset:
                # Convert to tensor and ensure correct shape [particles, num_phases]
                # 转换为张量并确保正确的形状[particles, num_phases]
                processed_vf_list = []
                for vf_sample_np in batch_from_dataset[k_vf]:
                    vf_sample_tf = tf.convert_to_tensor(vf_sample_np, dtype=tf.float32)
                    # If dataset provides N-1 phases, construct the Nth phase
                    # 如果数据集提供N-1个相，则构造第N个相
                    if vf_sample_tf.shape[-1] == num_model_phases - 1 and num_model_phases > 1:
                        last_phase = 1.0 - tf.reduce_sum(vf_sample_tf, axis=-1, keepdims=True)
                        # Ensure valid
                        # 确保有效
                        last_phase = tf.clip_by_value(last_phase, 0.0, 1.0)
                        vf_sample_tf = tf.concat([vf_sample_tf, last_phase], axis=-1)
                    elif vf_sample_tf.shape[-1] != num_model_phases and num_model_phases > 1:
                         raise ValueError(f"Shape mismatch for {k_vf}: expected {num_model_phases} phases, "
                                          f"got {vf_sample_tf.shape[-1]}. Sample shape: {vf_sample_tf.shape}")
                    processed_vf_list.append(vf_sample_tf)
                batch_tf[k_vf] = processed_vf_list
        
        # Cd and Cf - assuming they are single scalar values per batch or lists of scalars
        # Cd和Cf - 假设它们是每批次的单个标量值或标量列表
        if 'cd' in batch_from_dataset:
            # batch_tf['cd'] will be used by train_step which expects a scalar or list of scalars
            # batch_tf['cd']将由train_step使用，它期望一个标量或标量列表
            batch_tf['cd'] = batch_from_dataset['cd']
        if 'cf' in batch_from_dataset:
            batch_tf['cf'] = batch_from_dataset['cf']

        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train_step(model, optimizer, batch_tf)
        # Update display string
        # 更新显示字符串
        display_str_list = ['loss', float(current_loss)]

        if trainer.current_step % 10 == 0:
            with trainer.summary_writer.as_default():
                tf.summary.scalar('TotalLoss', current_loss)
                tf.summary.scalar('LearningRate', optimizer.lr(trainer.current_step))

        if trainer.current_step % (1 * _k) == 0:
            # Only evaluate if validation files exist
            # 仅当验证文件存在时才进行评估
            if val_files:
                eval_results = evaluate(model,
                                        val_dataset, # val_dataset reader should also handle num_phases # val_dataset读取器也应处理num_phases
                                        frame_skip=cfg.get('evaluation', {}).get('frame_skip', 20),
                                        **cfg.get('evaluation', {}))
                with trainer.summary_writer.as_default():
                    for k_eval, v_eval in eval_results.items():
                        tf.summary.scalar('eval/' + k_eval, v_eval)
            else:
                print(f"Step {trainer.current_step}: Skipping evaluation as no validation files are present.")


    model_weights_name = "model_weights" + \
        date.today().strftime("_%Y_%m_%d") + ".h5"
    model_weights_save_path = os.path.join(train_dir, model_weights_name)
    model.save_weights(model_weights_save_path)
    print(f"Final model weights saved to: {model_weights_save_path}")

    if trainer.current_step >= train_params.max_iter:
        print("Training finished.")
        return trainer.STATUS_TRAINING_FINISHED
    else:
        print("Training stopped before max_iter.")
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    # It's good practice to set this for TensorFlow if using multiprocessing,
    # though 'spawn' is often default on non-Linux.
    # 'fork' can cause issues with CUDA in child processes.
    # 如果使用多处理，为TensorFlow设置这个是个好习惯，
    # 尽管'spawn'在非Linux系统上通常是默认的。
    # 'fork'可能会在子进程中导致CUDA问题。
    try:
        # force=True if already set
        # 如果已设置则force=True
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set or 'spawn' not supported.")

    sys.exit(main())