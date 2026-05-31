#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from debug_utils import debug_print
from evaluate_network import evaluate_tf as evaluate
from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
import tensorflow as tf
from datetime import date
import time
from glob import glob
from collections import namedtuple
from datasets.dataset_reader_physics import read_data_train, read_data_val
import numpy as np
import argparse
import yaml

_k = 1000

TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(50 * _k, 0.001, 16)
# train_params = TrainParams(20 * _k, 0.001, 16)


def create_model(gpu_id=2, **kwargs):
    if gpu_id is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # 仅使用指定的 GPU
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                # 允许 GPU 内存动态增长
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                print(f"use GPU {gpu_id}")
            except RuntimeError as e:
                print(e)
    from models.default_tf import MyParticleNetwork
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork(**kwargs)
    return model


def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("cfg",
                        type=str,
                        help="The path to the yaml config file")
    parser.add_argument('--gpu',
                        help='指定使用的GPU ID，例如：0,1,2',
                        type=str,
                        default='0')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # the train dir stores all checkpoints and summaries. The dir name is the name of this file combined with the name of the config file
    train_dir = os.path.splitext(
        os.path.basename(__file__))[0] + '_' + os.path.splitext(
            os.path.basename(args.cfg))[0] + date.today().strftime("_%Y_%m_%d")

    train_dir = os.path.join(cfg['train_dir'], train_dir)

    print(train_dir)  # eg. train_network_tf_6kbox

    val_files = sorted(
        glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst')))
    if val_files == []:
        val_files = sorted(
            glob(os.path.join(cfg['dataset_dir'], 'test', '*.zst')))
    train_files = sorted(
        glob(os.path.join(cfg['dataset_dir'], 'train', '*.zst')))

    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=3,
                              num_workers=2,
                              **cfg.get('train_data', {}))
    data_iter = iter(dataset)

    trainer = Trainer(train_dir)

    gpu_id = int(args.gpu)
    model = create_model(gpu_id, **cfg.get('model', {}))

    boundaries = [
        25 * _k,
        30 * _k,
        35 * _k,
        40 * _k,
        45 * _k,
    ]
    lr_values = [
        train_params.base_lr * 1.0,
        train_params.base_lr * 0.5,
        train_params.base_lr * 0.25,
        train_params.base_lr * 0.125,
        train_params.base_lr * 0.5 * 0.125,
        train_params.base_lr * 0.25 * 0.125,
    ]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, lr_values)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,
                                         epsilon=1e-6)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     model=model,
                                     optimizer=optimizer)
    # checkpoint.write('aa')
    # checkpoint.save('aa')

    manager = MyCheckpointManager(checkpoint,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1,
                                            1 * _k)))

    def euclidean_distance(a, b, epsilon=1e-9):
        return tf.sqrt(tf.reduce_sum((a - b)**2, axis=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = tf.exp(-neighbor_scale * num_fluid_neighbors)
        return tf.reduce_mean(importance *
                              euclidean_distance(pr_pos, gt_pos)**gamma)

    @tf.function(experimental_relax_shapes=True)
    def train(model, batch):
        with tf.GradientTape() as tape:
            losses = []

            batch_size = train_params.batch_size
            for batch_i in range(batch_size):
                inputs = ([
                    batch['pos0'][batch_i], batch['vel0'][batch_i], None,
                    batch['box'][batch_i], batch['box_normals'][batch_i]
                ])

                pr_pos1, pr_vel1 = model(inputs)

                l = 0.5 * loss_fn(pr_pos1, batch['pos1'][batch_i],
                                  model.num_fluid_neighbors)

                inputs = (pr_pos1, pr_vel1, None, batch['box'][batch_i],
                          batch['box_normals'][batch_i])
                pr_pos2, pr_vel2 = model(inputs)

                l += 0.5 * loss_fn(pr_pos2, batch['pos2'][batch_i],
                                   model.num_fluid_neighbors)
                losses.append(l)

            losses.extend(model.losses)
            total_loss = 128 * tf.add_n(losses) / batch_size

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        checkpoint.restore(manager.latest_checkpoint)

    display_str_list = []
    while trainer.keep_training(checkpoint.step,
                                train_params.max_iter,
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):

        data_fetch_start = time.time()
        batch = next(data_iter)
        batch_tf = {}
        for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals'):
            batch_tf[k] = [tf.convert_to_tensor(x) for x in batch[k]]
        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(
            5, 'DataLatency', data_fetch_latency)

        current_loss = train(model, batch_tf)
        display_str_list = ['loss', float(current_loss)]

        if trainer.current_step % 10 == 0:
            with trainer.summary_writer.as_default():
                tf.summary.scalar('TotalLoss', current_loss)
                tf.summary.scalar('LearningRate',
                                  optimizer.lr(trainer.current_step))

        if trainer.current_step % (1 * _k) == 0:
            for k, v in evaluate(model,
                                 val_dataset,
                                 frame_skip=20,
                                 **cfg.get('evaluation', {})).items():
                with trainer.summary_writer.as_default():
                    tf.summary.scalar('eval/' + k, v)

    model.save_weights('model_weights.h5')
    if trainer.current_step == train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
