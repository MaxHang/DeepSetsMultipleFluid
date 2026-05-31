#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
import argparse, yaml, time
from glob import glob
from datetime import datetime, date
from collections import namedtuple

from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
from models.default_tf_mix_separate_pos_phase_v13 import MultiPhaseParticleNetwork
from datasets.dataset_reader_h5_mix import read_data_train, read_data_val
from scripts.evaluate_mix_spearate_pos_phase_v1 import evaluate_tf as evaluate

tf.debugging.enable_check_numerics()

# ===========================
# 训练参数
# ===========================
_k = 1000
TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(50000, 0.001, 32)

LossParams = namedtuple(
    'LossParams',
    [
        'pos',
        'vf',
        'zero',
        'entropy',
        'mass',
        'gamma',
        'neighbor_scale',
        'use_importance',
    ]
)
loss_params = LossParams(
    pos=1.0,
    vf=5.0,
    zero=2.0,
    entropy=0.01,
    mass=1.0,
    gamma=1.0,
    neighbor_scale=1.0 / 40.0,
    use_importance=True,
)


# ===========================
# GPU + Model
# ===========================
def create_model(gpu_id=0, **kwargs):
    """
    创建模型，并绑定GPU
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        target_gpu = gpus[min(gpu_id, len(gpus)-1)]
        tf.config.set_visible_devices(target_gpu, 'GPU')
        tf.config.experimental.set_memory_growth(target_gpu, True)
        print(f"[INFO] Using GPU: {target_gpu.name}")
    else:
        print("[INFO] No GPU found, using CPU")

    model = MultiPhaseParticleNetwork(**kwargs)
    return model


# ===========================
# Loss Functions（核心修改）
# ===========================

def euclidean_distance(a, b, epsilon=1e-9):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1) + epsilon)


def compute_importance(num_fluid_neighbors):
    if (not loss_params.use_importance) or num_fluid_neighbors is None:
        return None
    num_fluid_neighbors = tf.cast(num_fluid_neighbors, tf.float32)
    return tf.exp(-loss_params.neighbor_scale * num_fluid_neighbors)


def weighted_mean(values, importance=None):
    values = tf.cast(values, tf.float32)
    if importance is None:
        return tf.reduce_mean(values)

    importance = tf.cast(importance, tf.float32)
    denom = tf.reduce_sum(importance) + 1e-8
    return tf.reduce_sum(values * importance) / denom


def kl_vf_loss(pr, gt, importance=None):
    """
    KL 散度损失
    强烈惩罚 GT=0 但 prediction>0 的情况
    """
    pr = tf.clip_by_value(tf.cast(pr, tf.float32), 1e-6, 1.0)
    gt = tf.clip_by_value(tf.cast(gt, tf.float32), 1e-6, 1.0)

    kl = tf.reduce_sum(gt * tf.math.log(gt / pr), axis=-1)
    return weighted_mean(kl, importance)


def zero_phase_penalty(pr, gt, importance=None):
    """
    GT=0 的相，prediction 应接近 0
    """
    pr = tf.cast(pr, tf.float32)
    gt = tf.cast(gt, tf.float32)
    mask = tf.cast(gt < 1e-6, tf.float32)
    penalty = tf.reduce_mean(mask * pr, axis=-1)
    return weighted_mean(penalty, importance)


def entropy_loss(pr, importance=None):
    """
    熵越高越接近均匀分布，因此加惩罚
    """
    pr = tf.cast(pr, tf.float32)
    ent = -tf.reduce_sum(pr * tf.math.log(pr + 1e-8), axis=-1)
    return weighted_mean(ent, importance)


def total_mass_conservation_loss(vf_next, vf_current, phase_densities):
    """
    全局质量守恒
    """
    vf_next = tf.cast(vf_next, tf.float32)
    vf_current = tf.cast(vf_current, tf.float32)
    phase_densities = tf.cast(phase_densities, tf.float32)

    rho_cur = tf.reduce_sum(vf_current * phase_densities, axis=-1, keepdims=True)
    rho_next = tf.reduce_sum(vf_next * phase_densities, axis=-1, keepdims=True)

    mass_cur = vf_current * rho_cur
    mass_next = vf_next * rho_next

    total_cur = tf.reduce_sum(mass_cur, axis=0)
    total_next = tf.reduce_sum(mass_next, axis=0)

    drift = tf.abs(total_next - total_cur) / (total_cur + 1e-8)
    return tf.reduce_mean(drift)


def loss_fn(pr_pos, gt_pos, pr_vf, gt_vf, cur_vf, densities, importance=None):
    """
    总损失函数
    """
    pos_err = euclidean_distance(pr_pos, gt_pos)
    pos_loss = weighted_mean(tf.pow(pos_err, loss_params.gamma), importance)

    vf_loss = kl_vf_loss(pr_vf, gt_vf, importance)
    zero_loss = zero_phase_penalty(pr_vf, gt_vf, importance)
    ent_loss = entropy_loss(pr_vf, importance)
    mass_loss = total_mass_conservation_loss(pr_vf, cur_vf, densities)

    total = (
        loss_params.pos * pos_loss +
        loss_params.vf * vf_loss +
        loss_params.zero * zero_loss +
        loss_params.entropy * ent_loss +
        loss_params.mass * mass_loss
    )

    return total, pos_loss, vf_loss, zero_loss, ent_loss, mass_loss


# ===========================
# Train Step
# ===========================

@tf.function(experimental_relax_shapes=True)
def train_step(model, optimizer, batch):

    with tf.GradientTape() as tape:
        losses = []

        pos_l, vf_l, zero_l, ent_l, mass_l = 0., 0., 0., 0., 0.

        for i in range(train_params.batch_size):
            pos0 = batch['pos0'][i]
            vel0 = batch['vel0'][i]
            box = batch['box'][i]
            box_n = batch['box_normals'][i]

            gt_pos1 = batch['pos1'][i]
            gt_pos2 = batch['pos2'][i]

            vf0 = batch['phase_fractions0'][i]
            vf1 = batch['phase_fractions1'][i]
            vf2 = batch['phase_fractions2'][i]

            dens = batch['density'][i]

            cd = tf.cast(batch['cd'][i], tf.float32)
            cf = tf.cast(batch['cf'][i], tf.float32)

            p1, v1, pred_vf1 = model(
                (pos0, vel0, vf0, box, box_n),
                phase_densities=dens,
                training=True, cd=cd, cf=cf
            )
            importance1 = model.num_fluid_neighbors
            l1, p_l1, vf_l1, z_l1, e_l1, m_l1 = loss_fn(
                p1, gt_pos1, pred_vf1, vf1, vf0, dens, importance1
            )

            p2, v2, pred_vf2 = model(
                (p1, v1, pred_vf1, box, box_n),
                phase_densities=dens,
                training=True, cd=cd, cf=cf
            )
            importance2 = model.num_fluid_neighbors
            l2, p_l2, vf_l2, z_l2, e_l2, m_l2 = loss_fn(
                p2, gt_pos2, pred_vf2, vf2, pred_vf1, dens, importance2
            )

            losses.append(0.5 * (l1 + l2))

            pos_l += p_l1 + p_l2
            vf_l += vf_l1 + vf_l2
            zero_l += z_l1 + z_l2
            ent_l += e_l1 + e_l2
            mass_l += m_l1 + m_l2

        total_loss = tf.add_n(losses) / float(train_params.batch_size)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss, pos_l, vf_l, zero_l, ent_l, mass_l


# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    print(f"[INFO] Loading config: {args.cfg}")
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    global train_params, loss_params, _k
    train_params = train_params._replace(**cfg.get('train_params', {}))
    loss_params = loss_params._replace(**cfg.get('loss_weights', {}))
    _k = train_params.max_iter // 50

    print(f"[INFO] Training params: {train_params}")
    print(f"[INFO] Loss params: {loss_params}")

    # ===== 自动创建 train_dir 并写入 cfg =====
    if '2025' not in cfg['train_dir'] and '2026' not in cfg['train_dir']:
        train_dir = os.path.join(cfg['train_dir'], datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(train_dir, exist_ok=True)

        cfg['train_dir'] = train_dir
        with open(os.path.join(train_dir, 'training_config.yaml'), 'w') as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    else:
        train_dir = cfg['train_dir']

    print(f"[INFO] Train directory: {train_dir}")

    train_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'train', '*.h5')))
    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.h5')))

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=3, # For 2-step prediction 
                              num_workers=cfg.get('num_workers', 2),
                              **cfg.get('train_data', {}))
    val_dataset = read_data_val(files=val_files, window=1)

    model = create_model(args.gpu, **cfg.get('model', {}))

    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [10*_k, 20*_k, 30*_k],
        [train_params.base_lr,
         train_params.base_lr*0.5,
         train_params.base_lr*0.25,
         train_params.base_lr*0.1]
    )
    optimizer = tf.keras.optimizers.Adam(lr)

    trainer = Trainer(train_dir)

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0, dtype=tf.int64),
        model=model,
        optimizer=optimizer
    )

    manager = MyCheckpointManager(
        ckpt,
        trainer.checkpoint_dir,
        keep_checkpoint_steps=list(range(_k, train_params.max_iter+1, _k))
    )

    data_iter = iter(dataset)

    print("[INFO] Start training...")

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        ckpt.restore(manager.latest_checkpoint)

    while trainer.keep_training(
        ckpt.step,
        train_params.max_iter,
        checkpoint_manager=manager
    ):
        batch = next(data_iter)
        batch_tf = {k: [tf.convert_to_tensor(x) for x in v] for k, v in batch.items()}

        total, pos_l, vf_l, zero_l, ent_l, mass_l = train_step(model, optimizer, batch_tf)

        if trainer.current_step % 10 == 0:
            print(
                f"[Step {trainer.current_step}] "
                f"Total={float(total):.4f} | "
                f"Pos={float(pos_l):.4f} | "
                f"VF={float(vf_l):.4f} | "
                f"Zero={float(zero_l):.4f} | "
                f"Entropy={float(ent_l):.4f} | "
                f"Mass={float(mass_l):.4f}"
            )

        if trainer.current_step % (_k) == 0 and val_files:
            print("[INFO] Running evaluation...")
            eval_res = evaluate(
                model, 
                val_dataset,
                frame_skip=cfg.get('evaluation', {}).get('frame_skip', 20),
                **cfg.get('evaluation', {})
            )
            print("[EVAL RESULT]", eval_res)

    model.save_weights(os.path.join(
        train_dir,
        "model_" + date.today().strftime("%Y%m%d") + ".h5"
    ))

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()