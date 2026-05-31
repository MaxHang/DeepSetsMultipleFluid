#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import tensorflow as tf
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_h5_mix import read_data_val
from fluid_evaluation_helper import FluidErrors


def evaluate_tf(model, val_dataset, frame_skip, fluid_errors=None, scale=1):
    """
    短时（2步）评估函数。
    """
    print('evaluating.. ', end='')
    

    if fluid_errors is None:
        fluid_errors = FluidErrors()

    skip = frame_skip

    last_scene_id = 0
    frames = []
    for data in val_dataset:
        if data['frame_id0'][0] == 1:
            frames = []
        if data['frame_id0'][0] % skip < 4:
            frames.append(data)
        if data['frame_id0'][0] % skip == 4:
            # 确保收集到足够的帧数
            if len(
                    set([
                        frames[0]['scene_id0'][0], frames[1]['scene_id0'][0],
                        frames[2]['scene_id0'][0]
                    ])) == 1:
                scene_id = frames[0]['scene_id0'][0]
                if last_scene_id != scene_id:
                    last_scene_id = scene_id
                    print(scene_id, end=' ', flush=True)

                frame0_id = frames[0]['frame_id0'][0]
                frame1_id = frames[1]['frame_id0'][0]
                frame2_id = frames[2]['frame_id0'][0]
                box = frames[0]['box'][0]
                box_normals = frames[0]['box_normals'][0]
                gt_pos1 = frames[1]['pos0'][0]
                gt_pos2 = frames[2]['pos0'][0]

                ## V4-MOD: 从数据中获取动态参数。假设整个短序列的物理属性不变。
                # 您的验证数据集读取器必须提供这些字段！
                num_phases_sample = tf.constant(frames[0]['num_phases'][0], dtype=tf.int32)
                densities_sample = tf.constant(frames[0]['density'][0], dtype=tf.float32)

                # 获取初始相分数 (如果存在)
                vf0 = frames[0].get('phase_fractions0', [None])[0]

                # 获取 Cd/Cf
                cd_val = tf.cast(frames[0].get('cd', [0.5])[0], tf.float32)
                cf_val = tf.cast(frames[0].get('cf', [0.5])[0], tf.float32)

                # 进行第一次预测
                inputs = (frames[0]['pos0'][0], frames[0]['vel0'][0], vf0, box, box_normals)
                
                pr_pos1, pr_vel1, pr_phase1 = model(
                    inputs, 
                    current_num_phases=num_phases_sample,
                    phase_densities=densities_sample,
                    cd=cd_val, cf=cf_val)
                
                # 第二次预测
                inputs = (pr_pos1, pr_vel1, pr_phase1, box, box_normals)
                pr_pos2, pr_vel2, pr_phase2 = model(
                    inputs, 
                    current_num_phases=num_phases_sample,
                    phase_densities=densities_sample,
                    cd=cd_val, cf=cf_val)


                # 确保帧ID是按照正确顺序的 (小的在前，大的在后)
                if frame0_id <= frame1_id:
                    fluid_errors.add_errors(scene_id, frame0_id, frame1_id,
                                        scale * pr_pos1, scale * gt_pos1)
                else:
                    # 如果帧顺序不对，交换一下
                    fluid_errors.add_errors(scene_id, frame1_id, frame0_id,
                                        scale * gt_pos1, scale * pr_pos1)
                    
                if frame0_id <= frame2_id:
                    fluid_errors.add_errors(scene_id, frame0_id, frame2_id,
                                        scale * pr_pos2, scale * gt_pos2)
                else:
                    # 如果帧顺序不对，交换一下
                    fluid_errors.add_errors(scene_id, frame2_id, frame0_id,
                                        scale * gt_pos2, scale * pr_pos2)

            frames = []

    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])

    print(result)
    print('done')

    return result


def evaluate_whole_sequence_tf(model,
                               val_dataset,
                               frame_skip,
                               fluid_errors=None,
                               scale=1):
    """
    长时（序列 rollout）评估函数。
    """
    print('evaluating.. ', end='')

    if fluid_errors is None:
        fluid_errors = FluidErrors()

    skip = frame_skip

    last_scene_id = None
    pr_pos = None
    pr_vel = None
    pr_phase = None
    
    for data in val_dataset:
        scene_id = data['scene_id0'][0]

        # 如果是新场景，进行初始化
        if last_scene_id is None or last_scene_id != scene_id:
            print(scene_id, end=' ', flush=True)
            last_scene_id = scene_id
            
            # 初始化状态
            box, box_normals = data['box'][0], data['box_normals'][0]
            pr_pos, pr_vel = data['pos0'][0], data['vel0'][0]

            # 获取初始相分数
            pr_phase = data.get('phase_fractions0', [None])[0]
        
        # --- 统一的预测步骤 ---
        # 准备输入
        inputs = (pr_pos, pr_vel, pr_phase, box, box_normals)
        
        # 获取 Cd/Cf
        cd_val = tf.cast(data.get('cd', [0.5])[0], tf.float32)
        cf_val = tf.cast(data.get('cf', [0.5])[0], tf.float32)

        # 预测结果会成为下一次迭代的输入
        pr_pos, pr_vel, pr_phase = model(inputs,
                                         cd=cd_val, cf=cf_val, training=False)

        # 在指定的帧上计算并记录误差
        frame_id = data['frame_id0'][0]
        if frame_id > 0 and frame_id % skip == 0:
            gt_pos = data['pos0'][0]
            # 使用0作为初始帧，确保初始帧始终小于当前帧
            init_frame = 0  # 使用固定的初始帧ID
            curr_frame = frame_id
            # 确保帧ID顺序正确
            if init_frame < curr_frame:
                fluid_errors.add_errors(scene_id,
                                    init_frame,
                                    curr_frame,
                                    scale * pr_pos,
                                    scale * gt_pos,
                                    compute_gt2pred_distance=True)
            else:
                print(f"警告: 跳过帧 {curr_frame}, 初始帧ID应小于当前帧ID")

    result = {}
    result['whole_seq_err'] = np.mean([
        v['gt2pred_mean']
        for k, v in fluid_errors.errors.items()
        if 'gt2pred_mean' in v
    ])

    print(result)
    print('done')

    return result


def eval_checkpoint(checkpoint_path, val_files, fluid_errors, options, cfg):
    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    if checkpoint_path.endswith('.index'):
        import tensorflow as tf

        model = trainscript.create_model(**cfg.get('model', {}))
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
        checkpoint.restore(
            os.path.splitext(checkpoint_path)[0]).expect_partial()

        evaluate_tf(model, val_dataset, options.frame_skip, fluid_errors,
                    **cfg.get('evaluation', {}))
        evaluate_whole_sequence_tf(model, val_dataset, options.frame_skip,
                                   fluid_errors, **cfg.get('evaluation', {}))
    elif checkpoint_path.endswith('.h5'):
        import tensorflow as tf

        model = trainscript.create_model(**cfg.get('model', {}))
        model.init()
        model.load_weights(checkpoint_path, by_name=True)
        evaluate_tf(model, val_dataset, options.frame_skip, fluid_errors,
                    **cfg.get('evaluation', {}))
        evaluate_whole_sequence_tf(model, val_dataset, options.frame_skip,
                                   fluid_errors, **cfg.get('evaluation', {}))
    else:
        raise Exception('Unknown checkpoint format')


def print_errors(fluid_errors):
    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])
    result['whole_seq_err'] = np.mean([
        v['gt2pred_mean']
        for k, v in fluid_errors.errors.items()
        if 'gt2pred_mean' in v
    ])
    print('====================\n', result)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates a fluid network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainscript",
                        type=str,
                        required=True,
                        help="The python training script.")
    parser.add_argument("--cfg",
                        type=str,
                        required=True,
                        help="The path to the yaml config file")
    parser.add_argument(
        "--checkpoint_iter",
        type=int,
        required=False,
        help="The checkpoint iteration. The default is the last checkpoint.")
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        help="If set uses the specified weights file instead of a checkpoint.")
    parser.add_argument("--frame-skip",
                        type=int,
                        default=5,
                        help="The frame skip. Default is 5.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    global trainscript
    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript = importlib.import_module(module_name)

    train_dir = module_name + '_' + os.path.splitext(os.path.basename(
        args.cfg))[0]
    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst')))

    if args.weights is not None:
        print('evaluating :', args.weights)
        output_path = args.weights + '_eval.json'
        if os.path.isfile(output_path):
            print('Printing previously computed results for :', args.weights,
                  output_path)
            fluid_errors = FluidErrors()
            fluid_errors.load(output_path)
        else:
            fluid_errors = FluidErrors()
            eval_checkpoint(args.weights, val_files, fluid_errors, args, cfg)
            fluid_errors.save(output_path)
    else:
        # get a list of checkpoints

        # tensorflow checkpoints
        checkpoint_files = glob(
            os.path.join(train_dir, 'checkpoints', 'ckpt-*.index'))
        # torch checkpoints
        checkpoint_files.extend(
            glob(os.path.join(train_dir, 'checkpoints', 'ckpt-*.pt')))
        all_checkpoints = sorted([
            (int(re.match('.*ckpt-(\d+)\.(pt|index)', x).group(1)), x)
            for x in checkpoint_files
        ])

        # select the checkpoint
        if args.checkpoint_iter is not None:
            checkpoint = dict(all_checkpoints)[args.checkpoint_iter]
        else:
            checkpoint = all_checkpoints[-1]

        output_path = train_dir + '_eval_{}.json'.format(checkpoint[0])
        if os.path.isfile(output_path):
            print('Printing previously computed results for :', checkpoint,
                  output_path)
            fluid_errors = FluidErrors()
            fluid_errors.load(output_path)
        else:
            print('evaluating :', checkpoint)
            fluid_errors = FluidErrors()
            eval_checkpoint(checkpoint[1], val_files, fluid_errors, args, cfg)
            fluid_errors.save(output_path)

    print_errors(fluid_errors)
    return 0


if __name__ == '__main__':
    sys.exit(main())
