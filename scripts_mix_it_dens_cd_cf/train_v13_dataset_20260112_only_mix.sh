#!/bin/bash

# 定义变量, 批量处理文件夹
TRAIN_SCRIPT="scripts/train_mix_v13.py"
YAML_FILE="scripts_mix_it_dens_cd_cf/mix_v13_dataset_20260112_only_mix.yaml"
DATE=$(date +"%Y%m%d_%H%M%S")  # 获取当前日期和时间，格式为 YYYYMMDD_HHMMSS
LOG_FILE="scripts_mix_it_dens_cd_cf/log_train/train_v13_dataset_20260112_only_mix_${DATE}.log"  # 定义日志文件名，包含日期和时间

GPU_ID="0"  # 默认使用GPU 0

# 检查输入参数
if [ "$#" -ge 1 ]; then
  GPU_ID="$1"  # 如果提供了参数，将其作为GPU ID
fi

# 检查脚本是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "错误: 脚本 '$TRAIN_SCRIPT' 未找到。"
  exit 1
fi

# 运行 Python 脚本，重定向输出和错误
nohup python "$TRAIN_SCRIPT" "$YAML_FILE" \
  --gpu "$GPU_ID" > "$LOG_FILE" 2>&1 &

PID=$!
echo "脚本已在后台运行，进程 ID: $PID"
echo "脚本已在后台运行，日志输出到 $LOG_FILE"
echo "可以使用以下命令查看训练进度:"
echo "  tail -f $LOG_FILE"
exit 0