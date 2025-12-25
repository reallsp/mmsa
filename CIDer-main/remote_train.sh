#!/bin/bash
# 远程GPU训练脚本
# 使用方法: screen -S training 然后运行此脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "COPA-PI数据集远程GPU训练"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

# 检查GPU
echo "1. 检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "警告: nvidia-smi未找到，可能没有GPU"
fi
echo ""

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    echo "2. 激活虚拟环境..."
    source venv/bin/activate
elif [ -d "$CONDA_PREFIX" ]; then
    echo "2. 使用conda环境: $CONDA_PREFIX"
else
    echo "2. 使用系统Python环境"
fi
echo ""

# 检查数据文件
echo "3. 检查数据文件..."
if [ ! -f "../train_12.16_1.pkl" ]; then
    echo "错误: 找不到数据文件 train_12.16_1.pkl"
    echo "请确保数据文件在项目根目录"
    exit 1
fi
echo "✓ 数据文件存在"
echo ""

# 数据预处理
echo "4. 数据预处理..."
if [ ! -f "./converted_data/copa_pi_aligned_50.pkl" ]; then
    echo "  转换数据..."
    python convert_data.py
else
    echo "  ✓ 数据已转换，跳过"
fi

if [ ! -f "./copa_pi_probs/iid/binary/copa_pi_train_binary.npy" ]; then
    echo "  生成辅助数据..."
    python generate_cls_data.py
else
    echo "  ✓ 辅助数据已生成，跳过"
fi
echo ""

# 创建模型保存目录
mkdir -p models
mkdir -p logs

# 生成日志文件名（带时间戳）
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "5. 开始训练..."
echo "  日志文件: $LOG_FILE"
echo "  使用GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# 运行训练
python main_run.py \
    --dataset copa_pi \
    --task binary \
    --aligned True \
    --batch_size 32 \
    --lr 1e-3 \
    --num_epochs 100 \
    --patience 10 \
    --data_path ./converted_data \
    --model_path ./models \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "训练完成！"
echo "结束时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "模型保存目录: ./models"
echo "=========================================="

