#!/bin/bash
# COPA-PI数据集训练脚本

echo "=========================================="
echo "COPA-PI数据集训练"
echo "=========================================="

# 步骤1: 数据转换
echo ""
echo "步骤1: 转换数据..."
python convert_data.py

# 步骤2: 生成辅助数据
echo ""
echo "步骤2: 生成类别特征和概率文件..."
python generate_cls_data.py

# 步骤3: 训练模型
echo ""
echo "步骤3: 开始训练..."
python main_run.py \
    --dataset copa_pi \
    --task binary \
    --aligned True \
    --batch_size 32 \
    --lr 1e-3 \
    --num_epochs 100 \
    --patience 10 \
    --data_path ./converted_data \
    --model_path ./models

echo ""
echo "训练完成！"

