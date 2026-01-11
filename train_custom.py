#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5-Fold 交叉验证训练（custom 数据集，TFN）"""
import sys
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from MMSA.run import MMSA_run

project_dir = Path(__file__).parent.absolute()

print('=' * 70)
print('TFN | custom | 5-Fold 交叉验证')
print('=' * 70)

# 检查GPU可用性
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'检测到 {gpu_count} 个GPU:')
    for i in range(gpu_count):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    gpu_ids = [0]  # 使用第一个GPU
    print(f'使用GPU: {gpu_ids}')
else:
    print('未检测到GPU，将使用CPU')
    gpu_ids = []

model_name = 'tfn'
dataset_name = 'custom'
seeds = [1111, 1112, 1113, 1114, 1115]

print(f'模型: {model_name}')
print(f'数据集: {dataset_name}')
print(f'折数: 5')
print(f'随机种子: {seeds}')
print('=' * 70)

all_results = []

for idx, seed in enumerate(seeds, 1):
    print(f'\n>>> 开始第 {idx}/5 折 (seed={seed})')
    fold_save_dir = project_dir / "saved_models" / f"fold{idx}"
    fold_res_dir = project_dir / "results" / f"fold{idx}"
    fold_log_dir = project_dir / "logs" / f"fold{idx}"

    try:
        results = MMSA_run(
            model_name=model_name,
            dataset_name=dataset_name,
            seeds=[seed],
            gpu_ids=gpu_ids,
            num_workers=4 if gpu_ids else 2,
            verbose_level=1,
            model_save_dir=str(fold_save_dir),
            res_save_dir=str(fold_res_dir),
            log_dir=str(fold_log_dir)
        )
        all_results.append(results)
        print(f'>>> 第 {idx} 折完成，结果: {results}')
    except Exception as e:
        print(f'\n训练出错 (Fold {idx}): {e}')
        if 'CUDA' in str(e) or 'kernel' in str(e).lower():
            print('\n检测到CUDA错误，可能是GPU兼容性问题。')
            print('可以尝试使用CPU模式：gpu_ids=[]')
        import traceback
        traceback.print_exc()

print('\n' + '=' * 70)
print('5-Fold 训练完成！')
print('=' * 70)
print('各折结果:', all_results)

