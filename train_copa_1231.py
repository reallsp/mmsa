#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5-Fold 交叉验证训练与测试（COPA 1231，TFN，跳过验证集）"""
import sys
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from MMSA.run import MMSA_run

# 获取当前项目目录
project_dir = Path(__file__).parent.absolute()

print('=' * 70)
print('TFN | copa_1231 | 5-Fold 交叉验证 (skip val)')
print('=' * 70)

# 检查GPU可用性
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'✓ 检测到 {gpu_count} 个GPU:')
    for i in range(gpu_count):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    gpu_ids = [0]  # 使用第一个GPU
    print(f'\n使用GPU: {gpu_ids[0]}')
    device_type = 'GPU'
else:
    print('✗ 未检测到GPU，将使用CPU')
    gpu_ids = []
    device_type = 'CPU'

model_name = 'tfn'
dataset_name = 'copa_1231'
seeds = [1111, 1112, 1113, 1114, 1115]

print(f'\n模型: {model_name}')
print(f'数据集: {dataset_name}')
print(f'折数: 5')
print(f'随机种子: {seeds}')
print(f'跳过验证集: 是')
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
            skip_validation=True,
            model_save_dir=str(fold_save_dir),
            res_save_dir=str(fold_res_dir),
            log_dir=str(fold_log_dir)
        )
        all_results.append(results)
        print(f'>>> 第 {idx} 折完成，结果: {results}')
    except RuntimeError as e:
        error_msg = str(e)
        if 'CUDA' in error_msg or 'kernel' in error_msg.lower():
            print('\n⚠ CUDA错误:', error_msg)
            print('\n这可能是因为GPU兼容性问题。')
            print('建议:')
            print('1. 升级PyTorch到支持sm_120的版本')
            print('2. 或使用CPU模式 (修改 gpu_ids=[])')
        else:
            print(f'\n训练出错 (Fold {idx}): {error_msg}')
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f'\n训练出错 (Fold {idx}): {e}')
        import traceback
        traceback.print_exc()

print('\n' + '=' * 70)
print(f'5-Fold 训练和测试完成！({device_type}模式)')
print('=' * 70)
print('\n各折结果:')
for i, r in enumerate(all_results, 1):
    print(f'  Fold {i}: {r}')

