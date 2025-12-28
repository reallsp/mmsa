#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用GPU训练模型"""
import sys
import os
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from MMSA.run import MMSA_run

# 获取当前项目目录
project_dir = Path(__file__).parent.absolute()

print('=' * 70)
print('GPU训练模式')
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
    gpu_ids = []  # 空列表使用CPU
    device_type = 'CPU'

print(f'\n模型: TFN')
print(f'数据集: custom')
print(f'随机种子: 1111')
print('=' * 70)

try:
    results = MMSA_run(
        model_name='tfn',
        dataset_name='custom',
        seeds=[1111],
        gpu_ids=gpu_ids,  # 使用GPU
        num_workers=4 if gpu_ids else 2,  # GPU模式可以使用更多workers
        verbose_level=1,
        model_save_dir=str(project_dir / "saved_models"),  # 保存到项目目录
        res_save_dir=str(project_dir / "results"),  # 保存到项目目录
        log_dir=str(project_dir / "logs")  # 保存到项目目录
    )
    print('\n' + '=' * 70)
    print(f'训练完成！({device_type}模式)')
    print('=' * 70)
    if results:
        print('结果:', results)
except RuntimeError as e:
    error_msg = str(e)
    if 'CUDA' in error_msg or 'kernel' in error_msg.lower():
        print('\n⚠ CUDA错误:', error_msg)
        print('\n这可能是因为GPU兼容性问题。')
        print('RTX 5090需要PyTorch 2.5+才能支持。')
        print('建议:')
        print('1. 升级PyTorch到支持sm_120的版本')
        print('2. 或使用CPU模式 (gpu_ids=[])')
    else:
        raise
except Exception as e:
    print(f'\n训练出错: {e}')
    import traceback
    traceback.print_exc()

