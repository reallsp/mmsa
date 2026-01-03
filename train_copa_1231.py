#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 COPA 1231 数据集训练和测试模型（跳过验证集流程）
"""
import sys
import os
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from MMSA.run import MMSA_run

# 获取当前项目目录
project_dir = Path(__file__).parent.absolute()

print('=' * 70)
print('COPA 1231 数据集训练和测试')
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
print(f'数据集: copa_1231')
print(f'随机种子: 1111')
print(f'跳过验证集: 是')
print('=' * 70)

try:
    results = MMSA_run(
        model_name='tfn',
        dataset_name='copa_1231',
        seeds=[1111],
        gpu_ids=gpu_ids,
        num_workers=4 if gpu_ids else 2,
        verbose_level=1,
        skip_validation=True,  # 跳过验证集流程
        model_save_dir=str(project_dir / "saved_models"),
        res_save_dir=str(project_dir / "results"),
        log_dir=str(project_dir / "logs")
    )
    print('\n' + '=' * 70)
    print(f'训练和测试完成！({device_type}模式)')
    print('=' * 70)
    print('\n结果已保存到:')
    print(f'  - 模型: saved_models/tfn-copa_1231.pth')
    print(f'  - 结果: results/normal/copa_1231.csv')
    print(f'  - 日志: logs/tfn-copa_1231.log')
    print('\n结果包含以下指标:')
    if results:
        copa_keys = [k for k in results.keys() if 'COPA' in k.upper()]
        if copa_keys:
            print(f'  ✓ COPA指标: {len(copa_keys)} 个')
            print(f'    - COPA整体准确率')
            print(f'    - COPA_P1~P12准确率')
        print(f'  ✓ 基础指标: 准确率、F1分数、MAE、相关系数等')
except RuntimeError as e:
    error_msg = str(e)
    if 'CUDA' in error_msg or 'kernel' in error_msg.lower():
        print('\n⚠ CUDA错误:', error_msg)
        print('\n这可能是因为GPU兼容性问题。')
        print('建议:')
        print('1. 升级PyTorch到支持sm_120的版本')
        print('2. 或使用CPU模式 (修改 gpu_ids=[])')
    else:
        raise
except Exception as e:
    print(f'\n训练出错: {e}')
    import traceback
    traceback.print_exc()

