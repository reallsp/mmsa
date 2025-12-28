#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用CPU训练模型"""
import sys
import os
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from MMSA.run import MMSA_run

# 获取当前项目目录
project_dir = Path(__file__).parent.absolute()

print('=' * 70)
print('CPU训练模式')
print('=' * 70)

# 强制使用CPU
gpu_ids = []  # 空列表使用CPU
device_type = 'CPU'

print(f'\n模型: TFN')
print(f'数据集: custom')
print(f'随机种子: 1111')
print(f'设备: CPU')
print('=' * 70)

try:
    # 创建自定义配置，减小batch size以适应CPU
    custom_config = {
        'batch_size': 4,  # 减小batch size以节省内存
    }
    
    # 设置环境变量以限制内存和线程使用
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['NUMEXPR_NUM_THREADS'] = '2'
    
    # 设置PyTorch使用CPU
    torch.set_num_threads(2)
    
    results = MMSA_run(
        model_name='tfn',
        dataset_name='custom',
        seeds=[1111],
        gpu_ids=gpu_ids,  # 空列表强制使用CPU
        num_workers=0,  # CPU模式使用0个worker
        config=custom_config,
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
except Exception as e:
    print(f'\n训练出错: {e}')
    import traceback
    traceback.print_exc()

