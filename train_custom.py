#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用新数据集训练模型（CPU模式）"""
import sys
import os
sys.path.insert(0, 'src')

# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from MMSA.run import MMSA_run

print('=' * 70)
print('开始训练模型（CPU模式）')
print('=' * 70)
print('模型: TFN')
print('数据集: custom')
print('随机种子: 1111')
print('=' * 70)

try:
    results = MMSA_run(
        model_name='tfn',
        dataset_name='custom',
        seeds=[1111],
        gpu_ids=[],  # 空列表强制使用CPU
        num_workers=2,
        verbose_level=1
    )
    print('\n' + '=' * 70)
    print('训练完成！')
    print('=' * 70)
    print('结果:', results)
except Exception as e:
    print(f'\n训练出错: {e}')
    import traceback
    traceback.print_exc()

