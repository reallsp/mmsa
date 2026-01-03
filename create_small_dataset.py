#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建小样本数据集用于快速测试
从原始数据集中采样少量样本
"""
import pickle
import numpy as np
from pathlib import Path

def create_small_dataset(
    train_file='/root/autodl-tmp/data/copa_train_1231_converted.pkl',
    test_file='/root/autodl-tmp/data/copa_test_1231_converted.pkl',
    output_file='/root/autodl-tmp/data/copa_1231_small.pkl',
    train_samples=100,  # 训练集样本数
    test_samples=50    # 测试集样本数
):
    """
    创建小样本数据集
    
    Args:
        train_file: 训练集文件路径
        test_file: 测试集文件路径
        output_file: 输出小数据集路径
        train_samples: 训练集样本数
        test_samples: 测试集样本数
    """
    print(f'加载训练集: {train_file}')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    print(f'加载测试集: {test_file}')
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    # 合并数据
    data = {
        'train': train_data.get('train', train_data),
        'test': test_data.get('test', test_data)
    }
    
    print(f'原始数据大小:')
    print(f'  训练集: {len(data["train"]["regression_labels"])} 样本')
    print(f'  测试集: {len(data["test"]["regression_labels"])} 样本')
    
    # 采样训练集
    train_size = len(data['train']['regression_labels'])
    train_indices = np.random.choice(train_size, min(train_samples, train_size), replace=False)
    
    # 采样测试集
    test_size = len(data['test']['regression_labels'])
    test_indices = np.random.choice(test_size, min(test_samples, test_size), replace=False)
    
    print(f'\n采样后大小:')
    print(f'  训练集: {len(train_indices)} 样本')
    print(f'  测试集: {len(test_indices)} 样本')
    
    # 创建小数据集
    small_data = {
        'train': {},
        'valid': {},  # 使用测试集作为验证集占位
        'test': {}
    }
    
    # 采样训练集数据
    for key in data['train'].keys():
        if isinstance(data['train'][key], np.ndarray):
            small_data['train'][key] = data['train'][key][train_indices]
        elif isinstance(data['train'][key], list):
            small_data['train'][key] = [data['train'][key][i] for i in train_indices]
        else:
            small_data['train'][key] = data['train'][key]
    
    # 采样测试集数据
    for key in data['test'].keys():
        if isinstance(data['test'][key], np.ndarray):
            small_data['test'][key] = data['test'][key][test_indices]
        elif isinstance(data['test'][key], list):
            small_data['test'][key] = [data['test'][key][i] for i in test_indices]
        else:
            small_data['test'][key] = data['test'][key]
    
    # 验证集使用测试集（占位）
    small_data['valid'] = small_data['test']
    
    # 保存小数据集
    print(f'\n保存小数据集: {output_file}')
    with open(output_file, 'wb') as f:
        pickle.dump(small_data, f)
    
    print('✅ 小数据集创建完成！')
    return output_file

if __name__ == '__main__':
    import sys
    # 设置随机种子
    np.random.seed(42)
    
    # 创建小数据集
    create_small_dataset(
        train_file='/root/autodl-tmp/data/copa_train_1231_converted.pkl',
        test_file='/root/autodl-tmp/data/copa_test_1231_converted.pkl',
        train_samples=100,  # 100个训练样本
        test_samples=50     # 50个测试样本
    )

