#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用训练和测试脚本
支持配置任意模型和数据集进行训练和测试
"""
import sys
import os
import argparse
import json
from pathlib import Path
sys.path.insert(0, 'src')

import torch
from MMSA.run import MMSA_run

# 支持的模型列表（singleTask）
SUPPORTED_MODELS = [
    'tfn', 'lmf', 'mfn', 'graph_mfn', 'ef_lstm', 'lf_dnn',
    'mult', 'misa', 'bert_mag', 'mfm', 'mmim', 'mctn',
    'cenet', 'almt', 'almt_cider'
]

# 支持的数据集列表
SUPPORTED_DATASETS = [
    'mosi', 'mosei', 'sims', 'simsv2',
    'custom', 'train_12_16', 'copa_1231'
]

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='MMSA 通用训练和测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
支持的模型:
  {', '.join(SUPPORTED_MODELS)}

支持的数据集:
  {', '.join(SUPPORTED_DATASETS)}

示例:
  # 使用 TFN 模型训练 copa_1231 数据集
  python train.py -m tfn -d copa_1231

  # 使用 LMF 模型训练 MOSI 数据集，使用多个随机种子
  python train.py -m lmf -d mosi -s 1111 1112 1113

  # 使用自定义配置
  python train.py -m tfn -d copa_1231 --config config.json

  # 使用 CPU 训练
  python train.py -m tfn -d copa_1231 --gpu-ids ""
        """
    )
    
    # 必需参数
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help='模型名称'
    )
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help='数据集名称'
    )
    
    # 可选参数
    parser.add_argument(
        '-s', '--seeds',
        type=int,
        nargs='+',
        default=[1111],
        help='随机种子列表 (默认: 1111)'
    )
    
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=None,
        help='GPU ID列表 (默认: 自动检测并使用第一个GPU，空列表使用CPU)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='数据加载器工作进程数 (默认: GPU模式4, CPU模式2)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='跳过验证集流程（直接训练和测试）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='自定义配置文件路径 (JSON格式)'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default=None,
        help='MMSA配置文件路径 (config_regression.json 或 config_tune.json)'
    )
    
    parser.add_argument(
        '--model-save-dir',
        type=str,
        default=None,
        help='模型保存目录 (默认: ./saved_models)'
    )
    
    parser.add_argument(
        '--res-save-dir',
        type=str,
        default=None,
        help='结果保存目录 (默认: ./results)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='日志保存目录 (默认: ./logs)'
    )
    
    parser.add_argument(
        '--verbose-level',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='日志详细程度: 0=ERROR, 1=INFO, 2=DEBUG (默认: 1)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='启用超参数调优模式'
    )
    
    parser.add_argument(
        '--tune-times',
        type=int,
        default=50,
        help='超参数调优次数 (默认: 50)'
    )
    
    return parser.parse_args()

def load_custom_config(config_path):
    """加载自定义配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def auto_detect_gpu():
    """自动检测并返回GPU ID"""
    if torch.cuda.is_available():
        return [0]
    return []

def main():
    args = parse_args()
    
    # 获取项目目录
    project_dir = Path(__file__).parent.absolute()
    
    # 打印配置信息
    print('=' * 70)
    print('MMSA 通用训练和测试')
    print('=' * 70)
    print(f'模型: {args.model.upper()}')
    print(f'数据集: {args.dataset.upper()}')
    print(f'随机种子: {args.seeds}')
    print(f'跳过验证集: {"是" if args.skip_validation else "否"}')
    if args.tune:
        print(f'超参数调优: 是 (次数: {args.tune_times})')
    
    # GPU配置
    if args.gpu_ids is None:
        gpu_ids = auto_detect_gpu()
    else:
        gpu_ids = args.gpu_ids
    
    if gpu_ids:
        if torch.cuda.is_available():
            print(f'使用GPU: {gpu_ids}')
            for gpu_id in gpu_ids:
                print(f'  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}')
        else:
            print('⚠ 警告: 指定了GPU但未检测到CUDA，将使用CPU')
            gpu_ids = []
    else:
        print('使用CPU')
    
    # 工作进程数
    if args.num_workers is None:
        num_workers = 4 if gpu_ids else 2
    else:
        num_workers = args.num_workers
    
    print(f'数据加载工作进程数: {num_workers}')
    print('=' * 70)
    
    # 加载自定义配置
    custom_config = None
    if args.config:
        print(f'\n加载自定义配置: {args.config}')
        custom_config = load_custom_config(args.config)
        print(f'  配置项: {list(custom_config.keys())}')
    
    # 准备参数
    mmsa_params = {
        'model_name': args.model,
        'dataset_name': args.dataset,
        'seeds': args.seeds,
        'gpu_ids': gpu_ids,
        'num_workers': num_workers,
        'verbose_level': args.verbose_level,
        'skip_validation': args.skip_validation,
    }
    
    # 目录配置
    if args.model_save_dir:
        mmsa_params['model_save_dir'] = args.model_save_dir
    else:
        mmsa_params['model_save_dir'] = str(project_dir / "saved_models")
    
    if args.res_save_dir:
        mmsa_params['res_save_dir'] = args.res_save_dir
    else:
        mmsa_params['res_save_dir'] = str(project_dir / "results")
    
    if args.log_dir:
        mmsa_params['log_dir'] = args.log_dir
    else:
        mmsa_params['log_dir'] = str(project_dir / "logs")
    
    # 配置文件
    if args.config_file:
        mmsa_params['config_file'] = args.config_file
    
    # 自定义配置
    if custom_config:
        mmsa_params['config'] = custom_config
    
    # 超参数调优
    if args.tune:
        mmsa_params['is_tune'] = True
        mmsa_params['tune_times'] = args.tune_times
    
    # 运行训练和测试
    try:
        print('\n开始训练和测试...')
        results = MMSA_run(**mmsa_params)
        
        print('\n' + '=' * 70)
        print('训练和测试完成！')
        print('=' * 70)
        print('\n结果文件:')
        print(f'  - 模型: {mmsa_params["model_save_dir"]}/{args.model}-{args.dataset}.pth')
        print(f'  - 结果: {mmsa_params["res_save_dir"]}/normal/{args.dataset}.csv')
        print(f'  - 日志: {mmsa_params["log_dir"]}/{args.model}-{args.dataset}.log')
        
        if results:
            copa_keys = [k for k in results.keys() if 'COPA' in k.upper()]
            if copa_keys:
                print(f'\n✓ 包含 {len(copa_keys)} 个 COPA 指标')
        
    except RuntimeError as e:
        error_msg = str(e)
        if 'CUDA' in error_msg or 'kernel' in error_msg.lower():
            print('\n⚠ CUDA错误:', error_msg)
            print('\n这可能是因为GPU兼容性问题。')
            print('建议:')
            print('1. 升级PyTorch到支持sm_120的版本')
            print('2. 或使用CPU模式: --gpu-ids ""')
        else:
            raise
    except Exception as e:
        print(f'\n❌ 训练出错: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

