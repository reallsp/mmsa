import json
import os
from pathlib import Path
import random
from easydict import EasyDict as edict


def get_config_regression(
    model_name: str, dataset_name: str, config_file: str = ""
) -> dict:
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    # 如果模型没有该数据集的配置，尝试使用fallback数据集
    fallback_datasets = ['sims', 'mosi', 'mosei']  # fallback顺序
    if dataset_name not in config_all[model_name]['datasetParams']:
        # 尝试使用fallback数据集
        model_dataset_args = {}
        for fallback in fallback_datasets:
            if fallback in config_all[model_name]['datasetParams']:
                model_dataset_args = config_all[model_name]['datasetParams'][fallback].copy()
                break
        if not model_dataset_args:
            raise KeyError(f"Model {model_name} has no configuration for dataset {dataset_name} and no fallback available")
    else:
        model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
    
    # 数据集参数
    if dataset_name not in config_all['datasetCommonParams']:
        # 如果数据集不在commonParams中，尝试使用custom的配置
        if 'custom' in config_all['datasetCommonParams']:
            dataset_args = config_all['datasetCommonParams']['custom'].copy()
            # 需要处理aligned/unaligned结构
            if isinstance(dataset_args, dict) and 'unaligned' in dataset_args:
                dataset_args = dataset_args['unaligned']
        else:
            raise KeyError(f"Dataset {dataset_name} not found in datasetCommonParams")
    else:
        dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    if isinstance(dataset_args, dict):
        if model_common_args['need_data_aligned'] and 'aligned' in dataset_args:
            dataset_args = dataset_args['aligned']
        elif 'unaligned' in dataset_args:
            dataset_args = dataset_args['unaligned']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    # 处理featurePath：如果是绝对路径或custom数据集，直接使用；否则拼接dataset_root_dir
    if os.path.isabs(config['featurePath']) or dataset_name.lower() in ['custom', 'train_12_16']:
        # 对于绝对路径或custom数据集，如果路径不存在，尝试从项目根目录查找
        if not os.path.exists(config['featurePath']):
            # 尝试从项目根目录查找
            project_root = Path(__file__).parent.parent.parent
            alt_path = project_root / config['featurePath'].lstrip('/')
            if alt_path.exists():
                config['featurePath'] = str(alt_path)
            else:
                # 如果还是不存在，尝试直接使用文件名（假设在项目根目录）
                simple_path = project_root / os.path.basename(config['featurePath'])
                if simple_path.exists():
                    config['featurePath'] = str(simple_path)
    else:
        config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_tune(
    model_name: str, dataset_name: str, config_file: str = "",
    random_choice: bool = True
) -> dict:
    """
    Get the tuning config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.
        random_choice: If True, will randomly choose a config from the list of configs.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name] if 'datasetParams' in config_all[model_name] else {}
    model_debug_args = config_all[model_name]['debugParams']
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    # random choice of args
    if random_choice:
        for item in model_debug_args['d_paras']:
            if type(model_debug_args[item]) == list:
                model_debug_args[item] = random.choice(model_debug_args[item])
            elif type(model_debug_args[item]) == dict: # nested params, 2 levels max
                for k, v in model_debug_args[item].items():
                    model_debug_args[item][k] = random.choice(v)

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    

    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_all(config_file: str) -> dict:
    """
    Get all default configs. This function is used to export default config file. 
    If you want to get config for a specific model, use "get_config_regression" or "get_config_tune" instead.

    Parameters:
        config_file: "regression" or "tune"
    
    Returns:
        config: all default configs
    """
    if config_file == "regression":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    elif config_file == "tune":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    else:
        raise ValueError("config_file should be 'regression' or 'tune'")
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    return edict(config_all)

def get_citations() -> dict:
    """
    Get paper titles and citations for models and datasets.

    Returns:
        cites (dict): {
            models: {
                tfn: {
                    title: "xxx",
                    paper_url: "xxx",
                    citation: "xxx",
                    description: "xxx"
                },
                ...
            },
            datasets: {
                ...
            },
        }
    """
    # TODO: add citations
    config_file = Path(__file__).parent / "config" / "citations.json"
    with open(config_file, 'r') as f:
        cites = json.load(f)
    return cites