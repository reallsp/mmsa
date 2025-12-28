# 项目结构说明

## 核心文件

### 数据文件
- `train_12.16_1_converted.pkl` - 转换后的训练数据（MMSA格式）

### 脚本文件
- `convert_train_data.py` - 数据格式转换脚本（将原始数据转换为MMSA格式）
- `train_gpu.py` - GPU训练脚本（主要训练脚本）

### 源代码
- `src/MMSA/` - MMSA框架核心代码
  - `config/` - 配置文件
  - `models/` - 模型定义
  - `trains/` - 训练器
  - `data_loader.py` - 数据加载器
  - `run.py` - 运行入口
  - `config.py` - 配置管理

### 配置文件
- `README.md` - 项目说明文档
- `LICENSE` - 许可证
- `pyproject.toml` - Python项目配置
- `setup.cfg` - 安装配置
- `MANIFEST.in` - 打包清单

## 使用说明

### 1. 数据准备
如果使用新的原始数据，运行：
```bash
python3 convert_train_data.py
```

### 2. 训练模型
运行GPU训练：
```bash
python3 train_gpu.py
```

### 3. 结果
- 模型保存位置: `/root/MMSA/saved_models/`
- 结果保存位置: `/root/MMSA/results/`

## 注意事项
- 确保已安装所需依赖（torch, librosa等）
- 数据文件较大（~494MB），请确保有足够存储空间
- GPU训练需要CUDA支持

