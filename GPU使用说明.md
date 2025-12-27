# GPU训练和测试使用说明

## 已完成的修改

### 1. 训练脚本 (`train_gpu.py`)
- ✅ 自动检测GPU可用性
- ✅ 支持GPU和CPU模式自动切换
- ✅ 如果GPU不可用，自动回退到CPU

### 2. 测试脚本 (`test_gpu.py`)
- ✅ 显示GPU信息
- ✅ 支持GPU模式配置

### 3. 代码修改
- ✅ 修改 `src/MMSA/run.py`，添加GPU检查
- ✅ 修改 `train_custom.py`，支持GPU模式

## 使用方法

### 方法1: 使用train_gpu.py脚本
```bash
python3 train_gpu.py
```

### 方法2: 使用命令行工具
```bash
python3 -m MMSA -m tfn -d custom -s 1111 -g 0 -n 4 -v 1
```
参数说明:
- `-g 0`: 使用GPU 0 (使用 `-g -1` 或省略GPU参数则使用CPU)
- `-n 4`: 使用4个数据加载workers
- `-s 1111`: 随机种子

### 方法3: 使用原始的train_custom.py
```bash
# 已修改为自动检测GPU
python3 train_custom.py
```

## GPU兼容性说明

**重要**: 当前环境使用的是 RTX 5090 GPU，其CUDA capability为 sm_120。

当前PyTorch版本可能不完全支持RTX 5090。如果遇到以下错误：
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案**:
1. **升级PyTorch**: 安装支持sm_120的PyTorch版本（需要PyTorch 2.5+）
2. **使用CPU模式**: 代码会自动检测并回退到CPU模式

## 验证GPU使用

运行以下命令检查GPU是否被使用：
```bash
# 检查训练时GPU使用情况
nvidia-smi

# 或者使用
watch -n 1 nvidia-smi
```

## 注意事项

1. **GPU内存**: 确保GPU有足够的显存
2. **CUDA版本**: 确保CUDA驱动和PyTorch CUDA版本匹配
3. **自动回退**: 如果GPU不可用或出现错误，代码会自动使用CPU

## 当前配置

- **GPU**: NVIDIA GeForce RTX 5090 (检测到但可能不完全兼容)
- **PyTorch版本**: 2.1.2+cu121
- **推荐**: 如需使用GPU，建议升级到PyTorch 2.5+

## 测试结果

模型已在CPU模式下成功训练，测试结果:
- MAE: 0.6159
- 相关系数: 0.3564
- 二分类准确率: 70.91%
- F1分数: 0.5884

