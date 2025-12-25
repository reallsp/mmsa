# COPA-PI数据集适配CIDer模型指南

## 概述

本指南说明如何将COPA-PI心理评估数据集适配到CIDer多模态情感分析模型。

## 数据集信息

- **数据集名称**: COPA-PI（用户心理评估个性分测验量表）
- **样本数量**: 366条
- **任务类型**: 二分类（0.0/1.0）
- **模态**: 文本、音频、视频、生理信号

## 适配步骤

### 步骤1: 数据转换

运行数据转换脚本，将原始pkl数据转换为CIDer期望的格式：

```bash
cd CIDer-main
python convert_data.py
```

**功能**:
- 将列表格式转换为字典格式（train/valid/test）
- 处理文本特征（BERT特征向量 -> 伪BERT token格式）
- 提取音频特征（波形 -> MFCC特征）
- 对齐序列长度
- 划分数据集（7:1.5:1.5）

**输出**:
- `./converted_data/copa_pi_aligned_50.pkl`

### 步骤2: 生成辅助数据

生成类别特征和概率文件（用于因果干预）：

```bash
python generate_cls_data.py
```

**功能**:
- 计算类别概率分布
- 计算每个类别的平均特征
- 保存为npy文件

**输出**:
- `./copa_pi_probs/iid/binary/copa_pi_train_binary.npy`
- `./copa_pi_feats/iid/binary/aligned/copa_pi_train_binary_aligned_{0,1}.npy`

### 步骤3: 配置数据路径

修改 `main_run.py` 中的数据路径：

```python
parser.add_argument('--data_path', type=str, default='./converted_data',
                    help='path for storing the dataset')
parser.add_argument('--dataset', type=str, default='copa_pi',
                    help='dataset to use')
```

### 步骤4: 运行训练

```bash
python main_run.py --dataset copa_pi --task binary --aligned True
```

## 主要修改点

### 1. 数据加载器 (`bert_dataloader.py`)

- 添加了 `__init_copa_pi()` 方法
- 支持二分类标签（0/1）
- 适配新的数据格式

### 2. 模型配置 (`main_run.py`)

- 添加了 `copa_pi_binary: 2` 到 `output_dim_dict`
- 添加了COPA-PI数据集的特征维度配置：
  - 文本: 768维
  - 音频: 13维（MFCC）
  - 视觉: 768维

### 3. 数据转换脚本 (`convert_data.py`)

- 文本特征转换：BERT特征向量 -> 伪BERT token格式
- 音频特征提取：波形 -> MFCC特征
- 序列对齐和填充

## 注意事项

1. **文本特征处理**: 
   - 原始数据中的text已经是BERT特征向量(768维)
   - 转换为伪BERT token格式以兼容现有代码
   - 如果需要，可以修改模型直接接受特征向量

2. **音频特征**:
   - 使用MFCC特征（13维）
   - 可以根据需要调整特征类型和维度

3. **视觉特征**:
   - 使用video_feature (197, 768)
   - 可以融合ir_feature如果需要

4. **标签**:
   - 二分类任务（0/1）
   - 已适配为binary分类

5. **类别不平衡**:
   - 标签0: 71.58%
   - 标签1: 28.42%
   - 建议使用类别权重或采样策略

## 训练参数建议

```bash
python main_run.py \
    --dataset copa_pi \
    --task binary \
    --aligned True \
    --batch_size 32 \
    --lr 1e-3 \
    --num_epochs 100 \
    --patience 10 \
    --data_path ./converted_data
```

## 故障排除

1. **音频特征提取失败**: 
   - 检查librosa是否安装: `pip install librosa`
   - 如果失败，会使用零特征填充

2. **内存不足**:
   - 减小batch_size
   - 减少序列长度

3. **数据格式错误**:
   - 检查转换后的pkl文件格式
   - 确认所有字段都存在

## 后续优化

1. **直接使用特征向量**: 修改模型以直接接受BERT特征向量，避免伪token转换
2. **多模态融合**: 考虑融合ir_feature和其他生理信号
3. **处理类别不平衡**: 使用加权损失或采样策略
4. **序列长度优化**: 根据实际数据调整序列长度

