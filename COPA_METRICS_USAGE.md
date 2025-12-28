# COPA心理范式准确率评估使用说明

## 一、功能概述

已在MMSA项目中集成了COPA（心理测试）范式准确率评估功能。该功能会在训练和测试过程中自动计算12个心理范式的准确率。

## 二、评估指标说明

### 2.1 COPA评估流程

1. **范式聚合**：将每个范式内的多个题目分数求和
2. **Z分数标准化**：使用常模数据（norm_data）进行标准化
3. **T分数转换**：转换为T分数（均值50，标准差10）
4. **等级划分**：将T分数映射到5个等级（低分、较低分、中等分、较高、高分）
5. **范围匹配**：判断预测T分数是否在真实标签对应的等级范围内

### 2.2 评估指标

- **COPA_overall_acc**: 整体准确率（所有范式的平均准确率）
- **COPA_P1_acc ~ COPA_P12_acc**: 每个范式的准确率

## 三、使用方法

### 3.1 自动评估

当使用`custom`或`train_12_16`数据集时，COPA评估会自动执行：

```python
from MMSA.run import MMSA_run

results = MMSA_run(
    model_name='tfn',
    dataset_name='custom',
    seeds=[1111],
    gpu_ids=[0]
)
```

在训练和测试过程中，评估结果会自动包含COPA指标：

```
MMSA - TEST-(tfn) >>  Has0_acc_2: 1.0000  Has0_F1_score: 1.0000  ...  COPA_overall_acc: 0.8500  COPA_P1_acc: 0.9000  ...
```

### 3.2 设置群体类型

COPA评估支持不同的群体类型（i1: 男犯, i2: 女犯, i3: 未成年）。默认使用`i1`。

如需修改，可以在配置文件中添加：

```python
config['copa_group_type'] = 'i2'  # 或 'i1', 'i3'
```

### 3.3 手动调用COPA评估

如果需要单独使用COPA评估功能：

```python
from MMSA.utils.metricsTop import MetricsTop
import numpy as np
import torch

# 创建评估器
metrics = MetricsTop('regression')

# 准备数据
y_pred = torch.tensor([...])  # 预测值
y_true = torch.tensor([...])  # 真实值
sample_indices = np.arange(len(y_pred))  # 样本索引

# 执行COPA评估
copa_results = metrics.eval_copa_paradigm_accuracy(
    y_pred, y_true,
    sample_indices=sample_indices,
    group_type='i1'  # 群体类型
)

print(copa_results)
# 输出: {'COPA_overall_acc': 0.85, 'COPA_P1_acc': 0.90, ...}
```

## 四、数据要求

### 4.1 样本组织

- 数据应按组组织，每组122个样本
- 每个样本需要有一个索引，用于确定属于哪个范式

### 4.2 预测值和真实值格式

- **回归模式**：预测值和真实值在[-1, 1]范围内，会自动转换为[0, 1]分类值
- **分类模式**：预测值和真实值已经是0或1

### 4.3 范式定义

12个范式的题目索引定义：

- **P1**: [0, 16, 32, 48, 64, 80, 96, 109] (8个题目)
- **P2**: [1, 17, 33, 49, 65, 81, 97, 110] (8个题目)
- **P3**: [3, 19, 35, 51, 67, 83, 99, 112] (8个题目)
- **P4**: [5, 21, 37, 53, 69, 85, 100, 113] (8个题目)
- **P5**: [23, 39, 55, 71, 87, 89, 102, 115] (8个题目)
- **P6**: [9, 25, 41, 57, 73, 74, 86, 117] (8个题目)
- **P7**: [11, 27, 43, 59, 75, 91, 105, 118] (8个题目)
- **P8**: [13, 29, 45, 61, 77, 93, 107, 120] (8个题目)
- **P9**: [7, 15, 31, 47, 63, 79, 95, 108] (8个题目)
- **P10**: [2, 18, 26, 34, 42, 44, 50, 52, 58, 66, 82, 90] (12个题目)
- **P11**: [4, 12, 20, 28, 36, 60, 68, 76, 84, 92, 104, 121] (12个题目)
- **P12**: [6, 10, 14, 22, 30, 38, 46, 54, 62, 70, 78, 94] (12个题目)

## 五、输出示例

训练/测试时的输出示例：

```
MMSA - TEST-(tfn) >>  
  Has0_acc_2: 1.0000  
  Has0_F1_score: 1.0000  
  Non0_acc_2: 1.0000  
  Non0_F1_score: 1.0000  
  Mult_acc_5: 0.9818  
  Mult_acc_7: 0.9818  
  MAE: 0.0855  
  Corr: 0.9913  
  Loss: 0.0788
  COPA_overall_acc: 0.8500
  COPA_P1_acc: 0.9000
  COPA_P2_acc: 0.8500
  ...
  COPA_P12_acc: 0.8000
```

## 六、注意事项

1. **数据完整性**：如果某个范式的题目数据不完整，该范式会被跳过
2. **样本索引**：确保样本索引正确，以便正确分配到各个范式
3. **群体类型**：根据实际数据选择合适的群体类型（i1/i2/i3）
4. **评估范围**：COPA评估只在`custom`和`train_12_16`数据集上自动执行

## 七、技术实现

### 7.1 代码位置

- **评估函数**: `src/MMSA/utils/metricsTop.py` - `eval_copa_paradigm_accuracy()`
- **集成位置**: `src/MMSA/trains/singleTask/TFN.py` - `do_test()`方法

### 7.2 修改内容

1. 在`MetricsTop`类中添加了`eval_copa_paradigm_accuracy()`方法
2. 在`TFN`训练器的`do_test()`方法中集成了COPA评估
3. 自动收集样本索引用于范式分配

## 八、测试

运行测试脚本验证功能：

```bash
python3 test_copa_metrics.py
```

该脚本会创建模拟数据并测试COPA评估功能。

