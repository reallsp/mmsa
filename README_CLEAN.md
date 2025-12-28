# 项目清理说明

## 已删除的文件和目录

### 1. 日志文件
- `*.log` - 所有训练日志文件
- `tfn-custom.log` - 模型日志

### 2. Python缓存
- `__pycache__/` - Python字节码缓存目录
- `*.pyc` - Python编译文件

### 3. 其他项目代码
- `CIDer-main/` - 另一个项目的代码（2.7MB）

### 4. 原始数据文件
- `train_12.16_1.pkl` - 原始数据文件（491MB），已保留转换后的版本

### 5. 临时文档和脚本
- `train_custom.py` - 重复的训练脚本（已保留train_gpu.py）
- `适配方案.md` - 开发过程中的临时文档
- `进度记录.md` - 开发过程中的临时文档
- `GPU使用说明.md` - 临时说明文档
- `评估结果（分数）的计算规则更新.doc` - 临时文档

## 保留的核心文件

### 数据文件
- `train_12.16_1_converted.pkl` - 转换后的训练数据（494MB）

### 脚本文件
- `convert_train_data.py` - 数据格式转换脚本
- `train_gpu.py` - GPU训练脚本（主训练脚本）

### 源代码
- `src/MMSA/` - MMSA框架核心代码（完整的模型和训练框架）

### 配置文件
- `README.md` - 项目说明文档
- `LICENSE` - 许可证
- `pyproject.toml`, `setup.cfg`, `MANIFEST.in` - Python项目配置
- `PROJECT_STRUCTURE.md` - 项目结构说明（新创建）
- `.gitignore` - Git忽略文件（新创建）

## 清理统计

- **删除文件总数**: 约10+个文件
- **释放空间**: 约500MB（主要是原始数据文件）
- **保留核心代码**: 完整保留MMSA框架和训练脚本

## 使用说明

详细的项目结构和使用说明请参考 `PROJECT_STRUCTURE.md`。

