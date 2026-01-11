# MMSA 训练测试快速参考

## 🚀 快速开始

### 使用通用训练脚本（推荐）

```bash
# 基本用法
python train.py -m tfn -d copa_1231

# 跳过验证集
python train.py -m tfn -d copa_1231 --skip-validation

# 使用多个随机种子
python train.py -m tfn -d copa_1231 -s 1111 1112 1113
```

### 使用专用脚本

```bash
# COPA 1231 数据集专用脚本
python train_copa_1231.py

# 测试已保存的模型
python test_copa_1231.py
```

---

## 📋 支持的模型

所有 `singleTask` 模型：
- `tfn`, `lmf`, `mfn`, `graph_mfn`
- `ef_lstm`, `lf_dnn`
- `mult`, `misa`, `bert_mag`
- `mfm`, `mmim`, `mctn`
- `cenet`, `almt`, `almt_cider`

---

## 📊 支持的数据集

- `mosi`, `mosei`, `sims`, `simsv2`
- `custom`, `train_12_16`
- `copa_1231`

---

## 📖 详细文档

- **通用训练测试指南**: `通用训练测试指南.md`
- **COPA 1231 专用指南**: `训练测试指南.md`
- **完整流程说明**: `训练测试流程说明.md`

---

## 💡 常用命令

```bash
# 查看帮助
python train.py --help

# 训练不同模型
python train.py -m lmf -d copa_1231
python train.py -m misa -d copa_1231
python train.py -m almt -d copa_1231

# CPU 训练
python train.py -m tfn -d copa_1231 --gpu-ids ""

# 使用自定义配置
python train.py -m tfn -d copa_1231 --config config_example.json
```

---

**更多信息请查看详细文档！**


