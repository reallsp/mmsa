import pandas as pd
import os

# 1. 范式定义
paradigms = {
    "P1": [0, 16, 32, 48, 64, 80, 96, 109],
    "P2": [1, 17, 33, 49, 65, 81, 97, 110],
    "P3": [3, 19, 35, 51, 67, 83, 99, 112],
    "P4": [5, 21, 37, 53, 69, 85, 100, 113],
    "P5": [23, 39, 55, 71, 87, 89, 102, 115],
    "P6": [9, 25, 41, 57, 73, 74, 86, 117],
    "P7": [11, 27, 43, 59, 75, 91, 105, 118],
    "P8": [13, 29, 45, 61, 77, 93, 107, 120],
    "P9": [7, 15, 31, 47, 63, 79, 95, 108],
    "P10": [2, 18, 26, 34, 42, 44, 50, 52, 58, 66, 82, 90],
    "P11": [4, 12, 20, 28, 36, 60, 68, 76, 84, 92,104, 121],
    "P12": [6, 10, 14, 22, 30, 38, 46, 54, 62, 70, 78, 94]
}

# 2. 常模数据
norm_data = {
    "P1": {"i1":{"mean":5.1285,"std":2.3828},"i2":{"mean":4.6214,"std":2.4538},"i3":{"mean":5.4530,"std":2.1273}},
    "P2": {"i1":{"mean":4.3345,"std":2.4344},"i2":{"mean":4.2658,"std":2.4801},"i3":{"mean":4.2108,"std":2.4044}},
    "P3": {"i1":{"mean":6.9572,"std":1.4691},"i2":{"mean":7.2494,"std":1.2188},"i3":{"mean":6.4582,"std":1.7123}},
    "P4": {"i1":{"mean":2.1710,"std":1.9742},"i2":{"mean":2.3374,"std":2.0306},"i3":{"mean":2.6899,"std":2.0704}},
    "P5": {"i1":{"mean":4.1637,"std":2.5371},"i2":{"mean":4.0531,"std":2.5925},"i3":{"mean":4.9487,"std":2.3277}},
    "P6": {"i1":{"mean":3.7783,"std":2.2979},"i2":{"mean":3.3951,"std":2.2326},"i3":{"mean":4.2770,"std":2.2951}},
    "P7": {"i1":{"mean":3.1612,"std":2.3235},"i2":{"mean":2.8749,"std":2.1610},"i3":{"mean":3.1681,"std":2.2884}},
    "P8": {"i1":{"mean":3.0347,"std":2.3788},"i2":{"mean":3.0872,"std":2.4887},"i3":{"mean":3.2021,"std":2.3921}},
    "P9": {"i1":{"mean":3.9035,"std":2.3100},"i2":{"mean":3.7218,"std":2.4190},"i3":{"mean":4.0192,"std":2.1929}},
    "P10": {"i1":{"mean":3.8228,"std":3.3359},"i2":{"mean":2.3228,"std":2.3648},"i3":{"mean":4.7190,"std":3.4341}},
    "P11": {"i1":{"mean":1.5697,"std":1.8542},"i2":{"mean":1.1868,"std":1.6012},"i3":{"mean":2.2030,"std":2.0456}},
    "P12": {"i1":{"mean":4.9915,"std":3.1602},"i2":{"mean":3.9119,"std":2.7829},"i3":{"mean":5.6280,"std":2.9183}}
}

# T 范围
t_ranges = {
    "低分": {"min": 0, "max": 35},
    "较低分": {"min": 36, "max": 45},
    "中等分": {"min": 46, "max": 54},
    "较高": {"min": 55, "max": 64},
    "高分": {"min": 65, "max": 100}
}
def get_level(T):
    if T <= 35: return "低分"
    elif T <= 45: return "较低分"
    elif T < 55: return "中等分"
    elif T < 65: return "较高"
    else: return "高分"


def is_group_complete(df, idx_list):
    for i in idx_list:
        if i >= len(df):
            return False
        if pd.isna(df.loc[i, "pred_class"]) or pd.isna(df.loc[i, "label"]):
            return False
    return True


# ========== 输入 ==========  
folder = r"E:\MMSA\MMSA\classification\csv1"
group_type = "i1"   # i1 男犯 / i2 女犯 / i3 未成年

# ========== 新增：存放每个组每个范式的 T 信息 ==========
detailed_records = []

# 统计容器
stats = {p: {"correct": 0, "total": 0} for p in paradigms}
overall_correct = 0
overall_total = 0

print("\n开始批量处理...\n")

for file in os.listdir(folder):
    if not file.endswith(".csv"):
        continue

    print(f"\n📌 正在处理文件：{file}\n")

    df = pd.read_csv(os.path.join(folder, file))
    df["pred_class"] = pd.to_numeric(df["pred_class"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    group_size = 122
    num_groups = len(df) // group_size
    if num_groups == 0:
        num_groups = 1

    for g in range(num_groups):
        start, end = g * group_size, (g + 1) * group_size
        group_df = df.iloc[start:end].reset_index(drop=True)

        print(f"=== 第 {g+1} 组 ===")

        for paradigm, indices in paradigms.items():

            if not is_group_complete(group_df, indices):
                print(f"{paradigm}: ⚠ 缺失数据 → 跳过该范式统计")
                continue

            pred_xj = group_df.loc[indices, "pred_class"].sum()
            label_xj = group_df.loc[indices, "label"].sum()

            norm = norm_data[paradigm][group_type]
            pred_Z = (pred_xj - norm["mean"]) / norm["std"]
            label_Z = (label_xj - norm["mean"]) / norm["std"]

            pred_T = int(50 + 10 * pred_Z)
            label_T = int(50 + 10 * label_Z)

            # 五分类等级
            pred_level = get_level(pred_T)
            label_level = get_level(label_T)

            in_range = t_ranges[label_level]["min"] <= pred_T <= t_ranges[label_level]["max"]

            stats[paradigm]["total"] += 1
            overall_total += 1

            if in_range:
                stats[paradigm]["correct"] += 1
                overall_correct += 1

            print(f"{paradigm}: {'✅' if in_range else '❌'}  预测T={pred_T}, 预测等级{pred_level},标签T={label_T}, 等级={label_level}")

            # ========= 新增：保存详细记录 ==========
            detailed_records.append({
                                "file": file,
                                "group": g + 1,
                                "paradigm": paradigm,
                                "pred_T": pred_T,
                                "pred_level": pred_level,
                                "label_T": label_T,
                                "label_level": label_level,
                                "match": in_range
                            })


# ========== 输出准确率表 ==========
print("\n===============================")
print("📊 最终准确率统计")
print("===============================\n")

rows = []
for p in paradigms:
    total = stats[p]["total"]
    correct = stats[p]["correct"]
    acc = correct / total if total > 0 else 0
    rows.append([p, total, round(acc, 4)])

summary_df = pd.DataFrame(rows, columns=["范式", "参与样本数", "准确率"])
print(summary_df)

overall_acc = overall_correct / overall_total if overall_total > 0 else 0
print("\n===============================")
print(f"🎯 整体准确率： {overall_acc:.4f}")
print("===============================\n")

# ========== 保存所有范式的 T 详细结果 ==========
detail_df = pd.DataFrame(detailed_records)
output_path = r"E:\MMSA\MMSA\classification\T_outputs\T_value_detailed_results_12.7_4.csv"
detail_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n📄 已保存每组每范式的T值与五分类结果：\n{output_path}\n")
