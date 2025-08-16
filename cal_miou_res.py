import pandas as pd
import numpy as np
from collections import defaultdict

# 8个类别
categories = ["plastic", "rubber","metal", "leather", "fabric", "wood", "stone", "ceramic", "bone", "cardboard", "concrete", "foliage", "fur", "gemstone", "glass", "paper", "soil", "sponge", "wax", "wicker"]

# 加载 CSV
csv_file = "dms_classification_results.csv"  # 确保文件存在
df = pd.read_csv(csv_file)

# 统计 IoU 和 Accuracy 相关数据
stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

for _, row in df.iterrows():
    gt = row["gt_material"]
    pred = row["pred_material"]

    if gt in categories and pred in categories:
        if gt == pred:
            stats[gt]["TP"] += 1  # 预测正确
        else:
            stats[gt]["FN"] += 1  # 真实类别被错误分类
            stats[pred]["FP"] += 1  # 误预测为该类别

# 计算 IoU, Accuracy, mIoU 和 平均 Accuracy
ious, accuracies = {}, {}
valid_ious, valid_accuracies = [], []

print("\n类别统计结果：")
print(f"{'类别':<10} {'IoU':<10} {'Accuracy':<10}")

for category in categories:
    TP, FP, FN = stats[category]["TP"], stats[category]["FP"], stats[category]["FN"]
    
    # 计算 IoU
    iou_denom = TP + FP + FN
    iou = TP / iou_denom if iou_denom > 0 else None
    ious[category] = iou
    if iou is not None:
        valid_ious.append(iou)
    
    # 计算 Accuracy
    acc_denom = TP + FN
    acc = TP / acc_denom if acc_denom > 0 else None
    accuracies[category] = acc
    if acc is not None:
        valid_accuracies.append(acc)

    # 打印每个类别的结果
    iou_str = f"{iou:.2%}" if iou is not None else "No Data"
    acc_str = f"{acc:.2%}" if acc is not None else "No Data"
    print(f"{category:<10} {iou_str:<10} {acc_str:<10}")

# 计算 mIoU 和 平均 Accuracy
miou = np.mean(valid_ious) if valid_ious else 0
mean_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0

print(f"\nMean IoU (mIoU): {miou:.2%}")
print(f"Mean Accuracy: {mean_accuracy:.2%}")