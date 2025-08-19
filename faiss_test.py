import csv
import json
import numpy as np
import faiss
from collections import defaultdict

print("FAISS 版本:", faiss.__version__)


# 8 个类别
# categories = ["plastic", "rubber", "metal", "leather", "fabric", "wood", "stone", "ceramic"]
# categories = ["plastic", "rubber","metal", "leather", "fabric", "wood", "stone", "ceramic", "bone", "cardboard", "concrete", "foliage", "fur", "gemstone", "glass", "paper", "soil", "sponge", "wax", "wicker"]
categories = ["fabric","foliage","glass","leather","metal","paper","plastic","stone","water","wood"]
class_counts = defaultdict(lambda: {"correct": 0, "total": 0})

# CSV 结果文件
csv_file = "dms_classification_results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "gt_material", "pred_material"])  # 表头

def get_category(filename):
    """ 从文件名中提取类别（第一个'_'后面的部分） """
    parts = filename.split("_")
    if len(parts) > 1:
        category = parts[1]
        return category if category in categories else None
    return None

num_classes = len(categories)
conf_mat = np.zeros((num_classes, num_classes), dtype=int)
cat2idx = {c: i for i, c in enumerate(categories)}


def update_confmat(file_0, file_1):
    """更新混淆矩阵"""
    gt = get_category(file_0)
    pred = get_category(file_1)
    if gt in cat2idx and pred in cat2idx:
        i, j = cat2idx[gt], cat2idx[pred]
        conf_mat[i, j] += 1

def calculate_miou_from_confmat():
    """从混淆矩阵计算每类 IoU 和 mean IoU"""
    ious = {}
    iou_sum, valid_classes = 0.0, 0
    for c, idx in cat2idx.items():
        tp = conf_mat[idx, idx]
        fp = conf_mat[:, idx].sum() - tp
        fn = conf_mat[idx, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou = tp / denom
            ious[c] = iou
            iou_sum += iou
            valid_classes += 1
        else:
            ious[c] = None
    mean_iou = iou_sum / valid_classes if valid_classes > 0 else 0.0
    return ious, mean_iou

def print_confmat_and_miou():
    """打印混淆矩阵 (rows=GT, cols=Pred) + per-class Accuracy/IoU + 宏平均"""
    num_classes = len(categories)
    cat2idx = {c: i for i, c in enumerate(categories)}

    # -------- 尺寸/对齐设置 --------
    name_w = max(6, max(len(c) for c in categories))       # 类名字宽
    max_count = int(conf_mat.max()) if conf_mat.size > 0 else 0
    cell_w = max(6, len(str(max_count)) + 2)               # 单元格宽
    def fmt_cell(x): return f"{x:>{cell_w}d}"

    # -------- 合计 --------
    row_sums = conf_mat.sum(axis=1)
    col_sums = conf_mat.sum(axis=0)
    total_sum = int(row_sums.sum())

    # -------- 计算 per-class 指标 --------
    per_acc = {}   # TP / (TP + FN) = TP / 行和
    per_iou = {}   # TP / (TP + FP + FN)
    acc_sum = 0.0; acc_cnt = 0
    iou_sum = 0.0; iou_cnt = 0

    for i, c in enumerate(categories):
        tp = conf_mat[i, i]
        fp = col_sums[i] - tp
        fn = row_sums[i] - tp

        # accuracy（行召回）
        acc_denom = tp + fn
        if acc_denom > 0:
            acc = tp / acc_denom
            per_acc[c] = acc
            acc_sum += acc
            acc_cnt += 1
        else:
            per_acc[c] = None

        # IoU
        iou_denom = tp + fp + fn
        if iou_denom > 0:
            iou = tp / iou_denom
            per_iou[c] = iou
            iou_sum += iou
            iou_cnt += 1
        else:
            per_iou[c] = None

    mean_acc = (acc_sum / acc_cnt) if acc_cnt > 0 else 0.0
    mean_iou = (iou_sum / iou_cnt) if iou_cnt > 0 else 0.0

    # -------- 打印混淆矩阵 --------
    print("\n混淆矩阵 (rows=GT, cols=Pred):")
    header = " " * (name_w + 1) + "".join(f"{c:>{cell_w}s}" for c in categories) + f"{'Total':>{cell_w}s}"
    print(header)
    for i, c in enumerate(categories):
        row_cells = "".join(fmt_cell(int(conf_mat[i, j])) for j in range(num_classes))
        print(f"{c:<{name_w}s} " + row_cells + fmt_cell(int(row_sums[i])))
    print(f"{'Total':<{name_w}s} " + "".join(fmt_cell(int(col_sums[j])) for j in range(num_classes)) + fmt_cell(total_sum))

    # -------- 打印 per-class Accuracy & IoU --------
    print("\n各类别指标：")
    name_w2 = max(name_w, len("category"))
    print(f"{'category':<{name_w2}s}  {'accuracy':>10s}  {'iou':>10s}")
    for c in categories:
        acc = per_acc[c]; iou = per_iou[c]
        acc_str = f"{acc:.4%}" if acc is not None else "No Data"
        iou_str = f"{iou:.4%}" if iou is not None else "No Data"
        print(f"{c:<{name_w2}s}  {acc_str:>10s}  {iou_str:>10s}")

    print(f"\n总体平均分类准确率 (macro): {mean_acc:.4%}")
    print(f"总体平均 IoU (macro): {mean_iou:.4%}")


def parse_vector(subdict):
    """ 解析 JSON 结构中的 descriptor 向量 """
    if isinstance(subdict, dict) and "descriptor" in subdict:
        descriptor_data = subdict["descriptor"]

        # 如果 descriptor 是字符串，需要处理
        if isinstance(descriptor_data, str):
            descriptor_data = descriptor_data.replace('[', '').replace(']', '').replace(' ', '')
            try:
                return np.array(list(map(np.float32, descriptor_data.split(','))), dtype=np.float32)
            except:
                return None  # 解析失败返回 None

        # 如果 descriptor 已经是列表，直接转换
        elif isinstance(descriptor_data, list):
            return np.array(descriptor_data, dtype=np.float32)

    return None  # 数据异常返回 None

if __name__ == "__main__":
    # === 1️⃣ 读取 descriptor_dict.json 文件 ===
    json_file = "/home/qingran/Desktop/MatSim/logs/fmd_descriptor_dict.json"  # 假设 JSON 文件已经存在
    with open(json_file, "r") as f:
        desc_dict = json.load(f)
    
    sub_dict = desc_dict['descs']

    print("\n------  results---------\n Using FAISS for Nearest Neighbor Search")
    # print(desc_dict)
    # === 2️⃣ 构建 FAISS 索引 ===
    vector_names = list(sub_dict.keys())  # 文件名列表
    print(vector_names)
    print(len(vector_names))

    data_vectors = []
    for nm in sub_dict:
        dc = sub_dict[nm]
        dc = dc.replace('[', '').replace(']', '').replace(' ','')
        dc = np.array(eval(dc), dtype=np.float16)
        data_vectors.append(dc)
    
    data_vectors = np.vstack(data_vectors)


    d = data_vectors.shape[1]  # 计算向量维度
    print("d:",d)
    num_vectors = len(data_vectors)  # 向量数量
    print("num_vectors:",num_vectors)

    # 初始化 FAISS 索引 (IndexFlatL2 - 适用于小数据)
    index = faiss.IndexFlatL2(d)
    index.add(data_vectors)  # 添加所有向量

    # === 3️⃣ 进行最近邻搜索 ===
    for i, nm in enumerate(vector_names):
        query_vector = data_vectors[i].reshape(1, -1)  # 1xD 形状
        distances, indices = index.search(query_vector, k=2)  # k=2 因为第一个是自己

        best_match_idx = indices[0][1]  # 取最近邻（跳过自己）
        best_match_file = vector_names[best_match_idx]

        update_confmat(nm, best_match_file)
        print(f"Image: {nm} best match with {best_match_file}. With similarity: {distances[0][1]}")

    print_confmat_and_miou()