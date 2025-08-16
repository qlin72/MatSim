import csv
import json
import numpy as np
import faiss
from collections import defaultdict

print("FAISS 版本:", faiss.__version__)


# 8 个类别
# categories = ["plastic", "rubber", "metal", "leather", "fabric", "wood", "stone", "ceramic"]
categories = ["plastic", "rubber","metal", "leather", "fabric", "wood", "stone", "ceramic", "bone", "cardboard", "concrete", "foliage", "fur", "gemstone", "glass", "paper", "soil", "sponge", "wax", "wicker"]
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

def update_accuracy(file_0, file_1):
    """ 计算每个类别的分类准确率，并保存到 CSV """
    cat_0 = get_category(file_0)
    cat_1 = get_category(file_1)

    if cat_0 and cat_1:
        class_counts[cat_0]["total"] += 1
        if cat_0 == cat_1:
            class_counts[cat_0]["correct"] += 1

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([file_0, cat_0, cat_1])

def calculate_accuracies():
    """ 计算每个类别的准确率，并计算总体平均准确率 """
    accuracies = {}
    total_accuracy = 0
    total_classes = 0

    for category in categories:
        total = class_counts[category]["total"]
        correct = class_counts[category]["correct"]
        if total > 0:
            accuracy = correct / total
            accuracies[category] = accuracy
            total_accuracy += accuracy
            total_classes += 1
        else:
            accuracies[category] = None

    avg_accuracy = total_accuracy / total_classes if total_classes > 0 else 0
    return accuracies, avg_accuracy

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
    json_file = "/home/qingran/Desktop/MatSim/logs/dms_descriptor_dict_20.json"  # 假设 JSON 文件已经存在
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

        update_accuracy(nm, best_match_file)
        print(f"Image: {nm} best match with {best_match_file}. With similarity: {distances[0][1]}")

    # === 4️⃣ 计算并输出准确率 ===
    accuracies, avg_accuracy = calculate_accuracies()

    print("\n各类别分类准确率：")
    for category, acc in accuracies.items():
        print(f"{category}: {acc:.2%}" if acc is not None else f"{category}: No Data")

    print(f"\n总体平均分类准确率: {avg_accuracy:.2%}")