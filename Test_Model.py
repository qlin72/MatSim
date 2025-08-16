# Given a folder of images (x.jpg) and their corresponding ROI masks (x_MASK.png) find for each image the most similar image in the folder (should run out of the box with sample folder)
# ...............................Imports..................................................................
import os
import csv
import numpy as np
import argparse
import RunNetOnFolder as run
from collections import defaultdict
##################################Input paramaters#########################################################################################
parser = argparse.ArgumentParser(description='Given a folder of images (x.jpg) and their corresponding ROI masks (x_MASK.png) find for each image the most similar image in the folder (should run out of the box with sample folder)')
parser.add_argument('--input_folder', default=r"/home/qingran/Desktop/mat_sim_test_dms_20", type=str, help='path to folder with images and masks')
parser.add_argument('--train_model_path', default= r"/home/qingran/Desktop/MatSim/logs/Defult.torch", type=str, help='path to trained model')
parser.add_argument('--max_img_size', default= 900, type=int, help=' max image size, larger images will be shrinked')
parser.add_argument('--min_img_size', default= 200, type=int, help=' min image size, smaller images will be resized')
parser.add_argument('--use_roi_mask', default= True, type=bool, help=' read roi mask of the object in from a file x_MASK.png where x.jpg is the image file, otherwise the mask will generated to cover the all image')
parser.add_argument('--crop', default= False, type=bool, help=' crop image around ROI mask')
parser.add_argument('--mask', default= True, type=bool, help=' mask image around ROI mask')
parser.add_argument('--UseAverageMaskUnMask', default= False, type=bool, help='')
parser.add_argument('--save_to_file', default= True, type=bool, help='Save descriptor to file')
args = parser.parse_args()

# Usage python Test_Model.py --input_folder sample_data/test  --train_model_path logs/Defult.torch

# categories = ["plastic", "rubber", "metal", "leather", "fabric", "wood", "stone", "ceramic"]
categories =  ["plastic", "rubber","metal", "leather", "fabric", "wood", "stone", "ceramic", "bone", "cardboard", "concrete", "foliage", "fur", "gemstone", "glass", "paper", "soil", "sponge", "wax", "wicker"]
class_counts = defaultdict(lambda: {"correct": 0, "total": 0})

# CSV 日志文件路径
csv_file = "dms_classification_results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "gt_material", "pred_material"])  # 表头

def get_category(filename):
    """ 从文件名中提取类别（第一个'_'后面的部分） """
    parts = filename.split("_")
    if len(parts) > 1:
        category = parts[1]  # 获取第一个'_'后面的部分
        return category if category in categories else None
    return None

def update_accuracy(file_0, file_1):
    """ 计算每个类别的分类正确率 """
    cat_0 = get_category(file_0)
    cat_1 = get_category(file_1)
    print(cat_0)
    print(cat_1)

    if cat_0 and cat_1:  # 确保提取的类别有效
        class_counts[cat_0]["total"] += 1  # 统计类别总数
        if cat_0 == cat_1:  # 如果分类正确
            class_counts[cat_0]["correct"] += 1
    
     # 追加写入 CSV 文件
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([file_0, cat_0, cat_1])

def calculate_accuracies():
    """ 计算每个类别的准确率和总的平均准确率 """
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
            accuracies[category] = None  # 没有数据时，返回 None

    # 计算平均准确率（排除无数据的类别）
    avg_accuracy = total_accuracy / total_classes if total_classes > 0 else 0

    return accuracies, avg_accuracy

if __name__ == "__main__":
    print("Generating Descriptors for ",args.input_folder, "\\n Using model ",args.train_model_path,"\n\n\n\n")
    desc_dict=run.Run(DataPath = args.input_folder, args=  args, Trained_model_path= args.train_model_path ,  outpath = "logs/dms_descriptor_dict_20.json")
    desc=desc_dict['descs']
    print("\n------  results---------\n Top 1 match list")
    for nm in desc: # match descriptor
        max_sim=-1
        max_sim_file=""
        for nm2 in desc:
            if nm2!=nm:
                dc1 = desc[nm]
                dc2 = desc[nm2]
                # print(dc2)
                if type(dc2)==str:
                    # dc1 = np.fromstring(dc1.replace('[', '').replace(']', ''), dtype=np.float16, sep=', ')
                    # dc1 = np.array(dc1, dtype=np.float16)
                    # print(dc1)
                    dc1 = dc1.replace('[', '').replace(']', '').replace(' ','')
                    dc1 = np.array(eval(dc1), dtype=np.float16)
                    # dc2 = np.fromstring(dc2.replace('[', '').replace(']', ''), dtype=np.float32, sep=', ')
                    # dc2 = np.array(dc2, dtype=np.float16)
                    # print(dc2)
                    dc2 = dc2.replace('[', '').replace(']', '').replace(' ','')
                    dc2 = np.array(eval(dc2), dtype=np.float16)
                sim = ((dc1 * dc2).sum()) # Get cosine similarity
                # print(sim)
                if sim>=max_sim:
                    max_sim=sim
                    max_sim_file = nm2
        update_accuracy(nm, max_sim_file)
        print("Image:",nm," best match with",max_sim_file,". With similarity:",str(max_sim))

    accuracies, avg_accuracy = calculate_accuracies()

    # 输出结果
    print("各类别分类准确率：")
    for category, acc in accuracies.items():
        print(f"{category}: {acc:.2%}" if acc is not None else f"{category}: No Data")

    print(f"\n总体平均分类准确率: {avg_accuracy:.2%}")

