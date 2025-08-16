
# import json

# # 读取 JSON 文件
# json_file = "/home/qingran/Desktop/MatSim/logs/descriptor_dict.json"
# with open(json_file, "r") as f:
#     desc_dict = json.load(f)

# # 只取前 5 条数据，防止输出过长
# subset_dict = dict(list(desc_dict.items())[:5])


# # 美观打印 JSON
# print(json.dumps(subset_dict, indent=4))

# import json

# # 读取 JSON 文件
# json_file = "/home/qingran/Desktop/MatSim/logs/descriptor_dict.json"
# with open(json_file, "r") as f:
#     desc_dict = json.load(f)

# # 只获取一条数据的键和值
# first_key = next(iter(desc_dict))
# print(f"📌 第一条数据的键: {first_key}")
# print(f"📌 第一条数据的值 (前 200 个字符)：\n{str(desc_dict[first_key])[:200]} ...")

import json

# 读取 JSON 文件
json_file = "/home/qingran/Desktop/MatSim/logs/descriptor_dict.json"

with open(json_file, "r") as f:
    desc_dict = json.load(f)

# 只取前 1 条数据，防止打印过多内容
subset_dict = {k: desc_dict[k] for k in list(desc_dict.keys())[:1]}

# 美观打印 JSON 结构
print(json.dumps(subset_dict, indent=4))