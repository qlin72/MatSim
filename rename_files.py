import os

# 指定你的文件夹路径
folder_path = "/home/qingran/Desktop/mask"  # 修改成你的实际路径

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)
    
    # 确保它是文件（而不是子文件夹）
    if os.path.isfile(old_path):
        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_MASK{ext}"  # 添加"_MASK"
        new_path = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {filename} → {new_filename}")

print("✅ 所有文件重命名完成！")