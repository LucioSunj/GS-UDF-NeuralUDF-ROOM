import os
import shutil

# 定义变量n和x
n = 140  # 生成10张图片
x = 4  # 文件名总长度为4位

# 源文件名
source_filename = "0001.jpg"

# 检查源文件是否存在
if not os.path.isfile(source_filename):
    print(f"源文件 {source_filename} 不存在。")
else:
    for i in range(2, n + 2):
        # 生成目标文件名，老是指示有几个位数不足高位补零
        target_filename = f"{i:0{x}d}.jpg"

        # 复制文件
        shutil.copyfile(source_filename, target_filename)
        print(f"{source_filename} 复制为 {target_filename}")

print("复制完成。")