import pandas as pd
import os

period = 5

file_names = [f"torch_10M/image{period}_{i}.txt" for i in range(27)]

total_lines = 0
# 确保所有文件都存在
for file_name in file_names:
    if not os.path.isfile(file_name):
        print(f"文件 {file_name} 不存在。")
    else:
        with open(file_name, 'r') as infile:
            lines = sum(1 for line in infile)  # 计算当前文件的行数
            total_lines += lines  # 累加到总行数
            print(f"文件 {file_name} 有 {lines} 行。")  # 打印当前文件的行数

print(f"所有文件合并完成，总行数为 {total_lines}。")

# 合并文件
with open(f"./torch_10M/combined_image{period}.txt", 'w') as outfile:
    for file_name in file_names:
        with open(file_name, 'r') as infile:
            outfile.write(infile.read())  # 每个文件内容后加一个换行符

print(f"文件已合并到 torch_10M/combined_image{period}.txt")


