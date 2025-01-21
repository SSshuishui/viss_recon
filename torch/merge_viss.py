import pandas as pd
import os

period = 6
file_names = 18

# 1. 合并
# 创建一个空的 DataFrame 用于存储合并后的数据
combined_df = pd.DataFrame(columns=['viss_real', 'viss_imag'])
# 加载并合并每个 CSV 文件
for index in range(file_names):
    filename = f"torch_10M/viss{period}_{index}.csv"
    df = pd.read_csv(filename)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# 将合并后的 DataFrame 保存为一个新的 CSV 文件
combined_df.to_csv(f"torch_10M/viss{period}.csv", index=False)


# 2. 验证
directory = "torch_10M"
# 初始化累计行数
total_rows = 0

# 遍历所有CSV文件并累计行数
for index in range(file_names):
    filename = f"{directory}/viss{period}_{index}.csv"
    # 检查文件是否存在
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        total_rows += len(df)
    else:
        print(f"文件 {filename} 不存在。")
        break

combined_filename = f"{directory}/viss{period}.csv"
if os.path.exists(combined_filename):
    combined_df = pd.read_csv(combined_filename)
    combined_rows = len(combined_df)

print(f"所有文件的累计行数：{total_rows}")
print(f"合并后的文件行数：{combined_rows}")
# 比较累计行数与合并后的行数
if total_rows == combined_rows:
    print("合并正确：所有文件的累计行数与合并后的文件行数相匹配。")
else:
    print("合并错误：所有文件的累计行数与合并后的文件行数不匹配。")
    print(f"预期行数：{total_rows}, 实际行数：{combined_rows}")