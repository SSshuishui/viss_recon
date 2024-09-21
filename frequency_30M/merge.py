import pandas as pd
import os

# 设置工作目录到CSV文件所在的文件夹
# os.chdir('block100')

# # 初始化一个空的DataFrame列表，用于存储每个文件的数据
# dfs = []

# # 遍历所有的CSV文件
# for i in range(99):  # 假设文件名从0到29
#     file_name = f'segment_{i}.csv'
#     print("reading: ", file_name)
#     # 读取CSV文件并添加到列表中
#     dfs.append(pd.read_csv(file_name))

# # 使用concat函数合并所有的DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)

# combined_df = combined_df[['u','v','w']]
# print(combined_df.shape)
# com = combined_df.groupby(['u', 'v', 'w']).size().reset_index(name='freq')
# print('grouped size: ', com.shape)

# # 保存合并后的DataFrame到一个新的CSV文件
# combined_df.to_csv('combined_0_48.csv', index=False)



# dfs = []

# file_name1 = "block200/combined_freq_gt_1.csv"
# file_name2 = "output_segments/segment_0_freq_gt_1.csv"
# file_name3 = "output_segments/segment_1_freq_gt_1.csv"
# file_name4 = "output_segments/segment_2_freq_gt_1.csv"
# file_name5 = "output_segments/segment_3_freq_gt_1.csv"

# dfs.append(pd.read_csv(file_name1))
# dfs.append(pd.read_csv(file_name2))
# dfs.append(pd.read_csv(file_name3))
# dfs.append(pd.read_csv(file_name4))
# dfs.append(pd.read_csv(file_name5))

# combined_df = pd.concat(dfs, ignore_index=True)

# combined_df.to_csv('uvwMap30M_130.csv', index=False)


import pandas as pd

# 读取新CSV文件
new_file_path = 'uvw226frequency30M.txt'
new_df = pd.read_csv(new_file_path, delim_whitespace=True, header=None, names=['u', 'v', 'w'])

# 初始化一个空的DataFrame来存储合并后的结果
merged_dfs = []

# 逐个读取旧CSV文件并进行合并
for i in range(100):
    file_path = f'block100/segment_{i}.csv'
    df = pd.read_csv(file_path)
    merge_df = pd.merge(new_df, df, on=['u', 'v', 'w'], how='inner')
    merged_dfs.append(merge_df)

# 合并所有旧文件的合并结果
final_merged_df = pd.concat(merged_dfs)

print("new_df: ", new_df.shape)
print("final_merged_df: ", final_merged_df.shape)

final_merged_df.to_csv('final_merged_df.csv', index=False)

# 检查新文件中的坐标是否在合并后的文件中出现过
all_in_merged = new_df.set_index(['u', 'v', 'w']).equals(final_merged_df.set_index(['u', 'v', 'w']))

if all_in_merged:
    print("所有新文件中的坐标都在旧文件中出现过。")
else:
    print("新文件中的一些坐标没有在所有旧文件中出现过。")