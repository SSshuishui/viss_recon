import pandas as pd
import os
import time

# 定义原始的uvwMap文件路径
uvw_map_path = 'uvwMap10M_50_half.csv'

# 读取uvwMap文件，忽略第一行
uvw_map = pd.read_csv(uvw_map_path)

# 定义原始文件和更新后文件的文件夹路径
input_folder = './'
output_folder = './'

# 遍历文件
start_time = time.time()
for i in range(1, 51):
    # 构建文件名
    input_file = f'uvw{i}frequency10M_half.txt'
    output_file = f'updated_{input_file}'
    
    # 读取当前文件
    uvw_data = pd.read_csv(os.path.join(input_folder, input_file), delimiter=' ', header=None, names=['u', 'v', 'w'])
    
    # 初始化频次为1
    uvw_data['freq'] = 1

    # 合并数据，以u, v, w为键
    merged_data = pd.merge(uvw_data, uvw_map, on=['u', 'v', 'w'], how='left', suffixes=('', '_map'))
    
    # 使用map中的频次更新，如果map中没有则保持为1
    merged_data['freq'] = merged_data['freq_map'].combine_first(merged_data['freq'])
    
    # 选择需要的列
    updated_data = merged_data[['u', 'v', 'w', 'freq']]
    
    # 保存更新后的数据到新文件
    updated_data.to_csv(os.path.join(output_folder, output_file), index=False, sep=' ', header=False)

    print(f"文件 {input_file} 更新完成。")

print("所有文件已更新完成。")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")     # 28247.78s