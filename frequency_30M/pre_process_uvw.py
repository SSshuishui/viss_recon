# import pandas as pd
# import os

# # 读取uvwMap30M_130.csv文件并创建一个字典
# freq_map = pd.read_csv('uvwMap30M_130.csv')
# freq_dict = dict(zip(freq_map[['u', 'v', 'w']].apply(tuple, axis=1), freq_map['freq']))

# # 假设txt文件都放在同一个目录下
# txt_dir = './'
# txt_files = [f for f in os.listdir(txt_dir) if f.endswith('30M.txt')]

# # 遍历每个txt文件
# for txt_file in txt_files:
#     # 读取txt文件
#     file_path = os.path.join(txt_dir, txt_file)
#     print("processing: ", file_path)
#     df = pd.read_csv(file_path, header=None, names=['u', 'v', 'w'])

#     # 为每个坐标点添加freq列
#     df['freq'] = df[['u', 'v', 'w']].apply(lambda x: freq_dict.get((x['u'], x['v'], x['w']), 1))

#     # 写入新的频次列到新的txt文件
#     new_file_path = os.path.join(txt_dir, f'updated_{txt_file}')
#     df.to_csv(new_file_path, sep='\t', index=False, header=False)


import pandas as pd
import os

# 读取uvwMap30M_130.csv文件并创建一个字典
freq_map = pd.read_csv('uvwMap30M_130.csv')
freq_dict = dict(zip(freq_map[['u', 'v', 'w']].apply(tuple, axis=1), freq_map['freq']))

# 假设txt文件都放在同一个目录下
txt_dir = './'
txt_files = [f for f in os.listdir(txt_dir) if f.endswith('30M.txt')]

# 遍历每个txt文件
for txt_file in txt_files:
    # 读取txt文件，假设列是u, v, w，并且没有表头
    file_path = os.path.join(txt_dir, txt_file)
    df = pd.read_csv(file_path, header=None, names=['u', 'v', 'w'])

    # 为每个坐标点添加freq列，使用apply函数和lambda表达式
    df['freq'] = df.apply(lambda row: freq_dict.get((row['u'], row['v'], row['w']), 1), axis=1)

    # 写入新的频次列到新的txt文件
    new_file_path = os.path.join(txt_dir, f'updated_{txt_file}')
    df.to_csv(new_file_path, sep='\t', index=False, header=False)