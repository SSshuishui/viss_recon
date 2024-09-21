# import numpy as np 
# import pandas as pd
# import csv

# def read_in_chunks(file_objects, lines_per_chunk=100):
#     """Generate data from multiple file objects in chunks of lines."""
#     while True:
#         chunks = []
#         for file_object in file_objects:
#             lines = []
#             for _ in range(lines_per_chunk):
#                 line = file_object.readline().strip()
#                 if not line:
#                     break
#                 lines.append(float(line))
#             if not lines:
#                 break
#             chunks.append(lines)
#         if not chunks:
#             break
#         yield chunks


# def find_coordinates_index(merged_df, coordinates):
#     coordinates_index = []
#     for coordinate in coordinates:
#         mask = (merged_df == coordinate).all(axis=1)
#         indexes = merged_df.index[mask]
#         coordinates_index.extend(indexes)
#     return coordinates_index


# filenames = ['tmp/l1.txt', 'tmp/m1.txt', 'tmp/n1.txt', 'lmn10M/FF.txt']

# lines_per_chunk = 50000000  # 设置每次读取的行数


# # 打开所有文件并创建文件对象列表
# file_objects = [open(filename, 'r') for filename in filenames]

# # 使用生成器函数读取数据
# for i, chunks in enumerate(read_in_chunks(file_objects, lines_per_chunk)):
#     # 对每次读取的结果进行处理
#     lmn_data = pd.DataFrame(np.column_stack(chunks), columns=['l','m','n','FF'])
#     print(f"{i+1} chunk loaded!")
#     # 每一组单独去重，然后添加
#     result_df = lmn_data.groupby(['l', 'm', 'n']).agg({'FF': 'mean'}).reset_index()

#     result_df.to_csv(f"lmn10M/unique_FF_{(i+1)}chunk.csv", index=0)
#     print(f"{i+1} uniqued saved")


# # 关闭所有文件对象
# for file_object in file_objects:
#     file_object.close()



# ===========================================================================================================================

# import pandas as pd

# df1 = pd.read_csv('lmn10M/unique1_4.csv')
# df2 = pd.read_csv('lmn10M/unique5_9.csv')

# # 合并两个DataFrame并按行去重
# merged_df = pd.concat([df1, df2]).groupby(['l', 'm', 'n']).agg({'FF': 'mean'}).reset_index()

# # 将合并后的结果保存为新的CSV文件
# merged_df.to_csv('lmn10M/unique1_9.csv', index=False)

# print(merged_df.shape)


# 保存lmn
# import pandas as pd
# # 读取CSV文件
# df = pd.read_csv('lmn10M/lmnC.csv', header=0)

# # 提取FF列
# l_column = df['l']
# # 将数据保存到FF.txt文件中
# with open('lmn10M/l.txt', 'w') as f:
#     for value in l_column:
#         f.write(str(value) + '\n')
# f.close()
# print("l列已保存到l.txt文件中")
# del l_column


# m_column = df['m']
# with open('lmn10M/m.txt', 'w') as f:
#     for value in m_column:
#         f.write(str(value) + '\n')
# f.close()
# print("m列已保存到m.txt文件中")
# del m_column

# n_column = df['n']
# with open('lmn10M/n.txt', 'w') as f:
#     for value in n_column:
#         f.write(str(value) + '\n')
# f.close()
# print("n列已保存到n.txt文件中")



# ===========================================================================================================================

# 生成nt
# import numpy as np
# import numpy as np

# # 从文件加载列向量
# nt = np.loadtxt('n.txt')

# # 将nt复制到nt_copy
# nt_copy = np.copy(nt)

# # 将nt_copy中等于0的元素替换为1
# nt_copy[nt_copy == 0] = 1

# print(nt_copy)

# # 将结果保存到新文件nt.txt中
# np.savetxt('lmn10M/nt.txt', n_values, fmt='%1.2f')

# print("处理完成，并保存到nt.txt文件中")


# ===========================================================================================================================


# uvw处理
# import pandas as pd
# import numpy as np
# import os
# import glob

# def process_uvw_file(input_file, output_file, frequency_df):
#     uvw_df = pd.read_csv(input_file, delimiter=' ', header=None, names=['u', 'v', 'w'])
#     merged_df = pd.merge(uvw_df, frequency_df, on=['u', 'v', 'w'], how='left')
#     merged_df['frequency'] = merged_df['frequency'].fillna(1)
#     merged_df.to_csv(output_file, sep=' ', index=False, header=False)

# def main():
#     frequency_df = pd.read_csv('./lmn10M/uvwMapFrequency.txt', delimiter=' ', header=None, names=['u', 'v', 'w', 'frequency'])

#     output_directory = './uvwMap/'
#     # 如果目录不存在，则创建
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     for uvw_file in glob.glob('../viss_1208/frequency10M/*.txt'):
#         output_file = os.path.join(output_directory, os.path.basename(uvw_file).replace('.txt', '_new.txt'))
#         process_uvw_file(uvw_file, output_file, frequency_df)
#         print(f"已生成新文件：{output_file}")

# if __name__ == "__main__":
#     main()


# ===========================================================================================================================


# 预处理lmn，除上dl dm dn
# import pandas as pd

# RES = 20940

# # 读取文件并对每个值进行操作
# def process_file(filename):
#     # 读取文件
#     df = pd.read_csv(filename, header=None)
#     # 对每个值进行操作
#     df = df.apply(lambda x: x / (2 * RES / (RES - 1)))
#     return df

# # 文件名列表
# file_list = ['lmn10M/l.txt', 'lmn10M/m.txt', 'lmn10M/n.txt']

# # 对每个文件进行操作并保存新文件
# for filename in file_list:
#     processed_df = process_file(filename)
#     new_filename = filename.split('.')[0] + '_new.txt'
#     print(f"Processed {filename} and saved as {new_filename}:")
#     print(processed_df)
#     # 保存新文件
#     processed_df.to_csv(new_filename, index=False, header=False)


# ===========================================================================================================================


# # 更新 lmnC.csv
# import pandas as pd

# # 读取lmnC.csv文件
# lmnC_df = pd.read_csv('lmn10M/lmnC.csv')

# # 处理l_new.txt, m_new.txt, n_new.txt文件
# for file_name in ['lmn10M/l.txt', 'lmn10M/m.txt', 'lmn10M/n.txt']:
#     # 读取当前文件
#     with open(file_name, 'r') as f:
#         data = f.readlines()
    
#     # 根据文件名更新对应列
#     col_name = file_name.split('_')[0]  # 提取列名
#     lmnC_df[col_name] = [float(x.strip()) for x in data]

# # 保存更新后的数据到lmnC_new.csv
# lmnC_df.to_csv('lmnC_new.csv', index=False)

# print("更新后的lmnC_new.csv文件已保存。")



# -------------------------------------------------------------------------
import os

# 指定包含.txt文件的目录
directory = './frequency_10M/uvw_seg/'

# 新的合并文件的名称
output_file = 'uvwMap1M_130.txt'

# 打开输出文件准备写入
with open(output_file, 'w') as outfile:
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为.txt文件
        if filename.endswith('.txt'):
            # 构建文件的完整路径
            filepath = os.path.join(directory, filename)
            # 打开每个.txt文件并读取内容
            with open(filepath, 'r') as infile:
                # 将文件内容写入到输出文件中
                outfile.write(infile.read())

print(f'All .txt files in {directory} have been merged into {output_file}.')