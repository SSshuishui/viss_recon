import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# 定义w轴的范围和分段
# w_ranges = [(-4999 + i * 1000, -4999 + (i + 1) * 1000) for i in range(10)]
w_ranges = [(-4999 + i * 200, -4999 + (i + 1) * 200) for i in range(50)]

def process_segment(filename, w_min, w_max):
    print("reading file: ", filename)
    # 读取txt文件并只处理属于当前w范围段的数据
    df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['u', 'v', 'w'])
    segment_df = df[(df['w'] >= w_min) & (df['w'] < w_max)]
    
    return segment_df

def process_and_save_segment(segment_idx, w_min, w_max, txt_files):
    # 初始化一个空DataFrame用于合并数据
    combined_segment_df = pd.DataFrame(columns=['u', 'v', 'w'])
    
    # 多线程处理文件
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 将每个文件的segment_df合并到combined_segment_df中
        for segment_df in executor.map(lambda file: process_segment(file, w_min, w_max), txt_files):
            combined_segment_df = pd.concat([combined_segment_df, segment_df], ignore_index=True)
    
    # 对合并后的数据进行一次groupby，统计所有文件中相同坐标的总频次
    combined_segment_df = combined_segment_df.groupby(['u', 'v', 'w']).size().reset_index(name='freq')
    

    # 保存合并后的数据
    if not combined_segment_df.empty:
        output_filename = f"block100/segment_{segment_idx}.csv"
        combined_segment_df[['u','v','w']].to_csv(output_filename, index=False)
        print(f"Segment {segment_idx} saved as {output_filename}")

        # 筛选并保存频次大于1的部分
        # freq_gt_1_df = combined_segment_df[combined_segment_df['freq'] > 1]
        # if not freq_gt_1_df.empty:
        #     output_filename_freq_gt_1 = f"block200/segment_{segment_idx}_freq_gt_1.csv"
        #     freq_gt_1_df.to_csv(output_filename_freq_gt_1, index=False)
        #     print(f"Segment {segment_idx} with freq > 1 saved as {output_filename_freq_gt_1}")


def main(input_dir):
    # 获取所有txt文件的路径
    txt_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith("uvw") and f.endswith('30M.txt')]
    
    # 遍历每个w轴范围段，依次处理并保存数据
    for idx, (w_min, w_max) in enumerate(w_ranges):
        print(f"Processing segment {idx}: w range {w_min} to {w_max}")
        process_and_save_segment(idx, w_min, w_max, txt_files)

if __name__ == "__main__":
    input_dir = "./"  # 设置为存放txt文件的目录
    main(input_dir)
