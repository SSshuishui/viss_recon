import torch
import pandas as pd
import time

def visscal(uvw_file, l_df, m_df, n_df, C_df, constant1, period):
    RES = 2094
    dl = 2*RES/(RES-1)
    dm = 2*RES/(RES-1)
    dn = 2*RES/(RES-1)

    # 记录生成开始时间
    start_time = time.time()

    # 读取uvw文件
    uvw_df = pd.read_csv(uvw_file, delimiter=' ', usecols=[0,1,2], header=None, names=['u', 'v', 'w'])
    print("读取 uvw 完毕")
    
    # 将数据转换为PyTorch张量并移动到GPU上
    uvw_tensor = torch.tensor(uvw_df[['u', 'v', 'w']].values, device='cuda', dtype=torch.float32)
    l_tensor = torch.tensor(l_df['l'].values, device='cuda', dtype=torch.float32)
    m_tensor = torch.tensor(m_df['m'].values, device='cuda', dtype=torch.float32)
    n_tensor = torch.tensor(n_df['n'].values, device='cuda', dtype=torch.float32)
    C_tensor = torch.tensor(C_df['C'].values, device='cuda', dtype=torch.float32)
    print("uvw_tensor: ", uvw_tensor.shape)
    print("l_tensor: ", l_tensor.shape)
    print("m_tensor: ", m_tensor.shape)
    print("n_tensor: ", n_tensor.shape)
    print("C_tensor: ", C_tensor.shape)

    # 初始化结果张量
    viss_tensor = torch.zeros((len(uvw_tensor), 2), dtype=torch.float32, device='cuda')

    # 遍历 uvw_df 中的每一行
    for index, row in enumerate(uvw_tensor):
        row_start_time = time.time()
        
        # 计算复数指数部分
        temp_y = constant1 * (row[0]*l_tensor/dl + row[1]*m_tensor/dm + row[2]*(n_tensor-1)/dn)

        viss = torch.sum(torch.exp(temp_y) * C_tensor)

        # 去除相位
        viss2 = viss * torch.exp(constant1 * row[2]/dn)

        viss_tensor[index, 0] = viss2.real
        viss_tensor[index, 1] = viss2.imag

        if index == 0:
            print('viss_tensor: ', viss_tensor)
   
        row_end_time = time.time()
        row_elapsed_time = row_end_time - row_start_time
        print("Generated row {} in {:.4f} seconds".format(index, row_elapsed_time))
    
    # 记录生成结束时间
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print("Generated all rows in {:.4f} seconds".format(total_elapsed_time))
    
    # 将结果转换为DataFrame并返回
    viss_df = pd.DataFrame(viss_tensor.cpu().numpy(), columns=['viss_real', 'viss_imag'])

    # 将结果保存到 CSV 文件
    viss_df.to_csv(f"frequency_1M_stride01s_sample2400_2/viss_exp{period}.csv", index=False)

    print(f"结果已保存到 viss_exp{period}.csv 文件中。")


def main():
    # matlab中pi是3.1416
    constant1 = -2 * torch.pi * 1j
    
    for period in range(1, 3): 
        print(f"开始处理第{period}个周期")

        uvw_file = f"frequency_1M_stride01s_sample2400_2/uvw{period}frequency1M.txt"
        l_df = pd.read_csv('frequency_1M_stride01s_sample2400_2/l.txt', header=0)
        m_df = pd.read_csv('frequency_1M_stride01s_sample2400_2/m.txt', header=None, names=['m'])
        n_df = pd.read_csv('frequency_1M_stride01s_sample2400_2/n.txt', header=None, names=['n'])
        l_df.columns = ['l']
        C_df = pd.read_csv('frequency_1M_stride01s_sample2400_2/C.txt', sep=' ', usecols=[0], names=['C'])

        print("读取l m n C完毕")

        visscal(uvw_file, l_df, m_df, n_df, C_df, constant1, period)


if __name__ == "__main__":
    main()


# lines = 14276713  # 总行数
# time_per_line = 0.1521  # 每行生成时间（秒）

# # 计算总时间（小时）
# total_time_seconds = lines * time_per_line
# total_time_hours = total_time_seconds / 3600
# print("生成所有行所需的时间：", total_time_hours, "小时")
# total_time_days = total_time_hours / 24
# print("生成所有行所需的时间：", total_time_days, "天")


