import torch
import pandas as pd
import numpy as np
import time
from cufinufft import Plan

def imagerecon(uvw_file, viss_file, l_df, m_df, n_df, constant2, period):
    # 读取viss文件
    viss_df = pd.read_csv(viss_file)
    viss_df['viss'] = viss_df['viss_real'] + viss_df['viss_imag']*1j
    print("读取 viss 完毕")

    # 读取uvw文件
    uvw_df = pd.read_csv(uvw_file, delimiter=' ', header=None, names=['u', 'v', 'w'])
    print("读取 uvw 完毕")

    RES = 2094
    dl = 2*RES/(RES-1)
    dm = 2*RES/(RES-1)
    dn = 2*RES/(RES-1)

    # 记录生成开始时间
    start_time = time.time()
    
    # 将数据转换为PyTorch张量并移动到GPU上
    viss_tensor = torch.tensor(viss_df['viss'].values, device='cuda')
    uvw_tensor = torch.tensor(uvw_df[['u', 'v', 'w']].values, device='cuda', dtype=torch.float32)
    l_tensor = torch.tensor(l_df['l'].values, device='cuda', dtype=torch.float32)
    m_tensor = torch.tensor(m_df['m'].values, device='cuda', dtype=torch.float32)
    n_tensor = torch.tensor(n_df['n'].values, device='cuda', dtype=torch.float32)
    print("uvw_tensor: ", uvw_tensor.shape)
    print("l_tensor: ", l_tensor.shape)
    print("m_tensor: ", m_tensor.shape)
    print("n_tensor: ", n_tensor.shape)

    print(viss_tensor)

    # 使用 cuFINUFFT 进行非均匀傅里叶逆变换
    plan = Plan(1, [len(uvw_tensor), len(uvw_tensor), len(uvw_tensor)], ntransf=1, eps=1e-6, gpu_method=1)
    plan.setpts(uvw_tensor[:, 0], uvw_tensor[:, 1], uvw_tensor[:, 2])

    # 执行非均匀傅里叶逆变换
    output = plan.execute(viss_tensor, iflag=-1) / len(uvw_tensor)

    # 将输出转换为实际的影像重建结果
    image_tensor = torch.abs(output)
    
    # 记录生成结束时间
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print("Generated all rows in {:.4f} seconds".format(total_elapsed_time))
    
    # 将结果转换为DataFrame并返回
    image_df = pd.DataFrame(image_tensor.cpu().numpy(), columns=['imagerecon_real'])

    # 将结果保存到 CSV 文件
    image_df.to_csv(f"F_recon_1M_stride01s_sample2400_2/image{period}.txt", sep='\t', header=False, index=False)

    print(f"结果已保存到 image{period}.txt 文件中。")


def main():
    # matlab中pi是3.1416
    constant1 = -2 * torch.pi * 1j
    constant2 = 2 * torch.pi * 1j
    
    for period in range(1, 3): 
        print(f"开始处理第{period}个周期")

        uvw_file = f"frequency_1M_stride01s_sample2400_2/uvw{period}frequency1M.txt"
        viss_file = f"frequency_1M_stride01s_sample2400_2/viss_exp{period}.csv"
        
        l_df = pd.read_csv('frequency_1M_stride01s_sample2400_2/l.txt', header=0)
        m_df = pd.read_csv('frequency_1M_stride01s_sample2400_2/m.txt', header=None, names=['m'])
        n_df = pd.read_csv('frequency_1M_stride01s_sample2400_2/n.txt', header=None, names=['n'])
        l_df.columns = ['l']

        print("读取l m n完毕")

        imagerecon(uvw_file, viss_file, l_df, m_df, n_df, constant2, period)


if __name__ == "__main__":
    main()

