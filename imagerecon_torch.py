import torch
import pandas as pd
import time

def imagerecon(uvw_file, viss_file, l_df, m_df, n_df, nt_df, constant1, constant2, period):
    # 读取viss文件
    viss_df = pd.read_csv(viss_file)
    viss_df['viss'] = viss_df['viss_real'] + viss_df['viss_imag']*1j
    print("读取 viss 完毕")

    # 读取uvw文件
    uvw_df = pd.read_csv(uvw_file, delimiter=' ', header=None, names=['u', 'v', 'w', 'freq'])
    print("读取 uvw 完毕")

    RES = 20940
    dl = 2*RES/(RES-1)
    dm = 2*RES/(RES-1)
    dn = 2*RES/(RES-1)

    # 记录生成开始时间
    start_time = time.time()
    
    # 将数据转换为PyTorch张量并移动到GPU上
    viss_tensor = torch.tensor(viss_df['viss'].values, device='cuda')
    uvw_tensor = torch.tensor(uvw_df[['u', 'v', 'w']].values, device='cuda', dtype=torch.float32)
    uvwFreqMap_tensor = torch.tensor(uvw_df['freq'].values, device='cuda', dtype=torch.float32)

    l_tensor = torch.tensor(l_df['l'].values, device='cuda', dtype=torch.float32)
    m_tensor = torch.tensor(m_df['m'].values, device='cuda', dtype=torch.float32)
    n_tensor = torch.tensor(n_df['n'].values, device='cuda', dtype=torch.float32)
    nt_tensor = torch.tensor(nt_df['nt'].values, device='cuda', dtype=torch.float32)
    print("uvw_tensor: ", uvw_tensor.shape)
    print("viss_tensor: ", viss_tensor.shape)
    print("uvwFreqMap_tensor: ", uvwFreqMap_tensor.shape)
    print("l_tensor: ", l_tensor.shape)
    print("m_tensor: ", m_tensor.shape)
    print("n_tensor: ", n_tensor.shape)
    print("nt_tensor: ", nt_tensor.shape)

    # 采样 l m n
    target_rows = 2094 * 2094
    num_elements = l_tensor.numel()
    interval = num_elements // target_rows
    # 生成均匀间隔的索引列表
    indices = torch.arange(0, num_elements, interval)[:target_rows]
    l_tensor = l_tensor[indices] 
    m_tensor = m_tensor[indices]  
    n_tensor = n_tensor[indices]  
    nt_tensor = nt_tensor[indices]  
    print("l_tensor shape: ", l_tensor.shape)
    print("m_tensor shape: ", m_tensor.shape)
    print("n_tensor shape: ", n_tensor.shape)
    print("nt_tensor shape: ", nt_tensor.shape)

    # viss 去除相位
    viss_tensor = viss_tensor * torch.exp(constant1 * uvw_tensor[:, 2] / dn)

    l_tensor = l_tensor / dl
    m_tensor = m_tensor / dm
    n_tensor = n_tensor / dn
    
    print("viss 去除相位")

    # 存储 imagerecon的部分
    image_tensor = torch.zeros(len(l_tensor), dtype=torch.float32, device='cuda')

    # 遍历 uvw_df 中的每一行
    for index, _ in enumerate(l_tensor):
        row_start_time = time.time()

        # 计算复数指数部分
        temp_y = constant2 * (uvw_tensor[:, 0]*l_tensor[index] + uvw_tensor[:, 1]*m_tensor[index] + uvw_tensor[:, 2]*n_tensor[index]) 
        
        # 计算结果
        image = torch.sum(uvwFreqMap_tensor * viss_tensor * torch.exp(temp_y)) * torch.abs(nt_tensor[index]) / len(uvw_tensor)

        image_tensor[index] = image.real
   
        row_end_time = time.time()
        row_elapsed_time = row_end_time - row_start_time
        print("Generated row {} in {:.4f} seconds".format(index, row_elapsed_time))
    
    # 记录生成结束时间
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print("Generated all rows in {:.4f} seconds".format(total_elapsed_time))
    
    # 将结果转换为DataFrame并返回
    image_df = pd.DataFrame(image_tensor.cpu().numpy(), columns=['imagerecon_real'])

    # 将结果保存到 CSV 文件
    image_df.to_csv(f"image{period}.txt", sep='\t', header=False, index=False)

    print(f"结果已保存到 image{period}.txt 文件中。")


def main():
    # matlab中pi是3.1416
    constant1 = -2 * torch.pi * 1j
    constant2 = 2 * torch.pi * 1j
    
    for period in range(1, 131): 
        print(f"开始处理第{period}个周期")
        uvw_file = f"uvwMap/uvw{period}frequency10M_new.txt"
        viss_file = f"viss{period}.csv"

        l_df = pd.read_csv('lmn10M/l.txt', header=None, names=['l'])
        m_df = pd.read_csv('lmn10M/m.txt', header=None, names=['m'])
        n_df = pd.read_csv('lmn10M/n.txt', header=None, names=['n'])
        nt_df = pd.read_csv('lmn10M/nt.txt', header=None, names=['nt'])

        print("读取l m n nt完毕")

        imagerecon(uvw_file, viss_file, l_df, m_df, n_df, nt_df, constant1, constant2, period)


if __name__ == "__main__":
    main()

