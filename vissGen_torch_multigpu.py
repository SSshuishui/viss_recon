import torch
import pandas as pd
import time
from multiprocessing import Pool


def visscal(uvw_part, l_df, m_df, n_df, C_df, constant1, period, gpu_id):
    RES = 20940
    dl = 2*RES/(RES-1)
    dm = 2*RES/(RES-1)
    dn = 2*RES/(RES-1)

    # 记录生成开始时间
    start_time = time.time()
    
    # 将数据转换为PyTorch张量并移动到GPU上
    uvw_tensor = torch.tensor(uvw_part[['u', 'v', 'w']].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    l_tensor = torch.tensor(l_df['l'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    m_tensor = torch.tensor(m_df['m'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    n_tensor = torch.tensor(n_df['n'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    C_tensor = torch.tensor(C_df['C'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    print("uvw_tensor: ", uvw_tensor.shape)
    print("l_tensor: ", l_tensor.shape)
    print("m_tensor: ", m_tensor.shape)
    print("n_tensor: ", n_tensor.shape)
    print("C_tensor: ", C_tensor.shape)

    l_tensor = l_tensor / dl
    m_tensor = m_tensor / dm
    n_tensor = (n_tensor-1) / dn

    # 初始化结果张量
    viss_tensor = torch.zeros((len(uvw_tensor), 2), dtype=torch.float32, device=f'cuda:{gpu_id}')

    # 遍历 uvw_df 中的每一行
    for index, row in enumerate(uvw_tensor):
        # row_start_time = time.time()
        
        # 计算复数指数部分
        temp_y = constant1 * (row[0]*l_tensor + row[1]*m_tensor + row[2]*n_tensor)
        
        # 计算结果 
        viss = torch.sum(torch.exp(temp_y) * C_tensor)
        viss_tensor[index, 0] = viss.real
        viss_tensor[index, 1] = viss.imag
   
        # row_end_time = time.time()
        # row_elapsed_time = row_end_time - row_start_time
        # print("Generated row {} in {:.4f} seconds".format(index, row_elapsed_time))
    
    # 记录生成结束时间
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print("GPU {} Generated all rows in {:.4f} seconds".format(gpu_id, total_elapsed_time))
    
    # 将结果转换为DataFrame并返回
    viss_df = pd.DataFrame(viss_tensor.cpu().numpy(), columns=['viss_real', 'viss_imag'])

    # 将结果保存到 CSV 文件
    viss_df.to_csv(f"viss{period}_{gpu_id}.csv", index=False)

    print(f"结果已保存到 viss{period}_{gpu_id}.csv 文件中。")


def process_period(period):
    constant1 = -2 * torch.pi * 1j

    print(f"开始处理第{period}个周期")

    uvw_file = f"uvwMap/uvw{period}frequency10M_new.txt"
    l_df = pd.read_csv('lmn10M/l.txt', header=None, names=['l'])
    m_df = pd.read_csv('lmn10M/m.txt', header=None, names=['m'])
    n_df = pd.read_csv('lmn10M/n.txt', header=None, names=['n'])
    C_df = pd.read_csv('lmn10M/C.txt', header=None, names=['C'])

    print("读取l m n C完毕")

    # uvw_df分块
    num_gpus = torch.cuda.device_count()

    uvw_df = pd.read_csv(uvw_file, delimiter=' ', header=None, names=['u', 'v', 'w', 'freq'])
    part_size = len(uvw_df) // num_gpus
    uvw_parts = [uvw_df[i * part_size:(i + 1) * part_size] for i in range(num_gpus)]  
    uvw_parts[-1] = uvw_df[(num_gpus - 1) * part_size:]

    # 调用函数并分发给每个GPU处理
    with Pool(processes=num_gpus) as pool:
        pool.starmap(visscal, [(uvw_part, l_df, m_df, n_df, C_df, constant1, period, gpu_id) for gpu_id, uvw_part in enumerate(uvw_parts)])


    # 创建一个空的 DataFrame 用于存储合并后的数据
    combined_df = pd.DataFrame(columns=['viss_real', 'viss_imag'])
    # 加载并合并每个 CSV 文件
    for gpu_id in range(num_gpus):
        filename = f"viss{period}_{gpu_id}.csv"
        df = pd.read_csv(filename)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # 将合并后的 DataFrame 保存为一个新的 CSV 文件
    combined_df.to_csv(f"viss{period}.csv", index=False)


if __name__ == "__main__":
    # 处理第一个周期
    process_period(1)
