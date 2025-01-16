import torch
import pandas as pd
import time
from multiprocessing import Pool
from datetime import datetime, timedelta

def imagerecon(uvw_df, viss_df, l_part, m_part, n_part, period, gpu_id, save_id):
    RES = 20940
    dl = 2*RES/(RES-1)
    dm = 2*RES/(RES-1)
    dn = 2*RES/(RES-1)

    assert len(uvw_df) == len(viss_df)
    # 记录生成开始时间
    start_time = time.time()

    constant1 = -2 * torch.pi * torch.tensor(1j, device=f"cuda:{gpu_id}")
    constant2 = 2 * torch.pi * torch.tensor(1j, device=f"cuda:{gpu_id}")
    
    # 将数据转换为PyTorch张量并移动到GPU上
    viss_tensor = torch.tensor(viss_df['viss'].values, device=f'cuda:{gpu_id}')
    u_tensor = torch.tensor(uvw_df['u'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    v_tensor = torch.tensor(uvw_df['v'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    w_tensor = torch.tensor(uvw_df['w'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    uvwFreqMap_tensor = torch.tensor(uvw_df['freq'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)

    l_tensor = torch.tensor(l_part['l'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    m_tensor = torch.tensor(m_part['m'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    n_tensor = torch.tensor(n_part['n'].values, device=f'cuda:{gpu_id}', dtype=torch.float32)
    print("u_tensor: ", u_tensor.shape)
    print("v_tensor: ", v_tensor.shape)
    print("w_tensor: ", w_tensor.shape)
    print("viss_tensor: ", viss_tensor.shape)
    print("uvwFreqMap_tensor: ", uvwFreqMap_tensor.shape)
    print("l_tensor: ", l_tensor.shape)
    print("m_tensor: ", m_tensor.shape)
    print("n_tensor: ", n_tensor.shape)

    del l_part
    del m_part
    del n_part
    del uvw_df
    del viss_df

    l_tensor = l_tensor / dl
    m_tensor = m_tensor / dm
    n_tensor = n_tensor / dn
    print("l m n 预处理完成")

    # viss 去除相位
    viss_tensor = viss_tensor * torch.exp(constant1 * w_tensor / dn)
    print("viss 去除相位完成")

    # 存储 imagerecon的部分
    image_tensor = torch.zeros(len(l_tensor), dtype=torch.float32, device=f'cuda:{gpu_id}')

    # 遍历 uvw_df 中的每一行
    for index in range(l_tensor.shape[0]):
        # row_start_time = time.time()
        if index % 5000000 == 0:
            print(f"GPU: {gpu_id} - 已处理 {index} 行")

        # # 计算复数指数部分
        temp_y = constant2 * (u_tensor*l_tensor[index] + v_tensor*m_tensor[index] + w_tensor*n_tensor[index]) 
        
        # # 计算结果
        image = torch.sum(uvwFreqMap_tensor * viss_tensor * torch.exp(temp_y)) / len(u_tensor)

        image_tensor[index] = image.real
   
        # row_end_time = time.time()
        # row_elapsed_time = row_end_time - row_start_time
        # print("Generated row {} in {:.4f} seconds".format(index, row_elapsed_time))
    
    # 记录生成结束时间
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print("Generated all rows in {:.4f} seconds".format(total_elapsed_time))
    
    # 将结果转换为DataFrame并返回
    image_df = pd.DataFrame(image_tensor.cpu().numpy(), columns=['imagerecon_real'])

    # 将结果保存到 CSV 文件
    image_df.to_csv(f"torch_10M/image{period}_{save_id}.txt", sep='\t', header=False, index=False)

    print(f"结果已保存到 torch_10M/image{period}_{save_id}.txt 文件中。")


def process_period(period, block):

    print(f"开始处理第{period}个周期")
    print(f"开始处理第{block}个块")
    uvw_file = f"frequency_10M/updated_uvw{period}frequency10M.txt"
    viss_file = f"torch_10M/viss{period}.csv"

    viss_df = pd.read_csv(viss_file)
    viss_df['viss'] = viss_df['viss_real'] + viss_df['viss_imag']*1j
    print("读取 viss 完毕")

    l_df = pd.read_csv('lmn10M/l10M_dup.txt', header=None, names=['l'])
    m_df = pd.read_csv('lmn10M/m10M_dup.txt', header=None, names=['m'])
    n_df = pd.read_csv('lmn10M/n10M_dup.txt', header=None, names=['n'])

    print("读取l m n完毕", l_df.shape)

    # uvw_df分块
    num_gpus = torch.cuda.device_count()
    print("num_gpus: ", num_gpus)

    uvw_df = pd.read_csv(uvw_file, delimiter=' ', header=None, names=['u', 'v', 'w', 'freq'])
    print("读取 uvw 完毕", uvw_df.shape)

    part_size = 14700000
    num_parts = len(l_df) // part_size + 1
    print("num_parts: ", num_parts)

    l_parts = [l_df[i * part_size:(i + 1) * part_size] for i in range(num_parts)]  
    m_parts = [m_df[i * part_size:(i + 1) * part_size] for i in range(num_parts)]  
    n_parts = [n_df[i * part_size:(i + 1) * part_size] for i in range(num_parts)]  
    l_parts[-1] = l_df[(num_parts - 1) * part_size:]
    m_parts[-1] = m_df[(num_parts - 1) * part_size:]
    n_parts[-1] = n_df[(num_parts - 1) * part_size:]

    print('0: ', l_parts[0].shape)
    print('-1: ', l_parts[-1].shape)
    del l_df, m_df, n_df

    batches = list(zip(l_parts, m_parts, n_parts))
    batch_size = num_gpus
    gpu_chunks = [batches[i : i+batch_size] for i in range(0, len(batches), batch_size)]

    for gpu_id, batch in enumerate(gpu_chunks):
        print("gpu_id:", gpu_id)
        for i, (_, _, _) in enumerate(batch):
            print("i: ", i, i+gpu_id*num_gpus)
    

    with Pool(processes=num_gpus) as pool:
        for gpu_id, batch in enumerate(gpu_chunks):
            if gpu_id == block:
                # 为每个GPU分配一个任务
                args = [(uvw_df, viss_df, l, m, n, period, i, i+gpu_id*num_gpus) for i, (l, m, n) in enumerate(batch)]
                pool.starmap(imagerecon, args)
   
    
    # del l_parts, m_parts, n_parts
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import time
    process_period(6, 8)
    print("第1个周期处理完成，开始处理第2个周期")