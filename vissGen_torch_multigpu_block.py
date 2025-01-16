import torch
import pandas as pd
import time
from multiprocessing import Pool

def visscal(uvw_part, l_df, m_df, n_df, C_df, period, gpu_id, save_id):
    RES = 20940
    dl = 2*RES/(RES-1)
    dm = 2*RES/(RES-1)
    dn = 2*RES/(RES-1)

    constant1 = -2 * torch.pi * torch.tensor(1j, device=f"cuda:{gpu_id}")

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
    viss_df.to_csv(f"torch_10M/viss{period}_{save_id}.csv", index=False)

    print(f"结果已保存到 torch_10M/viss{period}_{save_id}.csv 文件中。")


def process_period(period, block, part_size, num_gpus):

    print(f"开始处理第{period}个周期")
    print(f"开始处理第{block}个块")

    uvw_file = f"frequency_10M/updated_uvw{period}frequency10M.txt"
    l_df = pd.read_csv('lmn10M/l10M_dup.txt', header=None, names=['l'])
    m_df = pd.read_csv('lmn10M/m10M_dup.txt', header=None, names=['m'])
    n_df = pd.read_csv('lmn10M/n10M_dup.txt', header=None, names=['n'])
    C_df = pd.read_csv('lmn10M/C10M.txt', header=None, names=['C'])

    print("读取l m n C完毕")
    print(l_df.shape[0], m_df.shape[0], n_df.shape[0], C_df.shape[0])

    # uvw_df分块
    uvw_df = pd.read_csv(uvw_file, delimiter=' ', header=None, names=['u', 'v', 'w', 'freq'])
    # 删除'freq'列
    uvw_df = uvw_df.drop('freq', axis=1)

    num_parts = len(uvw_df) // part_size + 1
    print("num_parts: ", num_parts)

    uvw_parts = [uvw_df[i * part_size:(i + 1) * part_size] for i in range(num_parts)]
    uvw_parts[-1] = uvw_df[(num_parts - 1) * part_size:]

    print(len(uvw_parts))
    print('0: ', uvw_parts[0].shape)
    print('-1: ', uvw_parts[-1].shape)


    for i in range(0, num_parts//num_gpus+1):
        for gpu_id,_ in enumerate(uvw_parts[i*num_gpus:(i+1)*num_gpus]):
            print("i: ", i, gpu_id, gpu_id+i*num_gpus)

    with Pool(processes=num_gpus) as pool:
        for i in range(0, num_parts//num_gpus+1):
            if i == block:
                print("处理i: ", i)
                # 为每个GPU分配一个任务
                args = [(uvw_part, l_df, m_df, n_df, C_df, period, gpu_id, gpu_id+i*num_gpus) for gpu_id,uvw_part in enumerate(uvw_parts[i*num_gpus:(i+1)*num_gpus])]
                pool.starmap(visscal, args)
                
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 处理一个周期
    # 总行数是不超过 14323000 n个GPU，每个GPU处理part_size行
    # 需要的block数就是 14323000 // (n * part_size) + 1， 最好保持14350000 / (n * part_size)差不多是整数（1.98这种），而不是比如2.07这种
    # 比如设置 part_size = 896875, 用8张卡， 那么就需要 block = 1.996， 不超过2， 也就是两天晚上跑完
    # 运行的时候就是 两天晚上跑， 第一天跑 process_period(period, 0, part_size, num_gpus), 第二天跑 process_period(period, 1, part_size, num_gpus)


    # 跑的时候可以先测试把38 和 48-51行注释打开，看一下一行是多少秒能算完，然后估计一下一晚上能跑多少行，再设置part_size

    part_size = 726050
    num_gpus = torch.cuda.device_count()
    print("num_gpus: ", num_gpus)

    lines = 14323000
    print("总行数: ", lines)
    print("总块数目：" , lines // (num_gpus * part_size))  

    process_period(7, 0, part_size, num_gpus)


