import os
import concurrent.futures


def split_line(line):
    return [float(value) for value in line.split()]


def process_file(file_path):
    local_min = [float('inf')] * 3
    local_max = [float('-inf')] * 3
    with open(file_path, 'r') as file:
        for line in file:
            values = split_line(line)
            for i in range(len(values)):
                local_min[i] = min(local_min[i], values[i])
                local_max[i] = max(local_max[i], values[i])
    return local_min, local_max


def is_target_file(filename):
    if filename.startswith('uvw') and filename.endswith('frequency30M.txt'):
        number_part = filename[3:-15]
        if number_part.isdigit():
            number = int(number_part)
            return 1 <= number <= 250
    return False


def update_global_min_max(global_min, global_max, local_min, local_max):
    for i in range(len(global_min)):
        global_min[i] = min(global_min[i], local_min[i])
        global_max[i] = max(global_max[i], local_max[i])

folder_path = './frequency_30M/'  # 替换为你的文件夹路径
filenames = ['uvw1frequency30M.txt', 'uvw2frequency30M.txt','uvw3frequency30M.txt','uvw4frequency30M.txt','uvw5frequency30M.txt','uvw6frequency30M.txt',
             'uvw100frequency30M.txt', 'uvw101frequency30M.txt', 'uvw102frequency30M.txt', 'uvw103frequency30M.txt','uvw104frequency30M.txt','uvw106frequency30M.txt',
             'uvw12frequency30M.txt','uvw21frequency30M.txt','uvw20frequency30M.txt','uvw23frequency30M.txt','uvw51frequency30M.txt','uvw33frequency30M.txt']
global_min = [float('inf')] * 3
global_max = [float('-inf')] * 3


# 创建一个线程池来处理文件
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_file, os.path.join(folder_path, filename)): filename for filename in filenames}

    # 等待所有线程完成，并更新全局最大最小值
    for future in concurrent.futures.as_completed(futures):
        local_min, local_max = future.result()
        global_min = [min(global_min[i], local_min[i]) for i in range(3)]
        global_max = [max(global_max[i], local_max[i]) for i in range(3)]



# 打印结果
print("Global Min:", global_min)
print("Global Max:", global_max)
