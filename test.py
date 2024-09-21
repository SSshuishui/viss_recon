# 定义每个数组的元素总数
OrbitRes = 2592000  # 假设的轨道分辨率
OrbitCounts = 140  # 假设的轨道计数
satnum = 8  # 假设的卫星数量

# 计算单个数组的元素总数
total_elements = OrbitRes * OrbitCounts * satnum

# 计算单个数组的内存占用（每个float占用4字节）
memory_per_element = 4  # float类型的大小（字节）

# 计算单个数组的内存占用
memory_per_array_bytes = total_elements * memory_per_element

# 将字节转换为MB和GB
memory_per_array_mb = memory_per_array_bytes / (1024 * 1024)  # 1MB = 1024 * 1024字节
memory_per_array_gb = memory_per_array_bytes / (1024 ** 3)  # 1GB = 1024 * 1024 * 1024字节

# 计算三个数组的总内存占用（字节、MB、GB）
total_memory_bytes = 3 * memory_per_array_bytes
total_memory_mb = 3 * memory_per_array_mb
total_memory_gb = 3 * memory_per_array_gb

print(f"每个数组的内存占用: {memory_per_array_bytes} 字节, {memory_per_array_mb:.2f} MB, {memory_per_array_gb:.2f} GB")
print(f"三个数组的总内存占用: {total_memory_bytes} 字节, {total_memory_mb:.2f} MB, {total_memory_gb:.2f} GB")
