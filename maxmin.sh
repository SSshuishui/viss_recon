#!/bin/bash

# 初始化最大值和最小值变量
# 假设初始值足够大或足够小
global_max1=-999999
global_max2=-999999
global_max3=-999999
global_min1=999999
global_min2=999999
global_min3=999999

# 循环遍历文件
for i in $(seq 1 250); do
    filename="uvw${i}frequency10M.txt"
    
    # 读取第一行和最后一行
    first_line=$(head -n 1 "$filename")
    last_line=$(tail -n 1 "$filename")
    
    # 提取值并比较
    for line in "$first_line" "$last_line"; do
        read -r col1 col2 col3 <<< "$line"
        
        # 更新最大值
        [[ "$col1" -gt "$global_max1" ]] && global_max1="$col1"
        [[ "$col2" -gt "$global_max2" ]] && global_max2="$col2"
        [[ "$col3" -gt "$global_max3" ]] && global_max3="$col3"
        
        # 更新最小值
        [[ "$col1" -lt "$global_min1" ]] && global_min1="$col1"
        [[ "$col2" -lt "$global_min2" ]] && global_min2="$col2"
        [[ "$col3" -lt "$global_min3" ]] && global_min3="$col3"
    done
done

# 打印结果
echo "Global Maximums: $global_max1, $global_max2, $global_max3"
echo "Global Minimums: $global_min1, $global_min2, $global_min3"
