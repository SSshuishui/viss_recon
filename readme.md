
# 准备文件

updated_uvw{***}frequency10M.txt  到 ```frequency10M``` 目录

l m n C .txt 到 ```lmn10M``` 目录

# 编译
### 1. 先编译
```
nvcc -o vissgen10M vissGen10M_float.cu -Xcompiler -fopenmp
```

### 2. 再运行
```
./vissgen10M
```


# 得到的结果再进行对齐
修改里面的period周期
```
python recon_combined_image.py
```