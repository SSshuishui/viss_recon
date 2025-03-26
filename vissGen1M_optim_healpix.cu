#include <cstdio>
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "error.cuh"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

#define _USE_MATH_DEFINES
#define EXP 0.0000000000

using namespace std;
using Complex = thrust::complex<float>;

// complexExp 函数的实现
__device__ thrust::complex<float> complexExp(const Complex &d) {
    float realPart = exp(d.real()) * cos(d.imag());
    float imagPart = exp(d.real()) * sin(d.imag());
    return thrust::complex<float>(realPart, imagPart);
}
// complexAbs 函数的实现
__device__ thrust::complex<float> ComplexAbs(const Complex &d) {
    // 复数的模定义为 sqrt(real^2 + imag^2)
    return thrust::complex<float>(sqrt(d.real() * d.real() + d.imag() * d.imag()));
}

struct timeval start, finish;
float total_time;

string address = "./frequency_1M/";
string F_address = "./F_recon_1M/";
string para;
string duration = "frequency1M";  // 第几个周期的uvw
string sufix = ".txt";

// 1 M
const int uvw_presize = 4000000;

// 定义常量
#define BLOCK_SIZE 128                     // 线程块大小
#define SHARED_MEM_SIZE BLOCK_SIZE         // 共享内存大小
#define MAX_THREADS_PER_BLOCK 1024        // GPU每个块的最大线程数



// 定义计算可见度核函数, 验证一致
__global__ void visscal(
    int uvw_index, int lmnC_index,
    Complex* __restrict__ viss,    
    const float* __restrict__ u,   
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ C,
    const Complex I1,             
    const Complex CPI,
    const Complex zero,
    const Complex two)
{
    // 声明共享内存
    __shared__ float s_l[SHARED_MEM_SIZE];
    __shared__ float s_m[SHARED_MEM_SIZE];
    __shared__ float s_n[SHARED_MEM_SIZE];
    __shared__ float s_C[SHARED_MEM_SIZE];

    // 获取线程索引
    const int uvw_ = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (uvw_ >= uvw_index) return;

    // 预先加载频繁使用的数据到寄存器
    const float u_val = u[uvw_];
    const float v_val = v[uvw_];
    const float w_val = w[uvw_];

    // 初始化累加器
    Complex acc = zero;
    // 分块处理 lmnC 数据
    for (int base = 0; base < lmnC_index; base += SHARED_MEM_SIZE) {
        const int current_chunk_size = min(SHARED_MEM_SIZE, lmnC_index - base);
        
        // 协作加载数据到共享内存
        for (int i = tid; i < current_chunk_size; i += blockDim.x) {
            const int global_idx = base + i;
            s_l[i] = l[global_idx];
            s_m[i] = m[global_idx];
            s_n[i] = n[global_idx];
            s_C[i] = C[global_idx];
        }

        // 确保所有线程完成数据加载
        __syncthreads();

        // 处理这个分块中的数据
        #pragma unroll 8  // 添加循环展开指令
        for (int i = 0; i < current_chunk_size; ++i) {
            // 计算相位
            const float phase = u_val * s_l[i] + v_val * s_m[i] + w_val * (s_n[i] - 1.0f);

            // 计算复指数
            const Complex exp_val = complexExp((zero - I1) * two * CPI * Complex(phase, 0.0f));
            
            // 累加结果
            acc += Complex(s_C[i], 0.0f) * exp_val;
        }

        // 确保所有线程完成计算后再加载下一块数据
        __syncthreads();
    }
    // 计算最终的复指数因子
    const Complex final_exp = complexExp((zero - I1) * two * CPI * Complex(w_val, 0.0f));
    // 存储最终结果
    viss[uvw_] = acc * final_exp;
}

// 启动核函数的包装函数
void launch_visscal(
    const int uvw_index,
    const int lmnC_index,
    Complex* d_viss,
    float* d_u,
    float* d_v,
    float* d_w,
    float* d_l,
    float* d_m,
    float* d_n,
    float* d_C,
    const Complex I1,
    const Complex CPI,
    const Complex zero,
    const Complex two)
{
    // 计算网格和块的大小
    const int threadsPerBlock = BLOCK_SIZE;
    const int blocksPerGrid = (uvw_index + threadsPerBlock - 1) / threadsPerBlock;

    // 计算共享内存大小
    size_t sharedMemSize = SHARED_MEM_SIZE * 4 * sizeof(float);  // 4个float数组
    // 设置缓存配置以优化共享内存的使用
    cudaFuncSetCacheConfig(visscal, cudaFuncCachePreferShared);

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    visscal<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(
        uvw_index, lmnC_index,
        d_viss,
        d_u, 
        d_v, 
        d_w,
        d_l, 
        d_m, 
        d_n, 
        d_C,
        I1, CPI, zero, two
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


// 定义图像反演核函数  验证正确
__global__ void imagerecon(
    const int uvw_index,
    const int lmnC_index,
    Complex* __restrict__ F,                    
    const Complex* __restrict__ viss,           
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ C,
    const float* __restrict__ uvwFrequencyMap,
    const Complex I1,                    
    const Complex CPI,
    const Complex zero,
    const Complex two)
{
    // 声明共享内存
    __shared__ float s_u[SHARED_MEM_SIZE];
    __shared__ float s_v[SHARED_MEM_SIZE];
    __shared__ float s_w[SHARED_MEM_SIZE];
    __shared__ float s_uvwFreq[SHARED_MEM_SIZE];
    __shared__ Complex s_viss[SHARED_MEM_SIZE];

    const int lmnC_ = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (lmnC_ >= lmnC_index) return;

    // 预计算常量
    const Complex amount(uvw_index, 0.0f);      // 转换为常量
    const float l_val = l[lmnC_];
    const float m_val = m[lmnC_];
    const float n_val = n[lmnC_];

    // 使用复数累加器
    Complex acc = zero;
    // 使用共享内存分块处理数据
    for (int base = 0; base < uvw_index; base += SHARED_MEM_SIZE) {
        const int current_chunk_size = min(SHARED_MEM_SIZE, lmnC_index - base);
        // 改进的协作加载方式
        for (int i = tid; i < current_chunk_size; i += blockDim.x) {
            const int global_idx = base + i;
            s_u[i] = u[global_idx];
            s_v[i] = v[global_idx];
            s_w[i] = w[global_idx];
            s_uvwFreq[i] = uvwFrequencyMap[global_idx];
            s_viss[i] = viss[global_idx];
        }
        // 确保所有线程完成数据加载
        __syncthreads();
        // 处理当前块中的数据
        #pragma unroll 8
        for (int i = 0; i < current_chunk_size; ++i) {
            // 计算相位
            const float phase = s_u[i] * l_val + s_v[i] * m_val + s_w[i] * n_val;

            // 计算复指数
            const Complex exp_val = complexExp(I1 * two * CPI * Complex(phase, 0.0f));

            // 累加结果
            acc += s_uvwFreq[i] * s_viss[i] * exp_val;
        }

        // 确保所有线程完成计算后再加载下一块数据
        __syncthreads();
    }
    // 归一化并存储结果
    F[lmnC_] = acc / amount;
}


// 启动核函数的包装函数
void launch_imagerecon(
    const int uvw_index,
    const int lmnC_index,
    Complex* d_F,
    Complex* d_viss,
    float* d_u,
    float* d_v,
    float* d_w,
    float* d_l,
    float* d_m,
    float* d_n,
    float* d_C,
    float* d_uvwFrequencyMap,
    const Complex I1,
    const Complex CPI,
    const Complex zero,
    const Complex two)
{
    // 计算网格和块的大小
    const int threadsPerBlock = BLOCK_SIZE;
    const int blocksPerGrid = (lmnC_index + threadsPerBlock - 1) / threadsPerBlock;

    // 检查共享内存大小
    size_t sharedMemSize = SHARED_MEM_SIZE * (5 * sizeof(float) + sizeof(Complex));
    
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 设置缓存配置以优化共享内存的使用
    cudaFuncSetCacheConfig(imagerecon, cudaFuncCachePreferShared);

    // 启动核函数
    imagerecon<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(
        uvw_index, lmnC_index,
        d_F, d_viss,
        d_u, d_v, d_w,
        d_l, d_m, d_n,
        d_C, d_uvwFrequencyMap,
        I1, CPI, zero, two
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


int vissGen(int id, int RES, int start_period) 
{   
    cout << "res: " << RES << endl;
    int days = 50;  // 一共有多少个周期  15月 * 30天 / 14天/周期
    cout << "periods: " << days << endl;
    Complex I1(0.0, 1.0);
    Complex zero(0.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);

    gettimeofday(&start, NULL);
    int nDevices=1;
    // 设置节点数量（gpu显卡数量）
    // CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    // 加载存储 l m n C nt的文件（对于不同的frequency不一样，只与frequency有关）
    string para, address_l, address_m, address_n, address_C;
    ifstream lFile, mFile, nFile, CFile;
    para = "l_nest_1m";
    address_l = address + para + sufix;
    lFile.open(address_l);
    cout << "address_l: " << address_l << endl;
    para = "m_nest_1m";
    address_m = address + para + sufix;
    mFile.open(address_m);
    cout << "address_m: " << address_m << endl;
    para = "n_nest_1m";
    address_n = address + para + sufix;
    nFile.open(address_n);
    cout << "address_n: " << address_n << endl;
    para = "C_nest_1m";
    address_C = address + para + sufix;
    CFile.open(address_C);
    cout << "address_C: " << address_C << endl;
    if (!lFile.is_open() || !mFile.is_open() || !nFile.is_open() ||!CFile.is_open()) {
        std::cerr << "无法打开一个或多个文件：" << std::endl;
        if (!lFile.is_open()) std::cerr << "无法打开文件: " << address_l << std::endl;
        if (!mFile.is_open()) std::cerr << "无法打开文件: " << address_m << std::endl;
        if (!nFile.is_open()) std::cerr << "无法打开文件: " << address_n << std::endl;
        if (!CFile.is_open()) std::cerr << "无法打开文件: " << address_C << std::endl;
        return -1; 
    }
    int lmnC_index = 512 * 512 * 12;
    cout << "lmnC index: " << lmnC_index << endl;

    std::vector<float> cl(lmnC_index), cm(lmnC_index), cn(lmnC_index), cC(lmnC_index);
    for (int i = 0; i < lmnC_index && lFile.good() && mFile.good() && nFile.good() && CFile.good(); ++i) {
        lFile >> cl[i];
        mFile >> cm[i];
        nFile >> cn[i];
        CFile >> cC[i];
    }
    lFile.close();
    mFile.close();
    nFile.close();
    CFile.close();

    // 导入uvw坐标的出现频次，txt文件的每一行每个坐标的频次
    auto uvwMapStart = std::chrono::high_resolution_clock::now();
    // 创建map存储
    std::unordered_map<std::string, float> cUVWFrequencyMap;
    string uvwmap_address = address + "uvwMap130.txt";
    std::ifstream uvwMapFile(uvwmap_address);
    if (uvwMapFile.is_open()) {
        // 读取第一行获取总行数
        string firstLine;
        std::getline(uvwMapFile, firstLine);
        int totalLines = std::stoi(firstLine);
        cout << "uvwMap totalLines: " << totalLines << endl;
        // 预分配内存
        cUVWFrequencyMap.reserve(totalLines);
        // 每一行的格式： -23 -288 -166 4
        string line;
        while (std::getline(uvwMapFile, line)) {
            std::istringstream iss(line);
            int u_point, v_point, w_point;
            int uvw_frequency;
            if (iss >> u_point >> v_point >> w_point >> uvw_frequency) {
                std::string key = std::to_string(u_point) + "_" + std::to_string(v_point) + "_" + std::to_string(w_point);
                cUVWFrequencyMap[key] = uvw_frequency;
            } else {
                cout << "Failed to parse line: " << line << endl; // 解析失败时的调试信息
            }
        }
        uvwMapFile.close();
    }
    // 打印测试确保是正确的
    int count = 0;
    int numElementsToPrint = 6; // 设定要打印的元素数量
    for (const auto& pair : cUVWFrequencyMap) {
        std::cout << pair.first << ": " << pair.second << std::endl;
        if (++count == numElementsToPrint) {
            break;
        }
    }
    cout << "Transfer uvw Frequency Success! " << endl;
    auto uvwMapFinish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> uvwMapElapsed = uvwMapFinish - uvwMapStart;
    cout << "Transfer uvw Frequency Elapsed Time: " << uvwMapElapsed.count() << " s\n";


    // 开启cpu线程并行
    // 一个线程处理1个GPU
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();  // 从 0 开始编号的并行执行线程
        cudaSetDevice(tid);
        std::cout << "Thread " << tid << " is running on device " << tid << std::endl;

        // 遍历所有开启的线程处理， 一个线程控制一个GPU 处理一个id*amount/total的块
        for (int p = tid+start_period; p < days; p += nDevices) 
        {
            cout << "for loop: " << p+1 << endl;

            // 将 l m n C NX 数据从cpu搬到GPU上        
            thrust::device_vector<float> l(cl.begin(), cl.end());
            thrust::device_vector<float> m(cm.begin(), cm.end());
            thrust::device_vector<float> n(cn.begin(), cn.end());
            thrust::device_vector<float> C(cC.begin(), cC.end());

            // 创建用来存储不同index中【u, v, w】
            std::vector<float> cu(uvw_presize), cv(uvw_presize), cw(uvw_presize);
            thrust::device_vector<float> u(uvw_presize), v(uvw_presize), w(uvw_presize);

            // 创建存储uvw坐标对应的频次
            std::vector<float> uvwMapVector(uvw_presize);
            thrust::device_vector<float> uvwFrequencyMap(uvw_presize);
        
            // 存储计算后的到的最终结果
            thrust::device_vector<Complex> F(lmnC_index);

            // 计时统计
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            // 记录开始事件
            cudaEventRecord(start);

            // 记录uvw开始事件
            cudaEvent_t uvwstart, uvwstop;
            cudaEventCreate(&uvwstart);
            cudaEventCreate(&uvwstop);
            cudaEventRecord(uvwstart);

            // 创建一个临界区，保证只有一个线程进入，用于构建u v w
            int uvw_index;
            #pragma omp critical
            {
                string address_uvw = address + "uvw" + to_string(p+1) + duration + sufix;
                cout << "address_uvw: " << address_uvw << std::endl;
                
                ifstream uvwFile(address_uvw);
                // 同时用一个向量保存每一个uvw坐标点的frequency
                uvw_index = 0;
                float u_point, v_point, w_point;
                string key_point;
                if (uvwFile.is_open()) {
                    while (uvwFile >> u_point >> v_point >> w_point) {
                        // 直接构造 key_point
                        key_point = std::to_string(static_cast<int>(u_point)) + "_" + 
                                    std::to_string(static_cast<int>(v_point)) + "_" + 
                                    std::to_string(static_cast<int>(w_point));

                        // 简化查找操作
                        auto it = cUVWFrequencyMap.find(key_point);
                        if (it != cUVWFrequencyMap.end()) {
                            uvwMapVector[uvw_index] = 1 / (it->second);  // 存储频次的倒数
                        } else {
                            uvwMapVector[uvw_index] = 1; 
                        }
                        // cu, cv, cw 需要存储原始坐标
                        cu[uvw_index] = u_point;
                        cv[uvw_index] = v_point;
                        cw[uvw_index] = w_point;
                        uvw_index++;
                    }
                }               
                cout << "uvw_index: " << uvw_index << endl; 
                
                // 复制到GPU上
                thrust::copy(cu.begin(), cu.begin() + uvw_index, u.begin());
                thrust::copy(cv.begin(), cv.begin() + uvw_index, v.begin());
                thrust::copy(cw.begin(), cw.begin() + uvw_index, w.begin());
                thrust::copy(uvwMapVector.begin(), uvwMapVector.begin() + uvw_index, uvwFrequencyMap.begin());
            }

            // 记录uvw结束事件
            cudaEventRecord(uvwstop);
            cudaEventSynchronize(uvwstop);
            // 计算经过的时间
            float uvwMS = 0;
            cudaEventElapsedTime(&uvwMS, uvwstart, uvwstop);
            printf("Period %d Load UWV Cost Time is: %f s\n", p+1, uvwMS/1000);
            // 销毁事件
            cudaEventDestroy(uvwstart);
            cudaEventDestroy(uvwstop);


            // 记录viss开始事件
            cudaEvent_t vissstart, vissstop;
            cudaEventCreate(&vissstart);
            cudaEventCreate(&vissstop);
            cudaEventRecord(vissstart);

            // 存储计算后的可见度
            thrust::device_vector<Complex> viss(uvw_index);
            launch_visscal(uvw_index, lmnC_index,
                thrust::raw_pointer_cast(viss.data()),
                thrust::raw_pointer_cast(u.data()),
                thrust::raw_pointer_cast(v.data()),
                thrust::raw_pointer_cast(w.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(C.data()),
                I1, CPI, zero, two
            );
            cout << "period" << p+1 << " viss compute success!" << endl;

            // 记录viss结束事件
            cudaEventRecord(vissstop);
            cudaEventSynchronize(vissstop);
            // 计算经过的时间
            float vissMS = 0;
            cudaEventElapsedTime(&vissMS, vissstart, vissstop);
            printf("Period %d Compute Viss Cost Time is: %f s\n", p+1, vissMS/1000);
            // 销毁事件
            cudaEventDestroy(vissstart);
            cudaEventDestroy(vissstop);


            // 记录imagerecon开始事件
            cudaEvent_t imagereconstart, imagereconstop;
            cudaEventCreate(&imagereconstart);
            cudaEventCreate(&imagereconstop);
            cudaEventRecord(imagereconstart);

            launch_imagerecon(uvw_index, lmnC_index,
                thrust::raw_pointer_cast(F.data()),
                thrust::raw_pointer_cast(viss.data()),
                thrust::raw_pointer_cast(u.data()),
                thrust::raw_pointer_cast(v.data()),
                thrust::raw_pointer_cast(w.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(C.data()),
                thrust::raw_pointer_cast(uvwFrequencyMap.data()),
                I1, CPI, zero, two
            );
            cout << "Period " << p+1 << "Image Reconstruction Success!" << endl;
            
            // 记录imagerecon结束事件
            cudaEventRecord(imagereconstop);
            cudaEventSynchronize(imagereconstop);
            // 计算经过的时间
            float imagereconMS = 0;
            cudaEventElapsedTime(&imagereconMS, imagereconstart, imagereconstop);
            printf("Period %d Image Reconstruction Cost Time is: %f s\n", p+1, imagereconMS/1000);
            // 销毁事件
            cudaEventDestroy(imagereconstart);
            cudaEventDestroy(imagereconstop);


            // 记录saveimage开始事件
            cudaEvent_t saveimagestart, saveimagestop;
            cudaEventCreate(&saveimagestart);
            cudaEventCreate(&saveimagestop);
            cudaEventRecord(saveimagestart);
            // 创建一个临界区，用于保存图像反演结果
            #pragma omp critical
            {   
                // 在CPU上创建变量保存F结果
                thrust::host_vector<Complex> tempF = F;

                std::ofstream F_File;
                string address_F = "F_recon_1M/F" + to_string(p+1) + "_healpix.txt";
                cout << "address_F: " << address_F << endl;
                F_File.open(address_F);
                if (!F_File.is_open()) {
                    std::cerr << "Error opening file: " << address_F << endl;
                }
                for (int c = 0; c < lmnC_index; c++) {
                    F_File << tempF[c].real() << std::endl;
                }
                F_File.close();
                std::cout << "Period " << p+1 << " save F success!" << std::endl;
            }

            // 记录saveimage结束事件
            cudaEventRecord(saveimagestop);
            cudaEventSynchronize(saveimagestop);
            // 计算经过的时间
            float saveimageMS = 0;
            cudaEventElapsedTime(&saveimageMS, saveimagestart, saveimagestop);
            printf("Period %d Save Image Cost Time is: %f s\n", p+1, saveimageMS/1000);
            // 销毁事件
            cudaEventDestroy(saveimagestart);
            cudaEventDestroy(saveimagestop);

            // 记录结束事件
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            // 计算经过的时间
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Period %d Elapsed time: %f s\n", p+1, milliseconds/1000);
            // 销毁事件
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
    
    gettimeofday(&finish, NULL);
    total_time = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_usec - start.tv_usec)) / 1000000.0;
    cout << "total time: " << total_time << "s" << endl;
    return 0;
}


int main()
{   
    int start_period = 0;  // 从哪个周期开始，一共是130个周期
    vissGen(0, 2094, start_period);
}

