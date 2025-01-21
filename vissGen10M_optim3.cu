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

string address = "./frequency_10M/";
string lmn_address = "./lmn10M/";
string duration = "frequency10M";  // 第几个周期的uvw
string sufix = ".txt";

// 10 M
const int uvw_presize = 14400000;

// 定义常量
#define BLOCK_SIZE 128                     // 线程块大小
#define SHARED_MEM_SIZE BLOCK_SIZE         // 共享内存大小
#define MAX_THREADS_PER_BLOCK 1024        // GPU每个块的最大线程数


struct clip_functor {
    __host__ __device__
    float operator()(float x) const {
        return max(-1.0f, min(1.0f, x));
    }
};


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
    const Complex two,
    const float dl,
    const float dm,
    const float dn)
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
    const float u_val = u[uvw_] / dl;
    const float v_val = v[uvw_] / dm;
    const float w_val = w[uvw_] / dn;

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
        #pragma unroll 4  // 添加循环展开指令
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
    const Complex two,
    const float dl,
    const float dm,
    const float dn)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, visscal, 0, 0);
    int blocksPerGrid = floor(uvw_index + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    const int numStreams = 4;  // 使用4个流
    const int itemPerStream = uvw_index / numStreams;  // 计算每个流处理的数据量

    // 创建流数组
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 计算共享内存大小
    size_t sharedMemSize = SHARED_MEM_SIZE * 4 * sizeof(float);  // 4个float数组
    // 设置缓存配置
    cudaFuncSetCacheConfig(visscal, cudaFuncCachePreferShared);

    // 启动多个流
    for(int i = 0; i < numStreams; i++) {
        const int streamStart = i * itemPerStream;
        const int streamSize = (i == numStreams-1) ? uvw_index-streamStart : itemPerStream;

        if(streamSize <= 0) break;

        const int streamBlocks = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

        // 启动核函数
        visscal<<<streamBlocks, threadsPerBlock, sharedMemSize, streams[i]>>>(
            streamSize, lmnC_index,
            d_viss + streamStart, // 只处理部分 viss
            d_u + streamStart, // 只处理部分 u
            d_v + streamStart, // 只处理部分 v
            d_w + streamStart, // 只处理部分 w
            d_l, 
            d_m, 
            d_n, 
            d_C,
            I1, CPI, zero, two,
            dl, dm, dn
        );
    }

    // 等待所有流完成（如果需要）
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 清理流
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
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
    const float* __restrict__ uvwFrequencyMap,
    const Complex I1,                    
    const Complex CPI,
    const Complex zero,
    const Complex two,
    const float dl,
    const float dm,
    const float dn)
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
    const float l_val = l[lmnC_] / dl;
    const float m_val = m[lmnC_] / dm;
    const float n_val = n[lmnC_] / dn;

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
        #pragma unroll 4
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
    float* d_uvwFrequencyMap,
    const Complex I1,
    const Complex CPI,
    const Complex zero,
    const Complex two,
    const float dl,
    const float dm,
    const float dn)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, visscal, 0, 0);
    int blocksPerGrid = floor(lmnC_index + threadsPerBlock - 1) / threadsPerBlock;

    // 检查共享内存大小
    size_t sharedMemSize = SHARED_MEM_SIZE * (5 * sizeof(float) + sizeof(Complex));
    
    // 设置流的数量
    const int numStreams = 4;  // 可以根据需要调整
    const int itemPerStream = (lmnC_index + numStreams - 1) / numStreams;

    // 创建流数组
    cudaStream_t streams[numStreams];
    for(int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 设置缓存配置
    cudaFuncSetCacheConfig(imagerecon, cudaFuncCachePreferShared);

    // 启动多个流
    for(int i = 0; i < numStreams; i++) {
        const int streamStart = i * itemPerStream;
        const int streamSize = (i == numStreams-1) ? lmnC_index-streamStart : itemPerStream;

        if(streamSize <= 0) break;

        const int streamBlocks = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

        // 启动核函数
        imagerecon<<<streamBlocks, threadsPerBlock, sharedMemSize, streams[i]>>>(
            uvw_index, streamSize,
            d_F + streamStart, // 只处理部分 F
            d_viss,
            d_u, 
            d_v, 
            d_w,
            d_l + streamStart, // 只处理部分 l
            d_m + streamStart, // 只处理部分 m
            d_n + streamStart, // 只处理部分 n
            d_uvwFrequencyMap,
            I1, CPI, zero, two,
            dl, dm, dn
        );
    }

    // 等待所有流完成（如果需要）
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 清理流
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
}


int vissGen(int id, int RES, int start_period) 
{   
    cout << "res: " << RES << endl;
    int days = 11;  // 一共有多少个周期  15月 * 30天 / 14天/周期
    cout << "periods: " << days << endl;
    Complex I1(0.0, 1.0);
    float dl = 2 * RES / (RES - 1);
    float dm = 2 * RES / (RES - 1);
    float dn = 2 * RES / (RES - 1);
    Complex zero(0.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);

    gettimeofday(&start, NULL);
    int nDevices;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    // 加载存储 l m n C 的文件（对于不同的frequency不一样，只与frequency有关）
    string para, address_l, address_m, address_n, address_C;
    ifstream lFile, mFile, nFile, CFile;
    para = "l";
    address_l = lmn_address + para + sufix;
    lFile.open(address_l);
    cout << "address_l: " << address_l << endl;
    para = "m";
    address_m = lmn_address + para + sufix;
    mFile.open(address_m);
    cout << "address_m: " << address_m << endl;
    para = "n";
    address_n = lmn_address + para + sufix;
    nFile.open(address_n);
    cout << "address_n: " << address_n << endl;
    para = "C";
    address_C = lmn_address + para + sufix;
    CFile.open(address_C);
    cout << "address_C: " << address_C << endl;
    if (!lFile.is_open() || !mFile.is_open() || !nFile.is_open() || !CFile.is_open()) {
        std::cerr << "无法打开一个或多个文件：" << std::endl;
        if (!lFile.is_open()) std::cerr << "无法打开文件: " << address_l << std::endl;
        if (!mFile.is_open()) std::cerr << "无法打开文件: " << address_m << std::endl;
        if (!nFile.is_open()) std::cerr << "无法打开文件: " << address_n << std::endl;
        if (!CFile.is_open()) std::cerr << "无法打开文件: " << address_C << std::endl;
        return -1; 
    }
    int lmnC_index = 0;
    lFile >> lmnC_index;  // 读取l的第一行的行数
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
            thrust::device_vector<float> C(cC.begin(), cC.end());
            thrust::device_vector<float> l(cl.begin(), cl.end());
            thrust::device_vector<float> m(cm.begin(), cm.end());
            thrust::device_vector<float> n(cn.begin(), cn.end());
            // 将 n 数据限制在 -1 到 1 之间
            thrust::transform(n.begin(), n.end(), n.begin(), clip_functor());
            cout << "period" << p+1 << " n transfer success!" << endl;

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
                string address_uvw = address + "updated_uvw" + to_string(p+1) + duration + sufix;
                cout << "address_uvw: " << address_uvw << std::endl;
                
                ifstream uvwFile(address_uvw);
                // 同时用一个向量保存每一个uvw坐标点的frequency
                uvw_index = 0;
                float u_point, v_point, w_point, freq_point;
                if (uvwFile.is_open()) {
                    while (uvwFile >> u_point >> v_point >> w_point >> freq_point) {
                        // cu, cv, cw 需要存储原始坐标
                        cu[uvw_index] = u_point;
                        cv[uvw_index] = v_point;
                        cw[uvw_index] = w_point;
                        uvwMapVector[uvw_index] = 1 / freq_point;
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
            cout << "Compute Viss ..." << endl;
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
                I1, CPI, zero, two, dl, dm, dn
            );
            CHECK(cudaDeviceSynchronize());
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

            cout << "Image Reconstruction ..." << endl;
            launch_imagerecon(uvw_index, lmnC_index,
                thrust::raw_pointer_cast(F.data()),
                thrust::raw_pointer_cast(viss.data()),
                thrust::raw_pointer_cast(u.data()),
                thrust::raw_pointer_cast(v.data()),
                thrust::raw_pointer_cast(w.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(uvwFrequencyMap.data()),
                I1, CPI, zero, two, dl, dm, dn
            );
            CHECK(cudaDeviceSynchronize());
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
            // 创建一个临界区，用于保存结果
            #pragma omp critical
            {   
                // 将数据从设备内存复制到主机内存
                std::vector<Complex> host_F(F.size());
                CHECK(cudaMemcpy(host_F.data(), thrust::raw_pointer_cast(F.data()), F.size() * sizeof(Complex), cudaMemcpyDeviceToHost));
                CHECK(cudaDeviceSynchronize());
                // 打开文件
                string address_F = "F_recon_10M/F" + to_string(p+1) + "period10M_optim3.txt";
                cout << "Period " << p+1 << " save address_F: " << address_F << endl;
                std::ofstream file(address_F);
                if (file.is_open()) {
                    // 按照指定格式写入文件
                    for(const Complex& value : host_F)
                    {
                        file << value.real() << std::endl;
                    }
                }
                // 关闭文件
                file.close();
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
    int start_period = 10;  // 从哪个周期开始，一共是130个周期
    vissGen(0, 20940, start_period);
}

