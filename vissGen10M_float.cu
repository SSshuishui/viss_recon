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


// 定义计算可见度核函数, 验证一致
__global__ void visscal(
            int uvw_index, int lmnC_index, 
            Complex *viss, float *u, float *v, float *w,
            float *l, float *m, float *n, float *C,
            Complex I1, Complex CPI, Complex zero, Complex two, 
            float dl, float dm, float dn)
{
    int uvw_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (uvw_ < uvw_index)
    {   
        for (int lmnC_ = 0; lmnC_ < lmnC_index; ++lmnC_) {
            Complex temp;
            Complex vari(u[uvw_]*l[lmnC_]/dl + v[uvw_]*m[lmnC_]/dm + w[uvw_]*(n[lmnC_]-1)/dn, 0.0f);
            temp = Complex(C[lmnC_], 0.0f) * complexExp((zero - I1) * two * CPI * vari);
            viss[uvw_] += temp;
        } 
        Complex cw(w[uvw_]/dn, 0.0f);
        viss[uvw_] *= complexExp((zero-I1) * two * CPI * cw);
    }
}


// 定义图像反演核函数  验证正确
__global__  void imagerecon(int uvw_index, int lmnC_index, 
            Complex *F, Complex *viss, float *u, float *v, float *w,
            float *l, float *m, float *n, float *C, float *uvwFrequencyMap,
            Complex I1, Complex CPI, Complex zero, Complex two, 
            float dl, float dm, float dn)
{
    Complex amount(uvw_index, 0.0);
    
    int lmnC_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (lmnC_ < lmnC_index){  
        for(int uvw_=0; uvw_<uvw_index; ++uvw_)
        {   
            Complex temp;
            Complex vari(u[uvw_]*l[lmnC_]/dl + v[uvw_]*m[lmnC_]/dm + w[uvw_]*n[lmnC_]/dn, 0.0f);
            temp = uvwFrequencyMap[uvw_] * viss[uvw_] * complexExp(I1 * two * CPI * vari);
            F[lmnC_] = F[lmnC_] + temp;
        }
        F[lmnC_] = F[lmnC_] / amount;
    }
}


int vissGen(int id, int totalnode, int RES, int start_period) 
{   
    int days = 9;  // 一共有多少个周期  15月 * 30天 / 14天/周期
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

    // 加载存储 l m n C nt的文件（对于不同的frequency不一样，只与frequency有关）
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
                        uvwMapVector[uvw_index] = freq_point;
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
            int blockSize;
            int minGridSize; // 最小网格大小
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, visscal, 0, 0);
            int gridSize = floor(uvw_index + blockSize - 1) / blockSize;;  
            cout << "Viss Computing, blockSize: " << blockSize << endl;
            cout << "Viss Computing, girdSize: " << gridSize << endl;
            printf("Viss Computing... Here is gpu %d running process %d on node %d\n", omp_get_thread_num(), p+1, id);
            // 调用函数计算可见度

            visscal<<<gridSize, blockSize>>>(uvw_index, lmnC_index,
                    thrust::raw_pointer_cast(viss.data()),
                    thrust::raw_pointer_cast(u.data()),
                    thrust::raw_pointer_cast(v.data()),
                    thrust::raw_pointer_cast(w.data()),
                    thrust::raw_pointer_cast(l.data()),
                    thrust::raw_pointer_cast(m.data()),
                    thrust::raw_pointer_cast(n.data()),
                    thrust::raw_pointer_cast(C.data()),
                    I1, CPI, zero, two, dl, dm, dn);
            // 进行线程同步
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

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, imagerecon, 0, 0);
            gridSize = floor(lmnC_index + blockSize - 1) / blockSize;
            cout << "Image Reconstruction, blockSize: " << blockSize << endl;
            cout << "Image Reconstruction, girdSize: " << gridSize << endl;
            printf("Image Reconstruction... Here is gpu %d running process %d on node %d\n",omp_get_thread_num(),p+1,id);
            // 调用image_recon函数计算图像反演
            imagerecon<<<gridSize,blockSize>>>(
                uvw_index, lmnC_index, 
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
                I1, CPI, zero, two, dl, dm, dn);
            // 进行线程同步
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
                string address_F = "cudaF/F" + to_string(p+1) + "period10M.txt";
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
    int start_period = 6;  // 从哪个周期开始，一共是130个周期
    vissGen(0, 1, 2094, start_period);
}

