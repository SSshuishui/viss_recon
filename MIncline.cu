#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_set>
#include <tuple>
#include <set>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include "error.cuh"
#include <omp.h>


// Define constants
const int satnum = 8;
const double mu_e = 4.902793455e3; // unit in km^3/s^2
const double d2r = M_PI / 180;
const double a = 2038.14;
const double e = 0;
const double incl = 30 * d2r;
const double argp = 0 * d2r;
const std::vector<double> r1 = {0, 1e3, 4.45e3, 6e3, 9.1e3, 16.2e3, 35.3e3, 100e3};
const std::vector<double> r2 = {0, 0.1e3, 0.445e3, 0.6e3, 0.91e3, 1.62e3, 3.53e3, 10e3};


// LinSpace函数
struct linspace_functor {
    double a, step;
    linspace_functor(double _a, double _step) : a(_a), step(_step) {}
    __host__ __device__ double operator()(const int& x) const { 
        return a + step * x;
    }
};

void generate_linspace(thrust::device_vector<double>& d_vec, double start, double end, int num) {
    // Calculate step size
    double step = (end - start) / double(num - 1);
    // Generate sequence [0, 1, 2, ..., num-1]
    thrust::sequence(d_vec.begin(), d_vec.end());
    // Transform sequence to linear space
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), linspace_functor(start, step));
}


struct asin_functor {
    double a, factor;
    asin_functor(double _a, double _factor) : a(_a), factor(_factor) {}

    __host__ __device__ double operator()(const double& x) const {
        return asin((x / a) * factor) * 2.0;
    }
};


__global__ void calculateOrbit(double* dM, double* x, double* y, double* z, double* Mt, int OrbitCounts, int OrbitRes, double d2r, double e, double a, double mu_e, double argp, double incl, int index) {
    // 计算全局索引
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算g和k的值
    int k = globalIdx / OrbitRes;  // 圈数
    int po = globalIdx % OrbitRes;  // 点数
    if(k < OrbitCounts && po < OrbitRes) {
        double raan = k * d2r * 0.08 + (po / OrbitRes) * d2r * 0.08 + (index - 1) * OrbitCounts * d2r * 0.08;
        double M = Mt[po] + dM[k * OrbitRes + po];
        double E0 = M;
        for (int i = 1; i < 100; ++i) {
            double M0 = E0 - e * sin(E0);
            double error = M - M0;
            if (abs(error) < 1e-15) {
                break;
            }
            E0 = E0 + error / (1 - e * cos(E0));
        }
        double temp = tan(E0 / 2) / sqrt((1 - e) / (1 + e));
        double theta = atan(temp) * 2;
        double r = a * (1 - e * e) / (1 + e * cos(theta));
        double w = theta + argp;
        int tmp = k * OrbitRes + po;
        x[tmp] = r * (cos(w) * cos(raan) - sin(w) * cos(incl) * sin(raan));
        y[tmp] = r * (cos(w) * sin(raan) + sin(w) * cos(incl) * cos(raan));
        z[tmp] = r * sin(w) * sin(incl);

    }
}


__global__ void uvwPosition(double* x, double* y, double* z, double* xt, double* yt, double* zt, int OrbitCounts, int OrbitRes, int interval, int span) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = globalIdx / OrbitRes + 1;  // 圈数
    int offset, offset_reset_index = 0;
    if(k < OrbitCounts){
        if ((k-1)*(OrbitRes+interval) < OrbitRes*OrbitCounts) {
            offset = (k-1)*interval;
            offset_reset_index = k;
        } else {
            offset = (k-1-offset_reset_index)*interval;
        }
        int endIndexWithoutMin = (k-1)*OrbitRes + span + offset;
        int startIdx = max(0, (k-1)*OrbitRes + offset); // Adjust for C++ 0-based indexing

        if(endIndexWithoutMin > OrbitRes * OrbitCounts){
            xt[globalIdx] = 0;
            yt[globalIdx] = 0;
            zt[globalIdx] = 0;
        }
        else{
            xt[globalIdx] = x[startIdx + (globalIdx-(k-1)*OrbitRes/3)];
            yt[globalIdx] = y[startIdx + (globalIdx-(k-1)*OrbitRes/3)];
            zt[globalIdx] = z[startIdx + (globalIdx-(k-1)*OrbitRes/3)];
        }
    }
}


void MIncline(int index, double frequency, double stride, int gpu_id) {
    int OrbitRes = 2.4 * 3600 * int(1.0 / stride);
    int OrbitCounts = 140;
    int interval = OrbitRes / 360 * 10;  
    int span = OrbitRes / 3; 
    int lambda = 3e8/frequency;

    // thrust::host_vector<double> h_x(OrbitRes * OrbitCounts, 0);  // 初始化为0
    // thrust::host_vector<double> h_y(OrbitRes * OrbitCounts, 0);
    // thrust::host_vector<double> h_z(OrbitRes * OrbitCounts, 0);

    thrust::host_vector<double> h_xt(OrbitRes * OrbitCounts / 3, 0);  // 初始化为0
    thrust::host_vector<double> h_yt(OrbitRes * OrbitCounts / 3, 0);
    thrust::host_vector<double> h_zt(OrbitRes * OrbitCounts / 3, 0);

    std::cout << "global x y z defined successe! " << std::endl;

    // 在GPU上创建xyz和Mt
    
    cudaSetDevice(gpu_id);
    thrust::device_vector<double> x(OrbitRes * OrbitCounts, 0);
    thrust::device_vector<double> y(OrbitRes * OrbitCounts, 0);
    thrust::device_vector<double> z(OrbitRes * OrbitCounts, 0);
    thrust::device_vector<double> Mt(OrbitRes);

    thrust::device_vector<double> xt(OrbitRes * OrbitCounts / 3, 0);
    thrust::device_vector<double> yt(OrbitRes * OrbitCounts / 3, 0);
    thrust::device_vector<double> zt(OrbitRes * OrbitCounts / 3, 0);
    

    generate_linspace(Mt, 0, 2*M_PI, OrbitRes);
    std::cout << "GPU " << gpu_id << " Mt_all compute successe! " << std::endl;

    for (int ss = 0; ss < satnum; ss++) {

        thrust::device_vector<double> d1(OrbitCounts * OrbitRes / 2);
        generate_linspace(d1, r1[ss], r2[ss], OrbitCounts * OrbitRes / 2);
        thrust::device_vector<double> d2(OrbitCounts * OrbitRes / 2);
        generate_linspace(d2, r2[ss], r1[ss], OrbitCounts * OrbitRes / 2);
        std::cout << "GPU " << gpu_id << " sat " << ss+1 << " d1 d2 compute successe! " << std::endl;

        // 创建dM向量，初始大小与d相同
        double a_factor = 2.0 / a / 1000.0;
        thrust::device_vector<double> dM(d1.size() + d2.size());
        // 对d1的每个元素进行转换，存入dM的前半部分
        thrust::transform(d1.begin(), d1.end(), dM.begin(), asin_functor(a, a_factor));
        // 对d2的每个元素进行转换，存入dM的后半部分
        thrust::transform(d2.begin(), d2.end(), dM.begin() + d1.size(), asin_functor(a, a_factor));
        std::cout << "GPU " << gpu_id << " sat " << ss+1 << " dM compute successe! " << std::endl;

        // 用完d1和d2后，释放它们的内存
        d1.clear();
        d1.shrink_to_fit();
        d2.clear();
        d2.shrink_to_fit();

        int blockSize;
        int minGridSize; // 最小网格大小
        int gridSize;    // 实际网格大
        // 记录position事件
        cudaEvent_t posstart, posstop;
        CHECK(cudaEventCreate(&posstart));
        CHECK(cudaEventCreate(&posstop));
        CHECK(cudaEventRecord(posstart));
        CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateOrbit, 0, 0));
        gridSize = floor(OrbitCounts * OrbitRes + blockSize - 1) / blockSize;
        std::cout << "Calculate position, blockSize: " << blockSize << std::endl;
        std::cout << "Calculate position, girdSize: " << gridSize << std::endl;
        calculateOrbit<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(dM.data()),
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(z.data()), 
            thrust::raw_pointer_cast(Mt.data()), 
            OrbitCounts, OrbitRes, d2r, e, a, mu_e, argp, incl, index
        );
        // 记录position结束事件
        CHECK(cudaEventRecord(posstop));
        CHECK(cudaEventSynchronize(posstop));
        // 计算经过的时间
        float posMS = 0;
        CHECK(cudaEventElapsedTime(&posMS, posstart, posstop));
        printf("Frequency-%f Index-%d Sat-%d On GPU-%d Calculate position xyz Cost Time is: %f s\n", frequency, index, ss+1, gpu_id, posMS/1000);
        std::cout << "sat " << ss+1 << " position compute successe! on GPU " << gpu_id << std::endl;
        CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    

        // 调用uvw_position核函数
        // 记录uvw开始事件
        cudaEvent_t uvwstart, uvwstop;
        CHECK(cudaEventCreate(&uvwstart));
        CHECK(cudaEventCreate(&uvwstop));
        CHECK(cudaEventRecord(uvwstart));
        CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, uvwPosition, 0, 0));
        gridSize = floor(OrbitCounts * OrbitRes / 3 + blockSize - 1) / blockSize;
        std::cout << "UVW position, blockSize: " << blockSize << std::endl;
        std::cout << "UVW position, girdSize: " << gridSize << std::endl;
        uvwPosition<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(z.data()), 
            thrust::raw_pointer_cast(xt.data()),
            thrust::raw_pointer_cast(yt.data()),
            thrust::raw_pointer_cast(zt.data()), 
            OrbitCounts, OrbitRes, interval, span
        );
        // 记录uvw结束事件
        CHECK(cudaEventRecord(uvwstop));
        CHECK(cudaEventSynchronize(uvwstop));
        // 计算经过的时间
        float uvwMS = 0;
        CHECK(cudaEventElapsedTime(&uvwMS, uvwstart, uvwstop));
        printf("Frequency-%f Index-%d Sat-%d On GPU-%d Compute xt yt zt Cost Time is: %f s\n", frequency, index, ss+1, gpu_id, uvwMS/1000);
        std::cout << "Will Save! Sat " << ss+1 << " uvw position compute successe! on GPU " << gpu_id << std::endl;
        CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    


        // 记录save开始事件
        cudaEvent_t savestart, savestop;
        CHECK(cudaEventCreate(&savestart));
        CHECK(cudaEventCreate(&savestop));
        CHECK(cudaEventRecord(savestart));
        // 复制到本地
        thrust::copy(xt.begin(), xt.end(), h_xt.begin());
        thrust::copy(yt.begin(), yt.end(), h_yt.begin());
        thrust::copy(zt.begin(), zt.end(), h_zt.begin());
        // 写入到文件中
        std::ostringstream fname;
        fname << "frequency_1M_stride01s2/index" << index << "frequency" << frequency << "sat" << ss+1 << "position.txt";
        std::ofstream file(fname.str());
        for(size_t i = 0; i < xt.size(); i++) {
            file << h_xt[i] << " " << h_yt[i] << " " << h_zt[i] << std::endl;
        }
        file.close();
        // 记录save结束事件
        CHECK(cudaEventRecord(savestop));
        CHECK(cudaEventSynchronize(savestop));
        // 计算经过的时间
        float saveMS = 0;
        CHECK(cudaEventElapsedTime(&saveMS, savestart, savestop));
        printf("Frequency %f Index %d Sat %d On GPU %d Save txt Cost Time is: %f s\n", frequency, index, ss+1, gpu_id, saveMS/1000);
        
    }
    std::cout << "GPU Compute position successe!" << std::endl;
}


int main()
{
    int period = 130;
    double stride = 0.01;
    double frequency = 1e6;
    int index = 1;

    MIncline(1, frequency, stride, 0); 

    // int nDevices;
    // CHECK(cudaGetDeviceCount(&nDevices));
    // // 设置并行区中的线程数
    // omp_set_num_threads(nDevices);
    // #pragma omp parallel
    // {   
    //     int tid = omp_get_thread_num();
    //     #pragma omp for
    //     for (int index=1; index<=period; index++){
    //         MIncline(index, frequency, stride, tid);  
    //     }
    // }

    return 0;
}
