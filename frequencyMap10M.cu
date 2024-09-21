#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <omp.h>
#include "error.cuh"
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct Coord {
    int x;
    int y;
    int z;
};

// 读取文件
void readtxt(const std::string &filename, thrust::host_vector<Coord> &coords) {
    std::ifstream file(filename); // Open the file
    Coord temp;
    // Check if the file is open
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    while (file >> temp.x >> temp.y >> temp.z) {
        coords.push_back(temp); 
    }
    file.close();
}


// 统计频次的GPU任务
__global__ void countFrequency(Coord* coords, size_t* freqArray, int nCoords, int dimX, int dimY, int dimZ, int blockZ, int period, int gpu, int nDevices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;  // 线程索引
    if (index < nCoords) {
        Coord coord = coords[index]; // 遍历每一个坐标
        int blockZStart = ((period - 1) * nDevices + gpu) * blockZ;
        int blockZEnd = (blockZStart + blockZ < dimZ) ? (blockZStart + blockZ) : dimZ;
        // 确保索引在范围内
        // 每一次z的范围是一个blockZ    0-150, 150-300, ...
        // if (coord.x+dimX/2-1<dimX && coord.y+dimY/2-1<dimY && coord.z+dimZ/2-1<dimZ) 
        if (coord.x + dimX / 2 - 1 >= 0 && coord.y + dimY / 2 - 1 >= 0 && coord.z + dimZ / 2 - 1 >= blockZStart &&
            coord.x + dimX / 2 - 1 < dimX && coord.y + dimY / 2 - 1 < dimY && coord.z + dimZ / 2 - 1 < blockZEnd)
        {
            // 计算索引
            size_t idx = static_cast<size_t>(coord.x+dimX/2-1) + static_cast<size_t>(dimX) * static_cast<size_t>(coord.y+dimY/2-1) + static_cast<size_t>(dimX) * static_cast<size_t>(dimY) * static_cast<size_t>(coord.z+dimZ/2-1 - blockZStart);
            // 使用原子操作增加频次计数
            atomicAdd(reinterpret_cast<unsigned int*>(freqArray + idx), 1);

        }
    }
}

int main(int argc, char **argv) {

    auto pos_start = std::chrono::high_resolution_clock::now();
    int num_files = 130; // 文件数目
    int dimX = 6646, dimY = 6646, dimZ = 3334;

    int nDevices;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    std::cout << "devices: " << nDevices << " File_Num: " << num_files << std::endl;
    
    size_t blockZ = 50; // 一次处理的长度
    size_t freqArraySize = static_cast<size_t>(dimX) * static_cast<size_t>(dimY) * static_cast<size_t>(blockZ); // 频次数组的大小

    for(int period=1; period<=dimZ/(blockZ*nDevices)+1; period++){
        // GPU预处理：在每个GPU上构建和初始化频次数组
        std::vector<thrust::device_vector<size_t>> deviceFreqArrays(nDevices);
        for (int i = 0; i < nDevices; ++i) {
            cudaSetDevice(i);
            deviceFreqArrays[i].resize(freqArraySize);
            thrust::fill(deviceFreqArrays[i].begin(), deviceFreqArrays[i].end(), 0); // 初始化为0
        }
        
        std::cout << "Period " << period << " GPU Initialized!" << std::endl;

        #pragma omp parallel
        {   
            int tid = omp_get_thread_num();  // 从 0 开始编号的并行执行线程
            int totalThreads = omp_get_num_threads(); // 获取总线程数

            // 分配文件给每个进程,所有文件都要读取
            for(int file = 0; file < num_files; file++) {
                std::string filename = "frequency_10M/uvw" + std::to_string(file + 1) + "frequency10M.txt";
                std::cout << "Process " << tid << " reading file: " << filename << std::endl;

                thrust::host_vector<Coord> h_Coords;  // 用于存储读取的坐标的CPU向量
                readtxt(filename, h_Coords);
            
                std::cout << "size: " << h_Coords.size() << std::endl;

                cudaSetDevice(tid % nDevices);
                std::cout << "Thread " << tid << " is running on device " << (tid % nDevices) << std::endl;

                // 将数据从CPU向量复制到GPU向量
                thrust::device_vector<Coord> d_Coords = h_Coords;

                // 调用核函数统计频次
                int threadsPerBlock = 1024;
                int blocksPerGrid =(h_Coords.size() + threadsPerBlock - 1) / threadsPerBlock;
                std::cout << "thread " << tid<< " threadsPerBlock " << threadsPerBlock << " blocksPerGrid " << blocksPerGrid << std::endl;

                // 调用核函数
                countFrequency<<<blocksPerGrid, threadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_Coords.data()), 
                    thrust::raw_pointer_cast(deviceFreqArrays[(tid%nDevices)].data()), 
                    d_Coords.size(), 
                    dimX, dimY, dimZ, blockZ, period, tid%nDevices, nDevices
                );

                std::cout << "file: " << filename << " Processed!" << std::endl;

                // 等待GPU完成任务
                CHECK(cudaDeviceSynchronize());
            }

            // 同步点，确保所有GPU任务都完成
            #pragma omp barrier
        }

        std::cout << "Period "<< period<<" GPU frequency computed success!" << std::endl;

        // 将所有GPU的频次数组传回CPU并存储
        for(int i = 0; i < nDevices; ++i) {
            // 传回当前GPU的频次数组
            thrust::host_vector<size_t> h_freqArray = deviceFreqArrays[i];
            int startZ = ((period - 1) * nDevices + i) * blockZ;
            
            // 写入文件中  验证正确
            std::ofstream outFile("frequency_10M/uvw_seg/uvwMap130_"+std::to_string(startZ)+"_"+std::to_string(startZ + blockZ)+".txt");
            for(int z = ((period-1)*nDevices+i)*blockZ; z < ((period-1)*nDevices+i+1)*blockZ; ++z) {
                for(int y = 0; y < dimY; ++y) {
                    for(int x = 0; x < dimX; ++x) {
                        // 计算频次数组中的索引
                        size_t freqIndex = static_cast<size_t>(x) + static_cast<size_t>(dimX) * (static_cast<size_t>(y) + static_cast<size_t>(dimY) * static_cast<size_t>(z-((period-1)*nDevices+i)*blockZ));
                        // 获取频次
                        int frequency = h_freqArray[freqIndex];
                        // 如果频次不为1，则写入文件
                        if(frequency > 1) {
                            // 计算映射回的原始坐标
                            int origX = x - dimX / 2 + 1;
                            int origY = y - dimY / 2 + 1;
                            int origZ = z - dimZ / 2 + 1;
                            // 写入文件
                            outFile << origX << " " << origY << " " << origZ << " " << frequency << std::endl;
                        }
                    }
                }
            }
            std::cout << "Save Frequency Map Success! Period: " << period << " GPU: " << i << std::endl;
            // 关闭文件
            outFile.close();

            // 释放h_freqArray的内存
            h_freqArray.clear();
        }

        std::cout << "Save Period: " <<period << " Success!" << std::endl;
    }
    auto pos_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pos_ms = pos_end - pos_start;
    std::cout << "Position Generated in " << pos_ms.count()/1000 << "s" << std::endl;
    return 0;
    
}


