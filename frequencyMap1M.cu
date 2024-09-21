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
__global__ void countFrequency(Coord* coords, int* freqArray, int nCoords, int dimX, int dimY, int dimZ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // 全局索引，用于访问coords
    if (index < nCoords) {
        // 获取当前坐标
        Coord coord = coords[index];
        // 确保坐标在数组维度范围内
        if(coord.x+dimX/2-1 < dimX && coord.y+dimY/2-1 < dimY && coord.z+dimZ/2-1 < dimZ) {
            // 计算在频次数组中的索引
            int freqIndex = (coord.x+dimX/2-1) + dimX * ((coord.y+dimY/2-1) + dimY * (coord.z+dimZ/2-1));

            // 原子地更新频次数组
            atomicAdd(&freqArray[freqIndex], 1);
        }
    }
}

int main(int argc, char **argv) {

    auto pos_start = std::chrono::high_resolution_clock::now();

    int nDevices;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(4);
    std::cout << "devices: " << nDevices << std::endl;
    
    int num_files; // 文件数目
    for(num_files=130; num_files<=130; num_files++){
        std::cout << "==================================" << std::endl;
        std::cout << "Num of Files: " << num_files << std::endl;
        std::cout << "==================================" << std::endl;

        int dimX = 666, dimY = 666, dimZ = 334;

        const size_t freqArraySize = dimX * dimY * dimZ; // 频次数组的大小
        // 全局频次坐标
        std::vector<int> globalFreqArray(freqArraySize, 0);

        // GPU预处理：在每个GPU上构建和初始化频次数组
        std::vector<thrust::device_vector<int>> deviceFreqArrays(nDevices);
        for (int i = 0; i < nDevices; ++i) {
            cudaSetDevice(i);
            deviceFreqArrays[i].resize(freqArraySize);
            thrust::fill(deviceFreqArrays[i].begin(), deviceFreqArrays[i].end(), 0); // 初始化为0
        }

        // 开启cpu线程并行
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();  // 从 0 开始编号的并行执行线程
            int totalThreads = omp_get_num_threads(); // 获取总线程数

            // 并行读取文件：每个omp进程读取分配给它的文件，并在关联的GPU上并行处理每个文件。
            thrust::host_vector<Coord> h_Coords;  // 用于存储读取的坐标的CPU向量
            std::vector<std::string> files_to_read;  // 每个进程的文件列表

            // 分配文件给每个进程
            for(int file = tid; file < num_files; file += totalThreads) {
                std::string filename = "frequency_1M_stride01s_sample2400_2/uvw" + std::to_string(file + 1) + "frequency1M.txt";
                files_to_read.push_back(filename);
            }

            // 输出当前进程将要处理的文件
            for (const auto& file : files_to_read) {
                std::cout << "Process " << tid << " reading file: " << file << std::endl;
                readtxt(file, h_Coords);
            }
            std::cout << "size: " << h_Coords.size() << std::endl;

            // 读取文件的同步点
            #pragma omp barrier   

            // 按照tid % nDevices来控制访问GPU，以实现GPU的串行处理
            for (int i = 0; i < omp_get_num_threads(); ++i) {
                if (i % nDevices == tid % nDevices) {
                    cudaSetDevice(tid % nDevices);
                    std::cout << "Thread " << tid << " is running on device " << (tid % nDevices) << std::endl;

                    // 将数据从CPU向量复制到GPU向量
                    thrust::device_vector<Coord> d_Coords = h_Coords;

                    // 调用核函数统计频次
                    int threadsPerBlock = 1024;
                    int blocksPerGrid =(h_Coords.size() + threadsPerBlock - 1) / threadsPerBlock;

                    // 调用核函数
                    countFrequency<<<blocksPerGrid, threadsPerBlock>>>(
                        thrust::raw_pointer_cast(d_Coords.data()), 
                        thrust::raw_pointer_cast(deviceFreqArrays[(tid % nDevices)].data()), 
                        d_Coords.size(), 
                        dimX, dimY, dimZ
                    );
                    // 等待GPU完成任务
                    cudaDeviceSynchronize();
                }
                // 同步所有线程，确保一个GPU任务完成后，再执行下一个
                #pragma omp barrier
            }
        }
        std::cout << "All GPU frequency computed success!" << std::endl;

        // 将所有GPU的频次数组传回CPU并合并
        for(int i = 0; i < nDevices; ++i) {
            // 传回当前GPU的频次数组
            thrust::host_vector<int> h_freqArray = deviceFreqArrays[i];
            // 累加到全局频次数组
            for(size_t j = 0; j < freqArraySize; ++j) {
                globalFreqArray[j] += h_freqArray[j];
            }
        }

        std::cout << "All frequency transfered to CPU success!" << std::endl;
        
        // 写入文件中  验证正确
        std::ofstream outFile("frequency_1M_stride01s_sample2400_2/uvwMapfreq1M"+std::to_string(num_files)+".txt");
        for(int z = 0; z < dimZ; ++z) {
            for(int y = 0; y < dimY; ++y) {
                for(int x = 0; x < dimX; ++x) {
                    // 计算频次数组中的索引
                    int freqIndex = x + dimX * (y + dimY * z);
                    // 获取频次
                    int frequency = globalFreqArray[freqIndex];
                    // 如果频次不为0，则写入文件
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
        // 关闭文件
        outFile.close();
        std::cout << "Save Frequency Map Success!" << std::endl;

        auto pos_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> pos_ms = pos_end - pos_start;
        std::cout << "Position Generated in " << pos_ms.count()/1000 << "s" << std::endl;
    }
    return 0;
}



