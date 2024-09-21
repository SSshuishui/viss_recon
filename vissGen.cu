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

#define _USE_MATH_DEFINES
#define EXP 0.0000000000

using namespace std;

class Complex {
public:

    __host__ __device__
    Complex(double real = 0.0, double image = 0.0)
    {
        _real = real;
        _image = image;
        //cout<<"Complex(double real,double image)"<<endl;
    }

    __host__ __device__
    Complex(const Complex &d)
    {
        _image = d._image;
        _real = d._real;
    }

    __host__ __device__
    ~Complex()
    {
        //cout<<"~Complex()"<<endl;
    }

    __device__
    Complex &operator=(const Complex &d)
    {
        //cout<<"Complex& operator=(const Complex& d)"<<endl;
        if (this != &d) {
            _real = d._real;
            _image = d._image;
        }
        return *this;
    }


    __device__
    Complex operator+(const Complex &d) {
        Complex ret;
        ret._real = (this->_real + d._real);
        ret._image = (this->_image + d._image);
        return ret;
    }

    __device__
    Complex operator-(const Complex &d) {
        Complex ret;
        ret._real = (this->_real - d._real);
        ret._image = (this->_image - d._image);
        return ret;
    }


    __device__
    Complex operator*(const Complex &d) {
        Complex multi;
        multi._real = this->_real * d._real - this->_image * d._image;
        multi._image = this->_real * d._image + this->_image * d._real;
        return multi;
    }

    __device__
    Complex operator/(const Complex &d) {
        Complex devide;
        double temp = d._real * d._real + d._image * d._image;
        devide._real = (this->_real * d._real + this->_image * d._image) / temp;
        devide._image = (this->_image * d._real - this->_real * d._image) / temp;
        return devide;
    }

    __device__
    static Complex complexExp(const Complex &d) {
        Complex result;
        result._real = exp(d._real) * cos(d._image);
        result._image = exp(d._real) * sin(d._image);
        return result;
    }

    __device__
    static Complex complexAbs(const Complex &d) {
        Complex result;
        result._real = sqrt((d._real * d._real + d._image * d._image));
        result._image = 0;
        return result;
    }

    double _real;
    double _image;
};

struct timeval start, finish;
double total_time;

string address = "./frequency_1M_stride01s_sample2400_2/";
string viss_address = "./Viss_1M/";
string F_address = "./F_recon_1M_stride01s_sample2400_2/";
string para;
string duration = "period";  // 第几个周期的uvw
string sufix = ".txt";
const int amount = 130;  // 一共有多少个周期  15月 * 30天 / 14天/周期

// 1 M
const int uvw_size = 4000000;  
const int lmnC_size = 5000000;
const int NX_size = 5000000;


// 定义计算C的核函数，每一个线程处理一个q的值，q为0-nn的范围， 但是NX中保存的索引是1-nn，因此需要对齐，验证正确
__global__ void computeC(Complex *NX, Complex *FF, Complex *C, int nn, int NX_size) {
    int q = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (q <= nn){
        Complex sum(0.0, 0.0);
        int count=0;
        for (int i=0; i < NX_size; ++i)
        {
            if (NX[i]._real == q)
            {
                sum._real += FF[i]._real;
                count++;
            }
        }
        if (count > 0) {
            C[q-1]._real = sum._real / count;
        }
    }
}


// 定义计算可见度核函数，每一个线程处理一个，验证正确
__global__ void visscal(int RES, int uvw_index, int lmnC_index, Complex *viss,
             Complex *u, Complex *v, Complex *w,
             Complex *l, Complex *m, Complex *n, Complex *C) 
{
    Complex I1(0.0, 1.0);
    // 对应vissGen中的dl dm dn
    Complex dl((double) 2 * RES / (RES - 1), 0.0);
    Complex dm((double) 2 * RES / (RES - 1), 0.0);
    Complex dn((double) 2 * RES / (RES - 1), 0.0);
    Complex zero(0.0, 0.0);
    Complex one(1.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);

    // viss[i] = viss[i] + C[j] * Complex::complexExp((zero - I1) * two * CPI * (u[i]*l[j]/dl + v[i]*m[j]/dm + w[i]*(n[j]-one) / dn)) / Complex::complexAbs(nt[j]);
    int uvw_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (uvw_ < uvw_index)
    {
        viss[uvw_] = zero;
        for (int j = 0; j < lmnC_index; ++j) {
            viss[uvw_] = viss[uvw_] + C[j] * Complex::complexExp((zero - I1) * two * CPI * (u[uvw_]*l[j]/dl + v[uvw_]*m[j]/dm + w[uvw_]*(n[j]-one)/dn));
        } 
        viss[uvw_] = viss[uvw_] * Complex::complexExp((zero-I1) * two * CPI * w[uvw_] / dn);
    }
}

// 定义图像反演核函数  验证正确
__global__  void imagerecon(int RES, int uvw_index, int lmnC_index, Complex *F,
             Complex *viss, Complex *u, Complex *v, Complex *w,
             Complex *l, Complex *m, Complex *n, Complex *C, Complex *uvwFrequencyMap)
{
    Complex I1(0.0, 1.0);
    Complex dl((double)2*RES/(RES-1),0.0);
    Complex dm((double)2*RES/(RES-1),0.0);
    Complex dn((double)2*RES/(RES-1),0.0);

    Complex zero(0.0,0.0);
    Complex one(1.0,0.0);
    Complex two(2.0,0.0);
    Complex CPI(M_PI,0.0);
    Complex amount((double)uvw_index, 0.0);

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < lmnC_index)
    {
        F[index] = zero;
        for(int i=0; i<uvw_index; i++)
        {
            Complex temp;
            temp = uvwFrequencyMap[i] * viss[i] * Complex::complexExp(I1 * two * CPI * (u[i]*l[index]/dl + v[i]*m[index]/dm + w[i]*n[index]/dn));
            F[index] = F[index] + temp;
        }
        F[index] = F[index] / amount;
    }
}


int vissGen(int id, int totalnode, int RES) 
{
    gettimeofday(&start, NULL);
    
    int nDevices;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    // 加载存储 l m n C nt的文件（对于不同的frequency不一样，只与frequency有关）
     // 创建复数变量
    Complex *cl;
    Complex *cm;
    Complex *cn;
    Complex *cNX;
    Complex *cFF;

    cl = (Complex *) malloc(lmnC_size * sizeof(Complex));
    cm = (Complex *) malloc(lmnC_size * sizeof(Complex));
    cn = (Complex *) malloc(lmnC_size * sizeof(Complex));
    cnt = (Complex *) malloc(lmnC_size * sizeof(Complex));
    cNX = (Complex *) malloc(NX_size * sizeof(Complex));
    cFF = (Complex *) malloc(NX_size * sizeof(Complex));

    string para, address_l, address_m, address_n, address_NX, address_FF, address_nt;
    ifstream lFile, mFile, nFile, NXFile, FFFile, ntFile;

    para = "l";
    address_l = address + para + sufix;
    lFile.open(address_l);
    cout << "address_l: " << address_l << endl;
    para = "m";
    address_m = address + para + sufix;
    mFile.open(address_m);
    cout << "address_m: " << address_m << endl;
    para = "n";
    address_n = address + para + sufix;
    nFile.open(address_n);
    cout << "address_n: " << address_n << endl;
    para = "NX";
    address_NX = address + para + sufix;
    NXFile.open(address_NX);
    cout << "address_NX: " << address_NX << endl;
    para = "FF";
    address_FF = address + para + sufix;
    FFFile.open(address_FF);
    cout << "address_FF: " << address_FF << endl;

    para = "nt";
    address_nt = address + para + sufix;
    ntFile.open(address_nt);
    cout << "address_nt: " << address_nt << endl;

    int lmnC_index = 0;
    int NX_index = 0;
    while (lFile >> cl[lmnC_index]._real && mFile >> cm[lmnC_index]._real && nFile >> cn[lmnC_index]._real) 
    { 
        lmnC_index++; 
    }
    cout << "lmnC index: " << lmnC_index << endl;

    while (NXFile >> cNX[NX_index]._real && FFFile >> cFF[NX_index]._real) 
    { 
        NX_index++; 
    }
    cout << "NX index: " << NX_index << endl;



    lFile.close();
    mFile.close();
    nFile.close();
    NXFile.close();
    FFFile.close();
    ntFile.close();

    // 导入uvw坐标的出现频次，txt文件的每一行每个坐标的频次
    auto uvwMapStart = std::chrono::high_resolution_clock::now();

    // 创建map存储
    std::unordered_map<std::string, double> cUVWFrequencyMap;
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

        // 每一行的格式： -1,-1,0 5
        string line;
        while (std::getline(uvwMapFile, line)) {
            std::istringstream iss(line);
            int u_point, v_point, w_point;
            double uvw_frequency;
            // char comma;  // 逗号
            
            // if (iss >> u_point >> comma >> v_point >> comma >> w_point >> uvw_frequency) {
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
    int numElementsToPrint = 10; // 设定要打印的元素数量
    for (const auto& pair : cUVWFrequencyMap) {
        std::cout << pair.first << ": " << pair.second << std::endl;
        if (++count == numElementsToPrint) {
            break;
        }
    }

    cout << "Transfer uvw Frequency Success! " << endl;
    auto uvwMapFinish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> uvwMapElapsed = uvwMapFinish - uvwMapStart;
    cout << "Transfer uvw Frequency Elapsed Time: " << uvwMapElapsed.count() << " s\n";


    // 使用一个GPU计算处理得到C
    auto computeCStart = std::chrono::high_resolution_clock::now();

    CHECK(cudaSetDevice(0)); 
    Complex *NX;
    Complex *FF;
    Complex *C;
    CHECK(cudaMalloc(&NX, NX_index * sizeof(Complex)));
    CHECK(cudaMalloc(&FF, NX_index * sizeof(Complex)));
    CHECK(cudaMalloc(&C, lmnC_index * sizeof(Complex)));

    CHECK(cudaMemcpy(NX, cNX, NX_index * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(FF, cFF, NX_index * sizeof(Complex), cudaMemcpyHostToDevice));
    

    // 调用CUDA核函数计算C
    int nn = lmnC_index;
    int blockSize = 512;   // 块大小,一个线程块里面用多少个线程
    int gridSize = (nn + blockSize - 1) / blockSize;;    // 线程块的数量
    cout << "Calculate C, blockSize: " << blockSize << endl;
    cout << "Calculate C, girdSize: " << gridSize << endl;

    computeC<<<gridSize, blockSize>>>(NX, FF, C, nn, NX_index);
    
    CHECK(cudaDeviceSynchronize());
    std::cout << "C is computed in GPU 0!" << std::endl;

    // 将计算得到的C复制回主机
    Complex *C_host = (Complex *) malloc(lmnC_index * sizeof(Complex));
    CHECK(cudaMemcpy(C_host, C, lmnC_index * sizeof(Complex), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(C)); // 释放GPU上的C
    CHECK(cudaFree(NX));
    CHECK(cudaFree(FF));

    auto computeCFinish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computeCElapsed = computeCFinish - computeCStart;
    cout << "Compute C Elapsed Time: " << computeCElapsed.count() << " s\n";


    // 开启cpu线程并行
    // 一个线程处理1个GPU
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();  // 从 0 开始编号的并行执行线程
        cudaSetDevice(tid);
        std::cout << "Thread " << tid << " is running on device " << tid << std::endl;

        // 将 l m n C NX 数据从cpu搬到GPU上
        Complex *l;
        Complex *m;
        Complex *n;
        Complex *C;
        // 分配内存空间
        CHECK(cudaMalloc(&l, lmnC_index * sizeof(Complex)));
        CHECK(cudaMalloc(&m, lmnC_index * sizeof(Complex)));
        CHECK(cudaMalloc(&n, lmnC_index * sizeof(Complex)));
        CHECK(cudaMalloc(&C, lmnC_index * sizeof(Complex)));
        
        // 将已有的部分变量复制到GPU上
        CHECK(cudaMemcpy(l, cl, lmnC_index * sizeof(Complex), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(m, cm, lmnC_index * sizeof(Complex), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(n, cn, lmnC_index * sizeof(Complex), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(C, C_host, lmnC_index * sizeof(Complex), cudaMemcpyHostToDevice));
        
        // 创建用来存储不同index中【u, v, w】
        Complex *u, *cu;
        Complex *v, *cv;
        Complex *w, *cw;
        // 将 u v w 复制到GPU上
        CHECK(cudaMalloc(&u, uvw_size * sizeof(Complex)));
        CHECK(cudaMalloc(&v, uvw_size * sizeof(Complex)));
        CHECK(cudaMalloc(&w, uvw_size * sizeof(Complex)));
        cu = (Complex *) malloc(uvw_size * sizeof(Complex));
        cv = (Complex *) malloc(uvw_size * sizeof(Complex));
        cw = (Complex *) malloc(uvw_size * sizeof(Complex));

        // 创建存储uvw坐标对应的频次
        Complex *uvwFrequencyMap, *uvwMapVector;
        CHECK(cudaMalloc(&uvwFrequencyMap, uvw_size * sizeof(Complex)));
        uvwMapVector = (Complex *) malloc(uvw_size * sizeof(Complex));

        // 存储计算后的到的最终结果
        Complex *F;
        CHECK(cudaMalloc(&F, lmnC_index*sizeof(Complex)));

        // 遍历所有开启的线程处理， 一个线程控制一个GPU 处理一个id*amount/total的块
        for (int p = tid + id*amount/totalnode; p < (id + 1) * amount / totalnode; p += nDevices) 
        // for (int p = tid + 10; p < (id + 1) * amount / totalnode; p += nDevices) 
        {
            cout << "for loop: " << p+1 << endl;

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

            int uvw_index;
            // 创建一个临界区，保证只有一个线程进入，用于构建u v w
            #pragma omp critical
            {
                string address_uvw = address + "uvw" + to_string(p+1) + duration + sufix;
                cout << "address_uvw: " << address_uvw << std::endl;
                
                ifstream uvwFile(address_uvw);

                // 同时用一个向量保存每一个uvw坐标点的frequency
                uvw_index = 0;
                double u_point, v_point, w_point;
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
                            uvwMapVector[uvw_index]._real = 1 / (it->second);  // 存储频次的倒数
                        } else {
                            uvwMapVector[uvw_index]._real = 1; 
                        }

                        // cu, cv, cw 需要存储原始坐标
                        cu[uvw_index]._real = u_point;
                        cv[uvw_index]._real = v_point;
                        cw[uvw_index]._real = w_point;

                        uvw_index++;
                    }
                }               
                cout << "uvw_index: " << uvw_index << endl; 


                // 保存uvwMap
                // if(p==0){
                //     std::ofstream file("map1period.txt");
                //     // 检查文件是否成功打开
                //     if (file.is_open()) {
                //         // 使用迭代器遍历vector并写入到文件中
                //         for (int i=0; i<uvw_index; ++i) {
                //             file << uvwMapVector[i]._real << endl;  // 将每个元素写入文件，每个元素后跟一个换行符
                //         }
                //         file.close();  // 关闭文件流
                //     }
                // }
                
                // 复制到GPU上
                CHECK(cudaMemcpy(u, cu, uvw_index * sizeof(Complex), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(v, cv, uvw_index * sizeof(Complex), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(w, cw, uvw_index * sizeof(Complex), cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(uvwFrequencyMap, uvwMapVector, uvw_index * sizeof(Complex), cudaMemcpyHostToDevice));
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
            Complex *viss;
            CHECK(cudaMalloc(&viss, uvw_index * sizeof(Complex)));

            int blockSize;
            int minGridSize; // 最小网格大小
            int gridSize;    // 实际网格大小

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, visscal, 0, 0);
            gridSize = floor(uvw_index + blockSize - 1) / blockSize;
            cout << "Viss Computing, blockSize: " << blockSize << endl;
            cout << "Viss Computing, girdSize: " << gridSize << endl;
            printf("Viss Computing... Here is gpu %d running process %d on node %d\n", omp_get_thread_num(), p+1, id);
            // 调用函数计算可见度
            visscal<<<gridSize, blockSize>>>(RES, uvw_index, lmnC_index, viss, u, v, w, l, m, n, C);
            // 进行线程同步
            CHECK(cudaDeviceSynchronize());
            if(p==0){
                // 转移到主机上
                Complex *hostViss = new Complex[uvw_index];
                // 从GPU复制到主机
                cudaMemcpy(hostViss, viss, uvw_index * sizeof(Complex), cudaMemcpyDeviceToHost);

                std::ofstream file("viss_original.txt");
                // 检查文件是否成功打开
                if (file.is_open()) {
                    // 使用迭代器遍历vector并写入到文件中
                    for (int i=0; i<uvw_index; ++i) {
                        file << hostViss[i]._real << endl;  // 将每个元素写入文件，每个元素后跟一个换行符
                    }
                    file.close();  // 关闭文件流
                }
            }
            cout << "period" << p+1 << "viss compute success!" << endl;

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
            // 调用image_recon函数计算图像反演
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, imagerecon, 0, 0);
            gridSize = floor(lmnC_index + blockSize - 1) / blockSize;
            cout << "Image Reconstruction, blockSize: " << blockSize << endl;
            cout << "Image Reconstruction, girdSize: " << gridSize << endl;
            printf("Image Reconstruction... Here is gpu %d running process %d on node %d\n",omp_get_thread_num(),p+1,id);
            imagerecon<<<gridSize,blockSize>>>(RES, uvw_index, lmnC_index, F, viss, 
                                                u, v, w, l, m, n, C, uvwFrequencyMap);
            // 进行线程同步
            CHECK(cudaDeviceSynchronize());

            if(p==0){
                // 转移到主机上
                Complex *hostF = new Complex[lmnC_index];
                // 从GPU复制到主机
                cudaMemcpy(hostF, F, lmnC_index * sizeof(Complex), cudaMemcpyDeviceToHost);

                std::ofstream file("F_original.txt");
                // 检查文件是否成功打开
                if (file.is_open()) {
                    // 使用迭代器遍历vector并写入到文件中
                    for (int i=0; i<uvw_index; ++i) {
                        file << hostF[i]._real << endl;  // 将每个元素写入文件，每个元素后跟一个换行符
                    }
                    file.close();  // 关闭文件流
                }
            }
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


            // // 记录save imagerecon开始事件
            // cudaEvent_t saveimagestart, saveimagestop;
            // cudaEventCreate(&saveimagestart);
            // cudaEventCreate(&saveimagestop);
            // cudaEventRecord(saveimagestart);
            // // 创建一个临界区，用于保存图像反演结果
            #pragma omp critical
            {   
                // 在CPU上创建变量保存F结果
                Complex *tempF;
                tempF = (Complex*)malloc(lmnC_index*sizeof(Complex));

                double *extendF;
                extendF = (double*)malloc(NX_index*sizeof(double));

                CHECK(cudaMemcpy(tempF, F, lmnC_index*sizeof(Complex), cudaMemcpyDeviceToHost));

                std::ofstream ExFile;
                string para = "Ex";
                string address_Ex = F_address + para + to_string(p+1) + duration + sufix;
                cout << "address_Ex: " << address_Ex << endl;

                ExFile.open(address_Ex);
                if (!ExFile.is_open()) {
                    std::cerr << "Error opening file: " << address_Ex << endl;
                }
                for (int c = 0; c < NX_index; c++) {
                    int tmp = static_cast<int>(cNX[c]._real) - 1;  // matlab中下标从1开始
                    extendF[c] = tempF[tmp]._real;
                    ExFile << extendF[c] << std::endl;
                }

                ExFile.close();
                free(extendF);
                free(tempF);
            }

            // // 记录save imagerecon结束事件
            // cudaEventRecord(saveimagestop);
            // cudaEventSynchronize(saveimagestop);
            // // 计算经过的时间
            // float saveimageMS = 0;
            // cudaEventElapsedTime(&saveimageMS, saveimagestart, saveimagestop);
            // printf("Period %d Save Image Cost Time is: %f s\n", p+1, saveimageMS/1000);
            // // 销毁事件
            // cudaEventDestroy(saveimagestart);
            // cudaEventDestroy(saveimagestop);


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

            CHECK(cudaFree(viss));

        }

        // 清除cuda缓存
        CHECK(cudaFree(u));
        CHECK(cudaFree(v));
        CHECK(cudaFree(w));
        CHECK(cudaFree(l));
        CHECK(cudaFree(m));
        CHECK(cudaFree(n));
        CHECK(cudaFree(C));
        CHECK(cudaFree(F));
        
        free(cu);
        free(cv);
        free(cw);
    }
    
    free(cl);
    free(cm);
    free(cn);
    free(C_host);
    
    gettimeofday(&finish, NULL);
    total_time = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_usec - start.tv_usec)) / 1000000.0;
    cout << "total time: " << total_time << "s" << endl;
    return 0;
}


int main()
{
    vissGen(0, 1, 2094);
}

