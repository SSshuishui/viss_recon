#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <string>

using namespace std;

int main() {
    string address = "frequency_30M/"; // 替换为文件所在的路径
    int p = 1; // 示例参数
    string duration = "frequency30M"; // 根据你的文件名格式
    string sufix = ".txt"; // 文件后缀

    string address_uvw = address + "updated_uvw" + to_string(p+1) + duration + sufix;
    cout << "address_uvw: " << address_uvw << endl;

    ifstream uvwFile(address_uvw);
    if (!uvwFile.is_open()) {
        cerr << "Error: Unable to open file " << address_uvw << endl;
        return 1;
    }

    vector<complex<float>> cu, cv, cw;
    vector<complex<float>> uvwMapVector;
    int uvw_index = 0;
    float u_point, v_point, w_point, freq_point;

    while (uvwFile >> u_point >> v_point >> w_point >> freq_point) {
        if (freq_point == 0) {
            cerr << "Error: Frequency point is zero, which is not allowed." << endl;
            continue; // 跳过这一行，因为频率不能为零
        }

        cu.push_back(complex<float>(u_point, 0));
        cv.push_back(complex<float>(v_point, 0));
        cw.push_back(complex<float>(w_point, 0));
        uvwMapVector.push_back(complex<float>(1 / freq_point, 0));
        uvw_index++;
    }

    if (uvw_index == 0) {
        cout << "No data read from the file." << endl;
    } else {
        cout << "Successfully read " << uvw_index << " data points." << endl;
    }

    uvwFile.close();

    // 打印测试确保是正确的
    int count = 0;
    int numElementsToPrint = 6; // 设定要打印的元素数量
    for (int i=0; i<numElementsToPrint; i++) {
        std::cout << cu[i] << " " << cv[i] << " " << cw[i] << ": " << uvwMapVector[i] << std::endl;
        if (++count == numElementsToPrint) {
            break;
        }
    }

    return 0;
}