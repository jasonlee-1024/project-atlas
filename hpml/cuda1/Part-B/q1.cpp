#include <iostream>
#include <cstdlib>      // for std::atoi
#include <chrono>       // for timing

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <integer>" << std::endl;
        return 1;
    }

    long long num = std::atoll(argv[1]);  // 用 atoll 防止大数溢出
    num = num * 1000000LL;                // 转为 long long

    // 分配数组
    float *arrA = new float[num];
    float *arrB = new float[num];

    // 初始化数组
    for (long long i = 0; i < num; i++) {
        arrA[i] = static_cast<float>(i);
        arrB[i] = static_cast<float>(i);
    }

    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    for (long long i = 0; i < num; i++) {
        arrA[i] = arrA[i] + arrB[i];
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration/1000.0 << " ms" << std::endl;

    // 释放数组
    delete[] arrA;
    delete[] arrB;

    return 0;
}