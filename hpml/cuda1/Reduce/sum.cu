#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define M 1000
#define N 1000

__global__ void sum(const float *A, float *B, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float localSum = 0.0f;

    for (int col = tid; col < n; col += BLOCK_SIZE) {
        localSum += A[row * n + col];
    }

    __shared__ float shared_A[BLOCK_SIZE];
    shared_A[tid] = localSum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared_A[tid] += shared_A[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        B[row] = shared_A[0];
    }
}

int main() {
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = M * sizeof(float);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);

    for (int i = 0; i < M * N; i++) {
        h_A[i] = 1.0f;
    }

    float *d_A, *d_B;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sum<<<grid, block>>>(d_A, d_B, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_B, d_B, sizeB, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < M; i++) {
        if (h_B[i] != (float)N) {
            printf("Mismatch at row %d: got %f, expected %f\n", i, h_B[i], (float)N);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Result correct!\n");
    }

    printf("Kernel execution time: %f ms\n", milliseconds);

    float bandwidth = (float)sizeA / 1e9f / (milliseconds / 1000.0f);
    printf("Effective bandwidth: %f GB/s\n", bandwidth);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}