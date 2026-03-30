// Includes
#include <stdio.h>
#include "timer.h"

// Variables for host and device vectors.
float* h_A; 
float* h_B; 
float* h_C; 
float* d_A; 
float* d_B; 
float* d_C; 

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

__global__ void AddVectors1(const float* A, const float* B, float* C, int N);
__global__ void AddVectors2(const float* A, const float* B, float* C, int N);
__global__ void AddVectors3(const float* A, const float* B, float* C, int N);


// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
    int K; 
    int S; // Scenario
    int N; //Vector size

	// Parse arguments.
    if(argc != 3){
      printf("Usage: %s <scenario number> <K>\n", argv[0]);
      exit(0);
    } else {
      sscanf(argv[1], "%d", &K);
      sscanf(argv[2], "%d", &S);
    }      

    N = K * 1000000;
    printf("Total vector size: %d\n", N); 
    // size_t is the total number of bytes for a vector.
    size_t size = N * sizeof(float);
    
    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup(false);
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup(false);
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup(false);

    // Allocate vectors in device memory.
    cudaError_t error;
    error = cudaMalloc((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);

    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     h_A[i] = (float)i;
     h_B[i] = (float)(N-i);   
    }

    // Copy host vectors h_A and h_B to device vectores d_A and d_B
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);

    // Tell CUDA how big to make the grid and thread blocks.
    // Since this is a vector addition problem,
    // grid and thread block are both one-dimensional.
    int GridWidth, BlockWidth;
    if (S == 1) {
        GridWidth = 1;
        BlockWidth = 1;
    } else if (S == 2) {
        GridWidth = 1;
        BlockWidth = 256;
    } else if (S == 3) {
        GridWidth = (N + 255) / 256;
        BlockWidth = 256;
    } else {
        printf("Invalid scenario number. Must be 1, 2, or 3.\n");
        Cleanup(false);
    }

    dim3 dimGrid(GridWidth);                    
    dim3 dimBlock(BlockWidth);             

    // Warm up
    if (S == 1) {
        AddVectors1<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    } else if (S == 2) {
        AddVectors2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    } else if (S == 3) {
        AddVectors3<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // Invoke kernel
    if (S == 1) {
        AddVectors1<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    } else if (S == 2) {
        AddVectors2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    } else if (S == 3) {
        AddVectors3<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();

	// Report timing data.
    printf( "Time: %lf (sec)\n", time);
     
    // Copy result from device memory to host memory
    error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) Cleanup(false);

    // Verify & report result
    for (i = 0; i < N; ++i) {
        float val = h_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");

    // Clean up and exit.
    Cleanup(true);
}


__global__ void AddVectors1(const float* A, const float* B, float* C, int N)
{
    for( int i=0; i<N; ++i ){
        C[i] = A[i] + B[i];
    }
}

__global__ void AddVectors2(const float* A, const float* B, float* C, int N)
{
    int index = threadIdx.x;

    for( int i=index; i<N; i+=blockDim.x ){
        C[i] = A[i] + B[i];
    }
}

__global__ void AddVectors3(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i]; 
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
    error = cudaDeviceReset();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}


