// Includes
#include <stdio.h>
#include "timer.h"

// Variables for host and device vectors.
float* A; 
float* B; 
float* C; 


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
    
    // Allocate vectors in device memory.
    cudaError_t error;
    error = cudaMallocManaged((void**)&A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&C, size);
    if (error != cudaSuccess) Cleanup(false);

    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     A[i] = (float)i;
     B[i] = (float)(N-i);   
    }

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
        AddVectors1<<<dimGrid, dimBlock>>>(A, B, C, N);
    } else if (S == 2) {
        AddVectors2<<<dimGrid, dimBlock>>>(A, B, C, N);
    } else if (S == 3) {
        AddVectors3<<<dimGrid, dimBlock>>>(A, B, C, N);
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // Invoke kernel
    if (S == 1) {
        AddVectors1<<<dimGrid, dimBlock>>>(A, B, C, N);
    } else if (S == 2) {
        AddVectors2<<<dimGrid, dimBlock>>>(A, B, C, N);
    } else if (S == 3) {
        AddVectors3<<<dimGrid, dimBlock>>>(A, B, C, N);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();

	// Report timing data.
    printf( "Time: %lf (sec)\n", time);
     
    // Verify & report result
    for (i = 0; i < N; ++i) {
        float val = C[i];
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
        
    if (A)
        cudaFree(A);
    if (B)
        cudaFree(B);
    if (C)
        cudaFree(C);

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


