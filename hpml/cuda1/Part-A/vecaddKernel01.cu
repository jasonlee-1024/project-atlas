__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int warpSize = 32;
    int totalWarps = blockDim.x * gridDim.x / warpSize;
    int totalData = blockDim.x * gridDim.x * N;
    int dataPerWarp = totalData / totalWarps;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = index / warpSize;
    int laneId = index % warpSize;

    int threadStartIndex = warpId * dataPerWarp + laneId;
    int threadEndIndex = threadStartIndex + dataPerWarp;

    int i;
    for( i=threadStartIndex; i<threadEndIndex; i=i+warpSize ){
        C[i] = A[i] + B[i];
    }

}
