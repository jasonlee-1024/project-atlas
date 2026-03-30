#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define C 3
#define H 1024
#define W 1024
#define K 64
#define FH 3
#define FW 3
#define PAD 1

int main() {

    double* h_input  = (double*)malloc(sizeof(double) * 1 * C * H * W);
    double* h_filter = (double*)malloc(sizeof(double) * K * C * FH * FW);

    // I[c, x, y] = c * (x + y)
    for (int c = 0; c < C; c++)
        for (int x = 0; x < H; x++)
            for (int y = 0; y < W; y++)
                h_input[c*H*W + x*W + y] = (double)c * (x + y);

    // F[k, c, i, j] = (c + k) * (i + j)
    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int i = 0; i < FH; i++)
                for (int j = 0; j < FW; j++)
                    h_filter[k*C*FH*FW + c*FH*FW + i*FW + j] = (double)(c + k) * (i + j);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(input_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
        1, C, H, W);

    cudnnSetFilter4dDescriptor(filter_desc,
        CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
        K, C, FH, FW);

    cudnnSetConvolution2dDescriptor(conv_desc,
        PAD, PAD,
        1, 1,
        1, 1,
        CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);

    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(conv_desc,
        input_desc, filter_desc,
        &out_n, &out_c, &out_h, &out_w);

    cudnnSetTensor4dDescriptor(output_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
        out_n, out_c, out_h, out_w);

    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t algoPerf;

    cudnnFindConvolutionForwardAlgorithm(cudnn,
        input_desc, filter_desc, conv_desc, output_desc,
        1,
        &returnedAlgoCount,
        &algoPerf);

    cudnnConvolutionFwdAlgo_t algo = algoPerf.algo;
    printf("Selected algorithm: %d, time: %.3f ms\n", algo, algoPerf.time);

    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_desc, filter_desc, conv_desc, output_desc,
        algo, &workspace_size);

    void* d_workspace = nullptr;
    if (workspace_size > 0)
        cudaMalloc(&d_workspace, workspace_size);

    double *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input,  sizeof(double) * 1 * C * H * W);
    cudaMalloc(&d_filter, sizeof(double) * K * C * FH * FW);
    cudaMalloc(&d_output, sizeof(double) * out_n * out_c * out_h * out_w);

    cudaMemcpy(d_input,  h_input,  sizeof(double)*1*C*H*W,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(double)*K*C*FH*FW, cudaMemcpyHostToDevice);

    const double alpha = 1.0, beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudnnConvolutionForward(cudnn,
        &alpha,
        input_desc,  d_input,
        filter_desc, d_filter,
        conv_desc,   algo,
        d_workspace, workspace_size,
        &beta,
        output_desc, d_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("cuDNN convolution time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ── checksum ─────────────────────────────────────────
    double* h_output = (double*)malloc(sizeof(double) * out_n * out_c * out_h * out_w);
    cudaMemcpy(h_output, d_output, 
               sizeof(double) * out_n * out_c * out_h * out_w,
               cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (int k = 0; k < out_c; k++)
        for (int x = 0; x < out_h; x++)
            for (int y = 0; y < out_w; y++)
                checksum += h_output[k * out_h * out_w + x * out_w + y];

    printf("Checksum: %f\n", checksum);

    free(h_output);

    free(h_input);
    free(h_filter);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    if (d_workspace) cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}