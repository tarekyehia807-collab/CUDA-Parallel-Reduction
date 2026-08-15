#include <iostream>
#include "CudaError.cuh"
#include <cuda_runtime.h>
#include <random>
#include <cstddef>

constexpr int blocksize = 128;

__device__ __forceinline__ float warpsum(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__global__ void reductionkernel(
    const float* __restrict__ idata,
    float* __restrict__ odata,
    size_t n)
{
    int tid = threadIdx.x;

    int i = blockIdx.x * (blocksize * 2) + threadIdx.x;

    extern __shared__ float sdata[];

    float sum = 0.0f;

    // Each thread processes multiple elements
    while (i < n)
    {
        sum += idata[i];

        if (i + blocksize < n)
        {
            sum += idata[i + blocksize];
        }

        i += blocksize * gridDim.x * 2;
    }

    sdata[tid] = sum;

    __syncthreads();

    // Block reduction
    if (blocksize >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = sum = sum + sdata[tid + 512];
        }

        __syncthreads();
    }

    if (blocksize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = sum = sum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blocksize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = sum = sum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blocksize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] = sum = sum + sdata[tid + 64];
        }

        __syncthreads();
    }

    // Final warp reduction
    if (tid < 32)
    {
        if (blocksize >= 64)
        {
            sum += sdata[tid + 32];
        }

        sum = warpsum(sum);
    }

    // One result per block
    if (tid == 0)
    {
        odata[blockIdx.x] = sum;
    }
}

int main()
{
    size_t n = 1 << 20;

    cudaStream_t stream1;
    cudaStream_t stream2;

    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEvent_t copyDone;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&copyDone));

    float* h_ptr;

    CUDA_CHECK(cudaMallocHost((void**)&h_ptr, n * sizeof(float)));

    std::mt19937 gen(12345);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t k = 0; k < n; k++)
    {
        h_ptr[k] = dist(gen);
    }

    float* d_ptr;

    CUDA_CHECK(cudaMalloc((void**)&d_ptr, n * sizeof(float)));

    int gridd = static_cast<int>((n + blocksize * 2 - 1) / (blocksize * 2));

    float* d_output;

    CUDA_CHECK(cudaMalloc((void**)&d_output, gridd * sizeof(float)));

    float* h_output;

    CUDA_CHECK(cudaMallocHost((void**)&h_output, gridd * sizeof(float)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_ptr, h_ptr, n * sizeof(float), cudaMemcpyHostToDevice, stream1));

    CUDA_CHECK(cudaEventRecord(copyDone, stream1));
    CUDA_CHECK(cudaStreamWaitEvent(stream2, copyDone, 0));

    // Warm-up run
    reductionkernel << <gridd, blocksize, blocksize * sizeof(float), stream2 >> > (d_ptr, d_output, n);

    int iterations = 100;
    float total_time = 0.0f;

    CUDA_CHECK(cudaEventRecord(start, stream2));

    for (int h = 0; h < iterations; h++) {
        reductionkernel << <gridd, blocksize, blocksize * sizeof(float), stream2 >> > (d_ptr, d_output, n);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, stream2));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
    float average_time = total_time / static_cast<float>(iterations);

    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, gridd * sizeof(float), cudaMemcpyDeviceToHost, stream2));

    CUDA_CHECK(cudaStreamSynchronize(stream2));

    float total = 0.0f;

    for (int i = 0; i < gridd; i++)
    {
        total += h_output[i];
    }

    int activeBlocksPerSM;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM, reductionkernel, blocksize, blocksize * sizeof(float)));
    int warpsperblock = blocksize / 32;
    int activewarps = activeBlocksPerSM * warpsperblock;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int MAXXWARPSPERSM = prop.maxThreadsPerMultiProcessor / 32;
    float occupancy = static_cast<float>(activewarps) / MAXXWARPSPERSM;

    std::cout << "Occupancy: " << occupancy * 100.0f << "%\n";
    std::cout << "Total kernel time: " << total_time << " ms\n";
    std::cout << "Average kernel time: " << average_time << " ms\n";

    CUDA_CHECK(cudaEventDestroy(copyDone));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_ptr));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_ptr));
    CUDA_CHECK(cudaFreeHost(h_output));
    return 0;
}