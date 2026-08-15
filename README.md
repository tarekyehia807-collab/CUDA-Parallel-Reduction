It's a CUDA C++ implementation of a parallel sum reduction kernel using shared memory and warp-level primitives.
The project focuses on understanding and profiling CUDA reduction performance rather than using a high-level reduction library.
## Techniques Used
- CUDA C++
- Parallel reduction
- Shared memory
- Warp shuffle (`__shfl_down_sync`)
- `__forceinline__`
- `#pragma unroll`
- Pinned host memory (`cudaMallocHost`)
- Asynchronous memory transfers (`cudaMemcpyAsync`)
- CUDA Streams
- CUDA Events
- CUDA Occupancy API
- NVIDIA Nsight Compute
- Performance benchmarking

Each thread processes multiple elements from the input array.

The reduction is performed in two stages:

1. Block-level reduction using shared memory.
2. Final warp-level reduction using `__shfl_down_sync`.

The kernel produces one partial sum per CUDA block.

The partial sums are then copied back to the host and accumulated on the CPU.

## Benchmark Configuration

Input size:

1 << 20 = 1,048,576 floats

Input size in memory:

4 MB

The input data is generated on the CPU using:

- `std::mt19937`
- `std::uniform_real_distribution<float>`

Pinned host memory is used for the input and output buffers.

## GPU

NVIDIA GeForce RTX 4050 Laptop GPU

## Block Size Experiments

Four block sizes were tested:

128, 256, 512, 1024 threads per block.

Results:

Block Size: 128
Occupancy: 100%
Average Kernel Time: 0.0129434 ms
Memory Throughput: 152.02 GB/s
DRAM Throughput: 90.69%
Compute Throughput: 17.01%
L1/TEX Hit Rate: 1.39%
L2 Hit Rate: 3.38%

Block Size: 256
Occupancy: 100%
Average Kernel Time: 0.0191693 ms
Memory Throughput: 151.62 GB/s
DRAM Throughput: approximately 90%
Compute Throughput: approximately 20-29%
L1/TEX Hit Rate: 0.71%
L2 Hit Rate: 2.10%

Block Size: 512
Occupancy: 100%
Average Kernel Time: 0.0157286 ms
Memory Throughput: 154.25 GB/s
DRAM Throughput: 92.09%
Compute Throughput: 24.89%
L1/TEX Hit Rate: 0.35%
L2 Hit Rate: 0.95%

Block Size: 1024
Occupancy: 66.67%
Average Kernel Time: 0.0267674 ms
Memory Throughput: 140.70 GB/s
DRAM Throughput: 73.56%
Compute Throughput: 28.65%
L1/TEX Hit Rate: 0.15%
L2 Hit Rate: 0.79%

## Best Configuration

The best measured configuration was:

Block Size: 128 threads

Occupancy: 100%

Average Kernel Time: ~12.94 microseconds

Memory Throughput: 152.02 GB/s

DRAM Throughput: 90.69%

Compute Throughput: 17.01%

This configuration was faster than the other tested block sizes despite not achieving the highest memory throughput.

## Performance Comparison

128 threads:
~12.94 us

256 threads:
~19.17 us

512 threads:
~15.73 us

1024 threads:
~26.77 us

The 128-thread configuration was approximately 17.7% faster than the 512-thread configuration and approximately 32.5% faster than the 256-thread configuration.

The 1024-thread configuration was the slowest configuration tested.

## Profiling Observations

Nsight Compute showed that the reduction kernel is primarily memory-bound rather than compute-bound.

The measured memory throughput was consistently much higher than compute throughput.

For the best configuration:

Memory Throughput: 90.69%

Compute Throughput: 17.01%

DRAM Throughput: 90.69%

The low L1 and L2 cache hit rates are expected for this workload because the reduction performs mostly streaming accesses over a relatively large input array with limited data reuse.

Occupancy alone does not determine kernel performance.

Both the 128-thread and 512-thread configurations achieved 100% occupancy, but the 128-thread configuration was faster.

Similarly, the 512-thread configuration achieved slightly higher memory throughput than the 128-thread configuration, but it was still slower.

Therefore, kernel performance was evaluated using actual execution time together with Nsight Compute metrics rather than optimizing for a single metric.

## Benchmarking Method

CUDA Events were used to measure GPU kernel execution time.

A warm-up kernel launch was performed before benchmarking.

The kernel was executed for 100 iterations.

The total execution time was measured using CUDA Events:

cudaEventRecord(start, stream);

for (int i = 0; i < iterations; i++)
{
    reductionkernel<<<...>>>(...);
}

cudaEventRecord(stop, stream);

The average kernel time was calculated as:

Average Time = Total Time / Number of Iterations


This project is primarily a learning and performance-analysis project focused on understanding how CUDA kernels behave at the hardware level.
