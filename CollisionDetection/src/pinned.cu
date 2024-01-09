// for testing pinned memory performance, 2 ways to use pinned memory
// nvcc -o main src/pinned.cu && ./main
// https://github.com/ledatelescope/bifrost/blob/3c68028ebd55651522c4de3862d5979c16b209e5/src/fft.cu
// Regular host_vector sorting time: 0.00257205 seconds
// Pinned host_vector sorting time:  0.00091251 seconds
// Pinned host_vector sorting time:  0.00100729 second


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory.h> // thrust::system::cuda::universal_host_pinned_memory_resource;
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <chrono>
#include <iostream>

int main() {
    const int size = 1000000;

    // 不使用pinned memory的host_vector
    thrust::host_vector<double> regular_host_vector(size);

    // 使用pinned memory的host_vector
    thrust::host_vector<double, thrust::cuda::experimental::pinned_allocator<double>> pinned_host_vector;
    pinned_host_vector.resize(size);
    thrust::host_vector<double, thrust::mr::stateless_resource_allocator<double, thrust::universal_host_pinned_memory_resource>> _pinned_host_vector(size);
    thrust::device_vector<double> device_vector(size);

    // 生成随机数据
    thrust::generate(regular_host_vector.begin(), regular_host_vector.end(), rand);
    // thrust::generate(pinned_host_vector.begin(), pinned_host_vector.end(), rand);
    thrust::copy(regular_host_vector.begin(), regular_host_vector.end(), pinned_host_vector.begin());
    thrust::copy(regular_host_vector.begin(), regular_host_vector.end(), _pinned_host_vector.begin());

    // 测试不使用pinned memory的性能
    auto start_regular = std::chrono::high_resolution_clock::now();
    // thrust::sort(regular_host_vector.begin(), regular_host_vector.end());
    device_vector = regular_host_vector;
    regular_host_vector = device_vector;
    auto end_regular = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_regular = end_regular - start_regular;
    std::cout << "Regular host_vector sorting time: " << duration_regular.count() << " seconds" << std::endl;

    // 测试使用pinned memory的性能
    auto start_pinned = std::chrono::high_resolution_clock::now();
    // thrust::sort(pinned_host_vector.begin(), pinned_host_vector.end());
    device_vector = pinned_host_vector;
    pinned_host_vector = device_vector;
    auto end_pinned = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pinned = end_pinned - start_pinned;
    std::cout << "Pinned host_vector sorting time:  " << duration_pinned.count() << " seconds" << std::endl;

    // 测试使用pinned memory的性能
    auto start_pinned_ = std::chrono::high_resolution_clock::now();
    // thrust::sort(pinned_host_vector.begin(), pinned_host_vector.end());
    device_vector = _pinned_host_vector;
    _pinned_host_vector = device_vector;
    auto end_pinned_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_pinned_ = end_pinned_ - start_pinned_;
    std::cout << "Pinned host_vector sorting time:  " << duration_pinned_.count() << " seconds" << std::endl;

    return 0;
}
