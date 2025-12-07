#include "cub/cub.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"

#include "my_cuda.cuh"

CUDA_Test::CUDA_Test() {

}

void CUDA_Test::test() {
    
    std::cout << "Test" << "\n";
    
    const int num_items = 1000;
    int h_in[] = {8, 6, 7, 5, 3, 0, 9}; // Host input array

    // Device pointers
    int *d_in = nullptr;
    int *d_out = nullptr;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate device memory for input and output
    cudaMalloc(&d_in, sizeof(int) * num_items);
    cudaMalloc(&d_out, sizeof(int) * num_items);

    // Copy input data from host to device
    cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    // The first call to ExclusiveSum with d_temp_storage = NULL calculates the required temp_storage_bytes
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    
    return;
}





