#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Custom Fused Unpack + Multiply-Accumulate GPU Kernel
__global__ void quantGEMVKernel(const uint8_t* d_packed_weights, const float* d_x, float* d_y, int rows, int packed_cols) {
    // Each thread calculates exactly ONE row of the output vector
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float accum = 0.0f;

        // Loop through the packed columns (each iteration handles 2 weights)
        for (int c = 0; c < packed_cols; ++c) {
            // Fetch the packed byte from VRAM
            uint8_t packed_byte = d_packed_weights[row * packed_cols + c];

            // Unpack both 4-bit weights into registers
            float w1 = static_cast<float>(packed_byte >> 4);
            float w2 = static_cast<float>(packed_byte & 0x0F);

            // Fetch the corresponding activations from Vector X
            float x1 = d_x[c * 2];
            float x2 = d_x[c * 2 + 1];

            // Multiply and accumulate
            accum += w1 * x1 + w2 * x2;
        }

        // Write final result to the output vector
        d_y[row] = accum;
    }
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " LAUNCHING QUANTIZED LINEAR LAYER (GEMV)" << std::endl;
    std::cout << "====================================================" << std::endl;

    const int rows = 512;        // Number of output features
    const int cols = 512;        // Number of input features
    const int packed_cols = cols / 2; // Compressed matrix columns (2 weights per byte)

    std::cout << "Weight Matrix Dimensions: [" << rows << " x " << cols << "]" << std::endl;
    std::cout << "Uncompressed Weight Size: " << (rows * cols * sizeof(float)) / 1024 << " KB" << std::endl;
    std::cout << "Compressed INT4 Weight Size: " << (rows * packed_cols * sizeof(uint8_t)) / 1024 << " KB (50% VRAM Saved!)" << std::endl;

    // 1. Initialize Host Weights and pack them
    std::vector<uint8_t> h_packed_weights(rows * packed_cols);
    // Fill the weights matrix with a predictable pattern: upper weight = 2, lower weight = 3
    uint8_t static_pack = (2 << 4) | (3 & 0x0F); // 0x23 in Hex
    for (int i = 0; i < rows * packed_cols; ++i) {
        h_packed_weights[i] = static_pack;
    }

    // 2. Initialize Input Vector X (Fill with 1.0f)
    std::vector<float> h_x(cols, 1.0f);
    std::vector<float> h_y(rows, 0.0f); // Output vector buffer

    // 3. Allocate Device VRAM
    uint8_t* d_packed_weights = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_packed_weights, rows * packed_cols * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * sizeof(float)));

    // 4. Transfer inputs to RTX 4060 VRAM
    CHECK_CUDA(cudaMemcpy(d_packed_weights, h_packed_weights.data(), rows * packed_cols * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Execution Configuration
    // 512 threads total, so we launch 2 blocks of 256 threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "\nLaunching Quantized Execution Kernel..." << std::endl;
    quantGEMVKernel<<<blocksPerGrid, threadsPerBlock>>>(d_packed_weights, d_x, d_y, rows, packed_cols);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. Gather results back to CPU RAM
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // 7. Verification Mathematics
    // Every packed column has weights 2 and 3. Vector X is all 1.0s.
    // For each column step, accumulation = (2 * 1.0) + (3 * 1.0) = 5.
    // There are 'packed_cols' (256) steps. Total expected value = 256 * 5 = 1280.
    float expected_val = static_cast<float>(packed_cols) * (2.0f + 3.0f);
    bool execution_valid = true;
    for (int i = 0; i < rows; ++i) {
        if (std::fabs(h_y[i] - expected_val) > 1e-4) {
            execution_valid = false;
            std::cout << "Mismatch at index " << i << ": Found " << h_y[i] << ", Expected " << expected_val << std::endl;
            break;
        }
    }

    if (execution_valid) {
        std::cout << "==>  QUANTIZED LINEAR LAYER SUCCESSFUL!" << std::endl;
        std::cout << "    Calculated exact dot product (" << expected_val << ") across all " << rows << " vector outputs." << std::endl;
    } else {
        std::cerr << "==>  VERIFICATION ERROR: Bit alignment drift inside register decompression loop." << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_packed_weights));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}