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

// ============================================================================
// CUDA KERNEL: Block-Scaled INT4 Matrix-Vector Multiplication (GEMV)
// ============================================================================
__global__ void scaledQuantGEMVKernel(const uint8_t* d_packed_weights, const float* d_scales, const float* d_x, float* d_y, int rows, int cols, int block_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float accum = 0.0f;
        int num_blocks = cols / block_size;
        int packed_per_block = block_size / 2; // 2 weights per byte

        // Iterate through each scaling block
        for (int b = 0; b < num_blocks; ++b) {
            // Fetch the scale factor for this specific 32-weight block
            float scale = d_scales[row * num_blocks + b];

            // Iterate through the packed bytes within this block
            for (int i = 0; i < packed_per_block; ++i) {
                int col_idx = b * block_size + i * 2;
                uint8_t packed_byte = d_packed_weights[row * (cols / 2) + (b * packed_per_block) + i];

                // Unpack, apply the -8 offset for signed values, and scale
                float w1 = (static_cast<float>(packed_byte >> 4) - 8.0f) * scale;
                float w2 = (static_cast<float>(packed_byte & 0x0F) - 8.0f) * scale;

                // Multiply by the input vector activations and accumulate
                accum += w1 * d_x[col_idx];
                accum += w2 * d_x[col_idx + 1];
            }
        }
        // Write the finalized dot-product to VRAM
        d_y[row] = accum;
    }
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " EXECUTING BLOCK-SCALED INT4 INFERENCE KERNEL" << std::endl;
    std::cout << "====================================================" << std::endl;

    const int rows = 512;
    const int cols = 512;
    const int block_size = 32; // Industry standard group size
    const int num_blocks_per_row = cols / block_size;
    const int packed_cols = cols / 2; 

    std::cout << "Matrix Dimensions: [" << rows << " x " << cols << "]" << std::endl;
    std::cout << "Quantization Block Size: " << block_size << " weights per scale factor" << std::endl;

    // 1. Allocate Host Memory
    std::vector<uint8_t> h_packed_weights(rows * packed_cols);
    std::vector<float> h_scales(rows * num_blocks_per_row);
    std::vector<float> h_x(cols, 1.0f); // Input vector filled with 1.0
    std::vector<float> h_y(rows, 0.0f);

    // 2. Deterministic Data Generation (Simulating Q4_0 Packing)
    // We will simulate weights of 3.0 and -2.0, with a scale factor of 2.0.
    // Real weights = (3.0 * 2.0) = 6.0, and (-2.0 * 2.0) = -4.0.
    uint8_t packed_val = ((3 + 8) << 4) | ((-2 + 8) & 0x0F); // 0xB6

    for (int r = 0; r < rows; ++r) {
        // Fill scales
        for (int b = 0; b < num_blocks_per_row; ++b) {
            h_scales[r * num_blocks_per_row + b] = 2.0f;
        }
        // Fill packed weights
        for (int c = 0; c < packed_cols; ++c) {
            h_packed_weights[r * packed_cols + c] = packed_val;
        }
    }

    // 3. Allocate Device Memory
    uint8_t* d_packed_weights = nullptr;
    float* d_scales = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_packed_weights, rows * packed_cols * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_scales, rows * num_blocks_per_row * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * sizeof(float)));

    // 4. Transfer to GPU
    CHECK_CUDA(cudaMemcpy(d_packed_weights, h_packed_weights.data(), rows * packed_cols * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales, h_scales.data(), rows * num_blocks_per_row * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "\nDispatching to GPU with Block-Scaling Math..." << std::endl;
    scaledQuantGEMVKernel<<<blocksPerGrid, threadsPerBlock>>>(d_packed_weights, d_scales, d_x, d_y, rows, cols, block_size);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. Retrieve Results
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // 7. Rigorous Math Validation
    // Math logic: Every byte holds (3 * 2.0 = 6.0) and (-2 * 2.0 = -4.0). 
    // Sum per byte = 2.0. There are 256 bytes per row. 
    // Expected output per row = 256 * 2.0 = 512.0.
    float expected_val = 512.0f;
    bool valid = true;
    for (int i = 0; i < rows; ++i) {
        if (std::fabs(h_y[i] - expected_val) > 1e-4) {
            valid = false;
            std::cout << "Math failure at row " << i << ": Got " << h_y[i] << ", Expected " << expected_val << std::endl;
            break;
        }
    }

    if (valid) {
        std::cout << "==> VALIDATION SUCCESS: Signed offset and block scaling calculated exactly " << expected_val << " for all outputs." << std::endl;
    } else {
        std::cerr << "==> VALIDATION FAILED: Register bit-drift detected." << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_packed_weights));
    CHECK_CUDA(cudaFree(d_scales));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}