#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Custom CUDA Unpacking Kernel
__global__ void unpackWeightsKernel(const uint8_t* d_packed, float* d_unpacked, int numPackedElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPackedElements) {
        // Fetch a single 8-bit byte containing TWO 4-bit weights from VRAM
        uint8_t packedByte = d_packed[idx];

        // Bitwise Operation 1: Extract the upper 4 bits (Shift right by 4)
        uint8_t upper4 = packedByte >> 4;

        // Bitwise Operation 2: Extract the lower 4 bits (Mask out the upper bits using 0x0F)
        uint8_t lower4 = packedByte & 0x0F;

        // Convert the raw integer weights into standard floating point formats inside registers
        // In real LLMs, we would multiply these by a scale factor
        d_unpacked[idx * 2]     = static_cast<float>(upper4);
        d_unpacked[idx * 2 + 1] = static_cast<float>(lower4);
    }
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " SUB-BYTE INT4 UNPACKING ENGINE" << std::endl;
    std::cout << "====================================================" << std::endl;

    const int numPackedElements = 8; // 8 bytes total
    const int numWeights = numPackedElements * 2; // Holds 16 distinct 4-bit weights

    // Host buffer representing a fake uncompressed weights array (Values from 0 to 15)
    std::vector<uint8_t> rawWeights = {12, 5, 8, 14, 0, 15, 3, 7, 2, 11, 4, 9, 1, 13, 6, 10};
    
    // Allocate Host storage buffer for the compressed data layout
    std::vector<uint8_t> h_packed(numPackedElements);

    // Pack pairs of 4-bit weights into individual 8-bit bytes on the Host
    for (int i = 0; i < numPackedElements; ++i) {
        uint8_t upper = rawWeights[i * 2];
        uint8_t lower = rawWeights[i * 2 + 1];
        // Shift upper value into place and combine using a bitwise OR (|)
        h_packed[i] = (upper << 4) | (lower & 0x0F);
    }

    std::cout << " Host-side 4-bit packing complete." << std::endl;

    // Allocate Device memory buffers
    uint8_t* d_packed = nullptr;
    float* d_unpacked = nullptr;
    CHECK_CUDA(cudaMalloc(&d_packed, numPackedElements * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_unpacked, numWeights * sizeof(float)));

    // Ship packed data over to the GPU
    CHECK_CUDA(cudaMemcpy(d_packed, h_packed.data(), numPackedElements * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Launch single-block unpacking routine
    unpackWeightsKernel<<<1, numPackedElements>>>(d_packed, d_unpacked, numPackedElements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Pull unpacked floating-point results back to CPU memory
    std::vector<float> h_unpacked(numWeights);
    CHECK_CUDA(cudaMemcpy(h_unpacked.data(), d_unpacked, numWeights * sizeof(float), cudaMemcpyDeviceToHost));

    // Print verification comparison
    std::cout << "\nVerification Report card:" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Original | Packed Byte (Hex) | Unpacked Output" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int i = 0; i < numPackedElements; ++i) {
        std::cout << " " << (int)rawWeights[i*2] << ", " << (int)rawWeights[i*2+1]
                  << "   |      0x" << std::hex << std::uppercase << (int)h_packed[i] << std::dec
                  << "        |   " << h_unpacked[i*2] << ", " << h_unpacked[i*2+1] << std::endl;
    }
    std::cout << "---------------------------------------" << std::endl;

    // Free resources
    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_unpacked));

    return 0;
}