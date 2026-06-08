# Block-Scaled INT4 Quantization Kernel in CUDA

A standalone CUDA implementation of a block-scaled INT4 matrix-vector multiplication (GEMV) kernel. This project serves as a technical demonstration of low-level memory bandwidth optimization techniques commonly used in LLM inference engines (such as the `Q4_0` format in `llama.cpp`).

## Objective
Modern AI inference is heavily bottlenecked by VRAM memory bandwidth. Compressing 32-bit floating-point weights down to 4-bit integers reduces the memory footprint by 87.5%, but requires on-the-fly register decompression during execution. This project implements that decompression layer directly in custom CUDA silicon logic.

## Technical Highlights
* **Bit-Level Manipulation:** Uses bitwise right-shifts (`>> 4`) and bitwise AND masks (`& 0x0F`) to unpack two 4-bit weights out of a single cached 8-bit byte.
* **Block Scaling (Group Quantization):** Implements a quantization block size of 32 weights per scale factor to minimize precision loss while maintaining low memory overhead.
* **Asymmetric Signed Range:** Maps unsigned 4-bit integers (0 to 15) to a signed range (-8 to +7) using a runtime register offset to support negative weight values.

## Memory Footprint Reduction
For a standard $512 \times 512$ matrix layer:
* **Uncompressed Matrix (FP32):** 1,024 KB
* **Compressed Matrix (INT4 + Scales):** 128 KB (Weights) + 32 KB (Scales) = 160 KB
* **Total VRAM Footprint Saved:** ~84.3% reduction for the weight matrix data.

## File Structure
* `block_scaled_gemv.cu`: Contains the host memory management, synthetic data generator, CUDA kernel, and host-side mathematical verification loop.

## Hardware Requirements & Compilation
Compiled and tested on an **Nvidia RTX 4060 Laptop GPU** using CUDA Toolkit.

To compile and execute the benchmark:
```powershell
nvcc -O3 -arch=sm_89 block_scaled_gemv.cu -o block_scaled_gemv.exe
.\block_scaled_gemv.exe
```
##Verification Output
The kernel includes a host-side math verification check to ensure bit-alignment and scaling match expected values:

```
====================================================
 EXECUTING BLOCK-SCALED INT4 INFERENCE KERNEL
====================================================
Matrix Dimensions: [512 x 512]
Quantization Block Size: 32 weights per scale factor

Dispatching to GPU with Block-Scaling Math...
==> VALIDATION SUCCESS: Signed offset and block scaling calculated exactly 512 for all outputs.
```
