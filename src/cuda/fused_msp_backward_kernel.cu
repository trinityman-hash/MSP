// fused_msp_backward_kernel.cu
//
// Thermal-gated gradient kernel for the structural plugin's low-rank
// matrix A. Companion to msp/plugin_layer.py's `gate_gradients` (the
// CPU/GPU-portable reference implementation of the same policy) -- this
// is the "do it for real, in the kernel, so throttled work is actually
// skipped" version described in the spec's Subsystem A.
//
// -----------------------------------------------------------------------
// WHAT WAS FIXED VS. THE v3.0 SPEC
// -----------------------------------------------------------------------
// 1. The spec's kernel body was a "..." placeholder -- there was no actual
//    matrix computation to review or test. This file implements a real,
//    complete (if intentionally naive -- see "Known limitation" below)
//    kernel: dA[row, col] = sum_b grad_u[b, row] * x[b, col], gated by a
//    per-row thermal check performed BEFORE any memory is touched for a
//    frozen row, so throttled threads retire immediately instead of
//    computing a value that just gets discarded -- that in-kernel skip is
//    the actual point of doing this on-device rather than in Python.
//
// 2. The spec's launch code did not check for CUDA errors at all. Kernel
//    launches are asynchronous, and without an explicit error check
//    (cudaGetLastError for launch-configuration errors, plus checking the
//    result of a following synchronize/memcpy for execution errors), a
//    silently failing kernel looks identical to a correct one that
//    happened to write nothing. msp_launch_gradient_A_kernel() below
//    checks both.
//
// 3. The spec's build command asked to target "Apple Metal" and "Google
//    Tensor TPU VM IR" from the same .cu file compiled with nvcc. This is
//    not possible: CUDA is NVIDIA-only. A Metal target needs a Metal
//    Shading Language kernel; a mobile NPU target needs its own delegate
//    (e.g. via LiteRT/NNAPI) or a retargetable compiler stack (the
//    ONNX -> Apache TVM/LLVM pipeline described in the Apex-Edge v2 doc is
//    the architecturally sound way to do this -- see docs/ARCHITECTURE.md).
//    This file is explicitly the CUDA/NVIDIA backend only.
//
// -----------------------------------------------------------------------
// VALIDATION STATUS
// -----------------------------------------------------------------------
// The sandbox this project was built in has no NVIDIA GPU or nvcc
// toolchain, so this kernel could not be compiled or run there. It has
// since been validated on real GPU hardware (Colab T4):
//   1. Compiled with nvcc, -arch=sm_75, 2026-07-17 (exit 0; only
//      pre-existing, harmless -Wcomment warnings from this file's own
//      multi-line build-instructions comment below).
//   2. Numerically cross-checked, 2026-07-20, via
//      tests/cuda/validate_gradient_kernel.py (compiles a second time
//      through NVRTC/cupy, independent of step 1's nvcc build) against
//      msp.plugin_layer.StructuralPluginLayer's PyTorch autograd
//      gradient for a fixed input (batch=4, in_features=16, rank=4):
//      output matches with thermal throttling off, and with throttling
//      on (freeze_stride=2) the active rows match exactly while the
//      frozen rows are left provably untouched (checked via a sentinel
//      value, not just "close to zero").
// Both steps passed. See docs/STATUS.md's "CUDA validation on Google
// Colab" section for the exact recipe and captured output -- re-run it
// if this file changes, since a passing result only covers the exact
// code that was checked in at the time it was run.
//
// Known limitation: this is a naive one-thread-per-output-element kernel
// with a serial loop over the batch dimension -- correct, but not
// bandwidth-optimal. A production version would tile the reduction across
// threads in a block and use shared memory (or just call a cuBLAS/cuDNN
// GEMM and apply the thermal gate as a separate, cheap masking pass on
// the row indices before the GEMM, which is likely simpler to get right
// than a hand-fused kernel). This naive version is deliberately kept
// simple so it is easy to verify correctness against the PyTorch
// reference first; optimize only after that cross-check passes.

#include <cuda_runtime.h>
#include <cstdio>

extern "C" {

// grad_A[row, col] = sum_over_batch( grad_u[b, row] * x[b, col] )
//
// where grad_u = dL/du, u = x @ A^T (i.e. grad_u is what a prior kernel
// or host-side op already computed as dL/d(x @ A^T); this kernel does not
// compute that upstream gradient, only dL/dA from it).
//
// Thermal gating: when current_temp_c > critical_temp_c, only rows where
// (row % freeze_stride == 0) are computed; all other rows are left
// untouched in grad_A (callers should zero-initialize grad_A once per
// step if they need frozen rows to read as zero, matching the Python
// reference's masking behavior -- this kernel does not do that zeroing
// itself, to avoid a wasted full-buffer write on every launch when only a
// few rows changed).
__global__ void msp_gradient_A_thermal_gated_kernel(
    const float* __restrict__ x,        // [batch_size, in_features]
    const float* __restrict__ grad_u,   // [batch_size, rank]
    float* __restrict__ grad_A,         // [rank, in_features]  (output)
    int batch_size,
    int in_features,
    int rank,
    float current_temp_c,
    float critical_temp_c,
    int freeze_stride
) {
    int row = blockIdx.y;                                   // 0 .. rank-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;         // 0 .. in_features-1

    if (row >= rank || col >= in_features) {
        return;
    }

    bool throttling = current_temp_c > critical_temp_c;
    if (throttling && (freeze_stride <= 0 || row % freeze_stride != 0)) {
        // Frozen row: retire the thread now. No global memory touched for
        // this (row, col) at all -- this is the actual FLOP/power saving,
        // not just a post-hoc mask.
        return;
    }

    float acc = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        acc += grad_u[b * rank + row] * x[b * in_features + col];
    }
    grad_A[row * in_features + col] = acc;
}

} // extern "C"

// ---------------------------------------------------------------------
// Host-side launch wrapper with explicit error checking (see fix #2
// above). Returns cudaSuccess on success; on failure, returns the
// failing cudaError_t and the caller can use cudaGetErrorString() on it.
// ---------------------------------------------------------------------
cudaError_t msp_launch_gradient_A_kernel(
    const float* d_x,
    const float* d_grad_u,
    float* d_grad_A,
    int batch_size,
    int in_features,
    int rank,
    float current_temp_c,
    float critical_temp_c,
    int freeze_stride,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    dim3 block(threads_per_block, 1, 1);
    dim3 grid((in_features + threads_per_block - 1) / threads_per_block, rank, 1);

    msp_gradient_A_thermal_gated_kernel<<<grid, block, 0, stream>>>(
        d_x, d_grad_u, d_grad_A,
        batch_size, in_features, rank,
        current_temp_c, critical_temp_c, freeze_stride
    );

    // Catches launch-configuration errors (e.g. invalid grid/block dims).
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[MSP CUDA] kernel launch failed: %s\n",
                cudaGetErrorString(launch_err));
        return launch_err;
    }

    // Catches execution errors (e.g. illegal memory access) that only
    // surface once the kernel actually runs. Synchronous by design here
    // so callers get a definitive success/failure signal; a
    // latency-sensitive caller that already knows its buffers are correct
    // can skip this and rely on the next synchronization point instead.
    cudaError_t exec_err = cudaStreamSynchronize(stream);
    if (exec_err != cudaSuccess) {
        fprintf(stderr, "[MSP CUDA] kernel execution failed: %s\n",
                cudaGetErrorString(exec_err));
        return exec_err;
    }

    return cudaSuccess;
}

// ---------------------------------------------------------------------
// Build (NVIDIA GPU + CUDA toolkit required -- not available in the
// sandbox this project was developed in):
//
//   nvcc -O3 -arch=sm_80 --compiler-options -Wall \
//        -c src/cuda/fused_msp_backward_kernel.cu \
//        -o fused_msp_backward_kernel.o
//
// Adjust -arch=sm_80 to match your target GPU's compute capability.
// There is no single flag that also targets Apple Metal or a mobile NPU
// from this file -- see the caveat at the top of this file.
// ---------------------------------------------------------------------
