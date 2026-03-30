// ============================================================
// Matrix multiplication using cuBLAS (Tensor Core accelerated)
// Target GPU : NVIDIA H100 (sm_90, Compute Capability 9.0)
// ============================================================
//
// TODO: Implement matrix multiplication using the cuBLAS library.
//
// Key headers to include:
//   #include <cublas_v2.h>
//
// Key API calls:
//   cublasCreate()       — initialise a cuBLAS handle
//   cublasSgemm()        — single-precision general matrix multiply (FP32)
//   cublasGemmEx()       — mixed-precision GEMM (FP16 compute → Tensor Cores)
//   cublasDestroy()      — clean up handle
//
// Link with: -lcublas  (add to Makefile LDFLAGS)
//
// Expected performance on H100: ~50–60 TFlop/s (FP16 Tensor Core path)
// ============================================================
