import matplotlib.pyplot as plt
import pandas as pd
import io

cublas_data = """dtype,N,ms,tflops
BF16,32,0.0076,0.01
BF16,64,0.0076,0.07
BF16,128,0.0046,0.91
BF16,256,0.0047,7.12
BF16,512,0.0056,47.53
BF16,1024,0.0117,183.18
BF16,2048,0.0294,585.11
BF16,4096,0.1776,773.71
"""
wmma_data = """dtype,N,ms,tflops
BF16,32,0.0035,0.02
BF16,64,0.0044,0.12
BF16,128,0.0069,0.61
BF16,256,0.0104,3.23
BF16,512,0.0173,15.48
BF16,1024,0.0313,68.56
BF16,2048,0.0947,181.50
BF16,4096,0.6661,206.33
"""
wgmma_data = """dtype,N,ms,tflops
BF16+WGMMA,128,0.0038,1.11
BF16+WGMMA,256,0.0050,6.74
BF16+WGMMA,512,0.0077,34.82
BF16+WGMMA,1024,0.0131,163.34
BF16+WGMMA,2048,0.0404,425.70
BF16+WGMMA,4096,0.2946,466.59
BF16+WGMMA,8192,2.2923,479.65
"""
wgmma_tma_data = """dtype,N,ms,tflops
BF16+TMA,128,0.0035,1.19
BF16+TMA,256,0.0041,8.10
BF16+TMA,512,0.0060,44.56
BF16+TMA,1024,0.0097,221.37
"""

def extract_df(csv_str): return pd.read_csv(io.StringIO(csv_str))

df_cublas = extract_df(cublas_data)
df_wmma = extract_df(wmma_data)
df_wgmma = extract_df(wgmma_data)
df_wgmma_tma = extract_df(wgmma_tma_data)

plt.figure(figsize=(10, 6))
plt.plot(df_cublas['N'], df_cublas['tflops'], 's-', label='cuBLAS (BF16)')
plt.plot(df_wmma['N'], df_wmma['tflops'], 'o-', label='WMMA (BF16)')
plt.plot(df_wgmma['N'], df_wgmma['tflops'], '^-', label='WGMMA (BF16)')
plt.plot(df_wgmma_tma['N'], df_wgmma_tma['tflops'], 'd-', label='WGMMA+TMA (BF16)')

plt.xscale('log', base=2)
# Using a linear scale for Y to properly show the magnitude of TFLOPS differences
plt.ylabel('Performance (TFLOPS) - HIGHER IS BETTER')
plt.xlabel('Matrix Dimension (N)')
plt.title('Performance Comparison (TFLOPS): cuBLAS vs WMMA vs WGMMA vs WGMMA+TMA')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.savefig('/root/cuda-assignment/build/benchmark_tflops.png')
print("Saved TFLOPS plot!")
