---
title: "Why does the HPL HPCG benchmark run slower on Nvidia v100 GPUs using half the power?"
date: "2025-01-30"
id: "why-does-the-hpl-hpcg-benchmark-run-slower"
---
The observed performance discrepancy between the HPCG benchmark on Nvidia V100 GPUs at reduced power consumption isn't solely attributable to a simple linear relationship between power and performance.  My experience optimizing HPC applications on similar hardware reveals that the HPCG benchmark's sensitivity to memory bandwidth and latency, coupled with the GPU's power-saving mechanisms, is the key factor.  While a 50% power reduction might suggest a proportional performance decrease, the reality is more complex, involving interactions between clock speed, memory clock, and voltage scaling.

**1. Detailed Explanation:**

The HPCG benchmark, unlike LINPACK-based benchmarks, stresses the entire memory subsystem of the GPU, including memory bandwidth and latency.  Power-saving modes implemented in Nvidia V100 GPUs typically affect multiple aspects of the hardware simultaneously.  Reducing power consumption often triggers a cascade of adjustments:

* **Clock Speed Reduction:** The GPU core clock frequency is directly reduced. This immediately impacts the number of floating-point operations per second (FLOPS) the GPU can achieve.  The effect is directly proportional – a 20% power reduction might lead to approximately a 20% clock speed reduction, but this is not always linear and depends on the specific power management scheme.

* **Memory Clock Reduction:**  The memory clock frequency also decreases under power-saving conditions.  This has a significant impact on HPCG, as its performance is heavily bound by the rate at which data can be accessed from and written to the GPU memory.  A reduced memory clock directly translates to lower memory bandwidth, leading to substantial performance degradation, potentially exceeding the reduction in core clock speed.

* **Voltage Scaling:**  Lowering the power supply voltage alongside the clock frequency reduction further affects the performance. Lower voltages can increase latency and decrease the reliability of computations, leading to potential performance degradation or even errors. This indirect effect can sometimes be significant, particularly under heavy memory access pressure as in HPCG.

* **Power Throttling:**  Aggressive power saving might activate power throttling mechanisms.  These can introduce unpredictable performance variations as the GPU dynamically adjusts its power consumption in response to thermal or power limits.  This non-deterministic behavior can further complicate the analysis of the observed performance slowdown.

In summary, the observed slower performance on Nvidia V100 GPUs operating at half power is a consequence of the combined effects of reduced core clock, memory clock, and potential voltage scaling, and possibly power throttling.  These factors interact nonlinearly, leading to performance degradation that's often more significant than a simple proportional decrease based on power consumption alone.  Optimizing for power efficiency typically necessitates a trade-off with peak performance.


**2. Code Examples & Commentary:**

The following examples demonstrate how these factors impact performance.  Note that these are simplified representations for illustrative purposes; real-world scenarios necessitate much more detailed profiling and analysis.


**Example 1:  Measuring Clock Speeds**

```c++
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int major, minor;
    cudaDriverGetVersion(&major, &minor);
    printf("CUDA Driver Version: %d.%d\n", major, minor);

    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU Name: %s\n", prop.name);
    printf("Clock Rate: %d MHz\n", prop.clockRate);
    printf("Memory Clock Rate: %d MHz\n", prop.memoryClockRate);

    //Further code to measure clock speed under different power states would be needed here.
    //This would involve using CUDA or NVIDIA SMI tools to manipulate power states.

    return 0;
}
```

This code snippet retrieves basic GPU information, including clock speeds.  To measure the impact of power scaling, one would need to integrate it with power management tools or libraries (e.g., the NVIDIA Management Library) to actively control power states and monitor the resulting clock speed changes.


**Example 2:  Simple HPCG Kernel (Illustrative)**

```c++
__global__ void hpcg_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(){
    //Memory allocation, data transfer, kernel launch...etc, omitted for brevity.
    //Focus is on illustrating memory bandwidth sensitivity.

    //Illustrating a simple addition operation; a real HPCG kernel is far more complex.
    hpcg_kernel<<<(n + 255)/256, 256>>>(d_a, d_b, d_c, n); //Kernel launch
    //Error checking, data transfer back to host...etc, omitted for brevity.
    return 0;
}
```

This simplified kernel illustrates the fundamental HPCG operation – element-wise addition.  Real-world HPCG involves far more complex operations, data structures, and communication patterns. This demonstrates, however, that the memory bandwidth is crucial due to the necessity of continuously reading and writing to memory.  Performance degradation with reduced memory clock is directly observable in increased kernel execution time.


**Example 3:  Profiling with NVIDIA Nsight Systems**

NVIDIA Nsight Systems is a crucial tool for performance analysis on NVIDIA GPUs.  It allows detailed profiling of GPU kernels, memory accesses, and other performance-critical aspects.  By running HPCG with Nsight Systems under different power states, one can obtain quantitative data on memory bandwidth, latency, and other relevant metrics.  The profiler output can directly highlight the effect of power reduction on the different performance bottlenecks in HPCG.

```bash
# This is a command-line example, the specific options depend on the version of Nsight Systems
nsys profile -t cuda --output profile_result HPCG_executable
```

This command-line instruction triggers profiling, enabling a detailed analysis of the HPCG performance under the specified power state.  Comparing profiles obtained with different power levels provides quantitative data to support the qualitative analysis presented earlier.


**3. Resource Recommendations:**

* NVIDIA CUDA Toolkit Documentation
* NVIDIA Nsight Systems User Guide
* High-Performance Computing (HPC) textbooks focusing on GPU programming and performance analysis.
* Research papers on GPU power management and performance optimization.  Focus on publications related to Nvidia V100 architecture.


In conclusion, the slower performance of HPCG at reduced power consumption on Nvidia V100 GPUs stems from a complex interplay of clock speed reduction, memory bandwidth limitations, potential voltage scaling effects, and possible power throttling.  Detailed profiling using tools like NVIDIA Nsight Systems is essential for quantitatively assessing the impact of each of these factors.  My experience across numerous HPC optimization projects reinforces the notion that simply halving power rarely results in a directly proportional performance reduction, particularly for memory-bandwidth-bound applications like HPCG.
