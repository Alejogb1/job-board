---
title: "How can convolution operations be optimized for ARM processors using NEON?"
date: "2025-01-30"
id: "how-can-convolution-operations-be-optimized-for-arm"
---
Directly exploiting Single Instruction Multiple Data (SIMD) capabilities on ARM processors via NEON instruction sets provides a significant avenue for optimizing convolutional neural network (CNN) layer computations. My experience optimizing embedded vision systems on ARM platforms has consistently demonstrated that naive implementations of convolutions, even when using libraries, often fall significantly short of theoretical performance limits due to inefficient memory access patterns and a failure to vectorize the inner loops. This response will detail how NEON intrinsics and manual memory management can unlock considerable performance gains in convolutional operations.

**Understanding the Bottlenecks**

Convolution, at its core, involves numerous multiply-accumulate operations across a sliding window of input data. A standard implementation may iterate through the output feature map, calculate the corresponding input region, and then process each channel of the kernel with nested loops. This approach introduces several inefficiencies: First, processing pixels individually results in low data throughput, failing to leverage the available parallel processing power of NEON. Second, the constant loading and unloading of weights and input values from memory becomes a major bottleneck. Third, the inherent structure of CNN layers often involves processing contiguous pixel data, which can be exploited to improve memory access latency, but a standard nested loop approach is usually not optimal in this respect.

**The Power of NEON**

NEON is an extension of the ARM instruction set that provides a SIMD architecture, allowing for parallel processing of multiple data points with single instructions. NEON registers are typically 64 or 128 bits wide, permitting 2-4 floating-point or 4-16 integer operations per cycle depending on data type and instruction. By adapting the convolution algorithm to operate on blocks of pixels rather than individually, we can harness the potential of NEON. Specifically, we target the inner product within each convolution to be vectorized.

**Optimization Strategies with Code Examples**

The key strategy is to arrange data in memory and process it in vectors using NEON intrinsics. We can achieve this by rewriting loops to process small regions of data (e.g., 4x1 blocks) simultaneously, packing input data and filter weights into appropriate vectors and utilizing NEON intrinsics. This significantly increases the computation per load and reduces the overhead of loop management. Below are three examples demonstrating this strategy:

**Example 1: 1D Convolution with Integer Data**

This example focuses on a 1D convolution to demonstrate the basic principles of vectorization. Consider a simple kernel of 3 elements and an input signal. We will aim to process 4 elements at a time using a 128-bit NEON register. This example is for 8-bit unsigned integer data.

```c
#include <arm_neon.h>

void conv1d_neon_u8(uint8_t *output, const uint8_t *input, const int8_t *kernel, int in_len, int k_len) {
    int out_len = in_len - k_len + 1;
    for(int i = 0; i < out_len; i++) {
        int32_t sum = 0;
        for (int j=0; j < k_len ; j++){
            sum += (int32_t)input[i+j] * (int32_t)kernel[j];
        }
        output[i] = (uint8_t)sum;
    }
}
```

```c
#include <arm_neon.h>

void conv1d_neon_u8_opt(uint8_t *output, const uint8_t *input, const int8_t *kernel, int in_len, int k_len) {
    int out_len = in_len - k_len + 1;
    int vec_len = 16; // Number of U8s in a 128 bit register
    for (int i = 0; i < out_len; i++){
       int32_t sum = 0;
       int j = 0;
       for(j=0; j < k_len - (k_len % vec_len); j += vec_len){
            uint8x16_t in_vec = vld1q_u8(input + i+j);
            int8x16_t k_vec = vld1q_s8(kernel + j);
            int16x8_t widened_in_low = vmovl_u8(vget_low_u8(in_vec));
            int16x8_t widened_in_high = vmovl_u8(vget_high_u8(in_vec));
            int16x8_t widened_k_low = vmovl_s8(vget_low_s8(k_vec));
            int16x8_t widened_k_high = vmovl_s8(vget_high_s8(k_vec));
           int32x4_t multiplied_low_1 = vmull_s16(vget_low_s16(widened_k_low), vget_low_s16(widened_in_low));
           int32x4_t multiplied_low_2 = vmull_s16(vget_high_s16(widened_k_low), vget_high_s16(widened_in_low));
           int32x4_t multiplied_high_1 = vmull_s16(vget_low_s16(widened_k_high), vget_low_s16(widened_in_high));
           int32x4_t multiplied_high_2 = vmull_s16(vget_high_s16(widened_k_high), vget_high_s16(widened_in_high));

           sum += vaddvq_s32(vaddq_s32(vaddq_s32(multiplied_low_1, multiplied_low_2),vaddq_s32(multiplied_high_1,multiplied_high_2)));
      }
      for (; j < k_len ; j++){
        sum += (int32_t)input[i+j] * (int32_t)kernel[j];
      }
      output[i] = (uint8_t)sum;
    }
}
```

*Commentary:* The non-optimized implementation `conv1d_neon_u8` computes the convolution by iterating pixel by pixel. The optimized version `conv1d_neon_u8_opt` loads the kernel and input data in vectors of 16 and performs the multiplications and sum using vectorized instructions. Note the use of `vld1q_u8` to load data into vectors, `vmovl_u8` to widen the data type to allow for multiplication without overflow, and `vmull_s16` for performing multiplications. `vaddvq_s32` performs a horizontal sum of the vector, and `vaddq_s32` performs vector addition. Also note the handling of remainder to do scalar multiplications where a full vector is not possible.

**Example 2: 2D Convolution with Floating-Point Data**

Now, let us extend this to a 2D convolution. Here we will process a small region of input data (e.g., 2x2 pixels) at a time using NEON. For simplicity, we consider a filter of size 3x3. The optimization target is the multiplication and accumulation of 9 weights with their corresponding input region. This will work with single-precision floating-point numbers.

```c
#include <arm_neon.h>

void conv2d_neon_fp32(float *output, const float *input, const float *kernel, int in_h, int in_w, int k_h, int k_w) {
  int out_h = in_h - k_h + 1;
  int out_w = in_w - k_w + 1;
    for (int row = 0; row < out_h; row++) {
      for (int col = 0; col < out_w; col++) {
          float sum = 0.0f;
          for (int krow = 0; krow < k_h; krow++){
             for (int kcol = 0; kcol < k_w; kcol++){
                sum += input[(row+krow)*in_w + (col+kcol)] * kernel[krow*k_w + kcol];
             }
           }
          output[row * out_w + col] = sum;
        }
    }
}
```

```c
#include <arm_neon.h>

void conv2d_neon_fp32_opt(float *output, const float *input, const float *kernel, int in_h, int in_w, int k_h, int k_w) {
  int out_h = in_h - k_h + 1;
  int out_w = in_w - k_w + 1;
  int vec_len = 4; // Number of floats in a 128 bit register.
  for(int row = 0; row < out_h; row++) {
      for (int col = 0; col < out_w; col++) {
          float32x4_t sum_vec = vdupq_n_f32(0.0f);
          int krow, kcol;
          for(krow=0; krow<k_h; krow++){
            for(kcol = 0; kcol < k_w - (k_w % vec_len); kcol += vec_len){
               float32x4_t input_vec = vld1q_f32(input + (row + krow) * in_w + (col+kcol));
               float32x4_t kernel_vec = vld1q_f32(kernel + krow * k_w + kcol);
              sum_vec = vmlaq_f32(sum_vec, input_vec, kernel_vec);
            }
            for (; kcol < k_w; kcol++){
               float sum_scalar = vgetq_lane_f32(sum_vec,0) + vgetq_lane_f32(sum_vec,1) + vgetq_lane_f32(sum_vec,2) + vgetq_lane_f32(sum_vec,3);
               sum_scalar += input[(row+krow)*in_w + (col+kcol)] * kernel[krow*k_w + kcol];
               sum_vec = vdupq_n_f32(sum_scalar);
            }
          }
          output[row * out_w + col] = vgetq_lane_f32(sum_vec,0) + vgetq_lane_f32(sum_vec,1) + vgetq_lane_f32(sum_vec,2) + vgetq_lane_f32(sum_vec,3);
      }
  }
}
```

*Commentary:* The optimized `conv2d_neon_fp32_opt` version loads four floats using `vld1q_f32`, uses `vmlaq_f32` for a fused multiply-add, and accumulates the partial sums in a vector. Note the `vdupq_n_f32` instruction which initializes the summation vector. The scalar loop handles cases when kernel width is not an even multiple of the vector width. The final sum is computed by adding the lanes of the resulting sum vector. The unoptimized `conv2d_neon_fp32` uses nested loops for a purely scalar implementation.

**Example 3: Data Reorganization for Improved Cache Usage**

Beyond vectorization, memory layout plays a critical role. Rather than storing feature maps in a channel-major format (e.g., [height, width, channel]) it is often beneficial to reorganize data into blocks suited for SIMD processing and more optimal cache use. This often entails using a temporary buffer to restructure the data, trading memory overhead for improved access speed. This method is highly dataset dependent so a code example is not shown as it may be less general than the others, however, the principle involves reshaping data to optimize for SIMD load patterns.

**Resource Recommendations**

To deepen understanding of NEON programming, several resources should be explored. Consider the ARM Architecture Reference Manual, which contains extensive details on NEON instructions and data types. Additionally, ARM provides developer guides specifically tailored for NEON optimization techniques. Furthermore, investigating open-source libraries like TensorFlow Lite or Arm Compute Library provides concrete examples of highly optimized convolutional kernels utilizing NEON instructions, often involving techniques like loop unrolling and tiling in addition to the vectorizations shown here. These resources, when combined with iterative experimentation on target hardware, form the basis for effective performance tuning of convolution operations on ARM platforms.
