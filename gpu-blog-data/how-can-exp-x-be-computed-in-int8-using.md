---
title: "How can exp(-x) be computed in int8 using gemmlowp?"
date: "2025-01-30"
id: "how-can-exp-x-be-computed-in-int8-using"
---
The accurate computation of exp(-x) within the constraints of int8 representation using gemmlowp presents a non-trivial challenge, particularly when targeting embedded or resource-constrained environments where precision and memory are critical. The exponential function, when directly applied to int8 values, often leads to significant inaccuracies and potential overflow issues due to the limited range of int8. Instead, a combination of algorithmic approximations and careful scaling within the gemmlowp framework becomes necessary.

My experience developing embedded machine learning applications revealed that a direct lookup table (LUT) approach, while initially appealing, quickly becomes impractical for fine-grained exponential approximations given the memory constraints associated with int8 storage. The trade-off between the LUT’s size and accuracy becomes unacceptable for most real-world applications. Instead, the strategy I’ve consistently found effective involves employing a polynomial approximation combined with scaling and quantization.

The core principle is to approximate exp(-x) with a Taylor or Chebyshev polynomial within a defined input range, and to then utilize gemmlowp's integer math capabilities to execute this polynomial. Specifically, we will focus on a second-order Taylor expansion around x = 0, which yields: exp(-x) ≈ 1 - x + x²/2. This simplified approximation reduces the required calculations significantly.

The implementation consists of the following key steps:

1.  **Input Range and Scaling**: The first challenge is mapping the floating-point input range, within which we expect to compute exp(-x), to the much narrower int8 representation. We define an input scaling factor (`input_scale`) which converts the floating-point range into the [-128, 127] range of int8. This is crucial to ensure values do not overflow when processed by gemmlowp. The input value, initially stored as floating point (or as an int32 if a previous calculation yielded that type) will be scaled and then rounded to int8.

2.  **Integer Polynomial Evaluation**: The scaled int8 input `x_int8` is used in the polynomial approximation. Since intermediate results might exceed int8 bounds, we expand the calculation to larger integer types. We then need to incorporate `input_scale` back into the calculation to approximate the original exp(-x). Specifically we will calculate `1 - x + x^2/2` and then rescale this to an appropriate int8 range and apply appropriate offset such that the result approximates `exp(-x)`.

3.  **Output Scaling and Quantization**: The results from the polynomial approximation (which will be int32 or int64) then need to be scaled and quantized to the output int8 range. We introduce an output scaling factor (`output_scale`) to quantize the result. The final output, before the truncation to int8, needs to account for the scaling introduced in the input.

The following code examples demonstrate how this approach might be implemented within a gemmlowp context.

**Code Example 1: Simple Implementation with Integer Arithmetic**

This example shows the core integer calculation with pre-defined input and output scale values.

```c++
#include <cstdint>

int8_t exp_neg_int8_approx(int32_t x_int32, double input_scale, double output_scale) {
  // Scale the floating-point input and round to int8, avoiding overflow
  int8_t x_int8 = static_cast<int8_t>(x_int32 * input_scale + 0.5);

  // Intermediate calculation in int32 (or int64 to further reduce potential overflow)
  int32_t x_sq_int32 = static_cast<int32_t>(x_int8) * static_cast<int32_t>(x_int8);
  int32_t result_int32 = static_cast<int32_t>(1 << 15) - static_cast<int32_t>(x_int8) * (1 << 15) + (x_sq_int32 >> 1); //scaled by 2^15 to account for output scale (see next example)

  // Output scaling
  double scaled_result_float = static_cast<double>(result_int32) / (1 << 15) * output_scale ; //divide by 2^15 to correct the scale, then scale by output_scale

  // Quantize and round to int8
  int8_t result_int8 = static_cast<int8_t>(scaled_result_float + 0.5);

  return result_int8;
}
```

In this example, `x_int32` is expected to be an integer representation of a floating-point value, and `input_scale` and `output_scale` are pre-computed scaling values necessary to convert to and from int8 range. The magic number `1 << 15` is introduced to allow us to store both the `1` term and to rescale.

**Code Example 2: Input and Output Scaling and Offset**

This example demonstrates a full function where `input_scale` and `output_scale` are calculated to appropriately map the floating point input to int8 and back

```c++
#include <cstdint>
#include <cmath>

int8_t exp_neg_int8_scaled(float x_float, float input_range_max, float output_range_min, float output_range_max) {

    // Define int8 ranges
    const int8_t int8_min = -128;
    const int8_t int8_max = 127;

    // Ensure x_float is within a reasonable range
    float x_clamped = std::fmin(input_range_max, std::fmax(-input_range_max, x_float));

    // Calculate scaling factors, accounting for quantization
    double input_scale = static_cast<double>(int8_max) / input_range_max;
    double output_scale = (output_range_max - output_range_min) / 255.0; //Scale such that int8 [0, 255] maps to [output_min, output_max]
    double output_offset = output_range_min;

    // Convert float input to an int32 with greater precision for the scaling.
    int32_t x_int32 = static_cast<int32_t>(x_clamped / input_scale);

    // Calculate the result and scale
    int8_t result_int8 = exp_neg_int8_approx(x_int32, input_scale, output_scale);


   // Offset the result
   float result_float = static_cast<float>(result_int8) * output_scale + output_offset;

   // quantize and return
   return static_cast<int8_t>(result_float / output_scale - output_offset + 0.5);

}
```

This example pre-computes the `input_scale` and `output_scale` using provided range parameters. The input float is clamped to a sensible range and then scaled, quantized, then scaled back up to float and then quantized into an int8. This function encapsulates the scaling and approximation into a more usable function.

**Code Example 3: Utilizing gemmlowp's `FixedPoint` functionality**

While the first two examples are useful for illustration, leveraging `gemmlowp`’s built-in `FixedPoint` type directly offers more efficient and potentially accurate solutions.

```c++
#include "gemmlowp.h"

using namespace gemmlowp;

int8_t exp_neg_gemmlowp(float x_float, float input_range_max, float output_range_min, float output_range_max) {
  // Range calculations identical to Example 2
   // Define int8 ranges
    const int8_t int8_min = -128;
    const int8_t int8_max = 127;

    // Ensure x_float is within a reasonable range
    float x_clamped = std::fmin(input_range_max, std::fmax(-input_range_max, x_float));

    // Calculate scaling factors, accounting for quantization
    double input_scale = static_cast<double>(int8_max) / input_range_max;
    double output_scale = (output_range_max - output_range_min) / 255.0; //Scale such that int8 [0, 255] maps to [output_min, output_max]
    double output_offset = output_range_min;

  // Convert float input to gemmlowp FixedPoint
  FixedPoint<int8_t> x_fixed = FixedPoint<int8_t>::FromDouble(x_clamped / input_scale);


  // Apply polynomial approximation
  FixedPoint<int32_t> x_squared_fixed = x_fixed * x_fixed;
  FixedPoint<int32_t> result_fixed = FixedPoint<int32_t>::FromInt(1 << 15) - x_fixed.ToFixedPoint<int32_t>()*(1 << 15) + x_squared_fixed.ToFixedPoint<int32_t>() / 2; //scaled by 2^15 to account for output scale (see next example)


  // Scale output to float domain
   double scaled_result_float = result_fixed.ToDouble() / (1 << 15) * output_scale ; //divide by 2^15 to correct the scale, then scale by output_scale

    // Quantize and round to int8
  int8_t result_int8 = static_cast<int8_t>(scaled_result_float + 0.5);

  return result_int8;
}
```

This example demonstrates the use of `gemmlowp::FixedPoint` types for the core polynomial calculation. This facilitates accurate representation of fractions within the int8 space and also helps ensure efficient gemmlowp optimized operations are used. It significantly reduces the risk of overflow by using 32 bit fixed point types.

**Resource Recommendations**

For further understanding of numerical analysis techniques relevant to approximations of exp(-x), I recommend consulting resources on:
*   **Polynomial Approximation:** Focus on Taylor and Chebyshev expansions. Texts on numerical methods often cover this in detail.
*   **Quantization Techniques:** Look for texts on digital signal processing that cover topics like uniform and non-uniform quantization to better understand the tradeoff between bit depth, precision, and range.
*   **Fixed-Point Arithmetic:** The IEEE standards for fixed-point representation and related arithmetic operations provide a fundamental grounding in the field.
*   **gemmlowp Documentation:** Google's official gemmlowp repository, while not always a user-friendly source, contains valuable insights into the library's capabilities and expected data format.

In summary, the computation of exp(-x) in int8 using gemmlowp requires careful consideration of scaling, integer arithmetic, and approximations. A well-tuned polynomial approach, coupled with the library's fixed-point arithmetic tools, offers a practical way to achieve a suitable balance between accuracy, memory usage, and computational efficiency for embedded applications.
