---
title: "How to convert float64 to 16-bit fixed-point for PYNQ?"
date: "2025-01-30"
id: "how-to-convert-float64-to-16-bit-fixed-point-for"
---
Understanding the nuances of numerical representation is crucial when working with embedded systems, particularly when interfacing high-level languages like Python with low-level hardware on platforms such as PYNQ. Specifically, directly representing a `float64` as a 16-bit fixed-point value requires careful consideration to avoid overflow and maintain acceptable precision. The inherent limitations of fixed-point necessitate a conversion process that addresses both the integer and fractional components of the floating-point value.

A fixed-point representation essentially allocates a fixed number of bits to represent the integer portion and the fractional portion of a number. Unlike floating-point representations, where the decimal point's position is dynamically determined using exponents, the position of the decimal point in fixed-point is predetermined and remains constant. This simplifies hardware implementation but introduces limitations on the dynamic range and precision.

Let's establish a practical scenario. Imagine I am developing a real-time signal processing application on a PYNQ board, where I receive sensor data as `float64` values. My hardware accelerator, however, operates on 16-bit fixed-point numbers. Consequently, conversion is unavoidable. A direct cast from `float64` to a 16-bit integer is a mistake; it loses all fractional information, and a substantial portion of the overall numerical value. A naive truncation can lead to severe data loss and inaccurate processing.

The core idea behind a conversion involves scaling the floating-point number by a carefully chosen factor and then converting it to an integer. This factor establishes the position of the implicit decimal point in our fixed-point representation. The selection of this scale factor hinges on the expected range and desired precision of the data. Suppose, for example, my data ranges from -10.0 to 10.0. A common convention for 16-bit representation is to use a signed 16-bit integer (`int16`). 

Let’s consider a 16-bit fixed-point format with, say, 7 bits dedicated to the integer part and 9 bits for the fractional part (often written as Q7.9). This means we need to determine a scaling factor that positions our numbers within the 16-bit range while preserving the desired precision. We know the maximum signed `int16` value is 32767, which can represent the absolute value of an integer up to `2^15-1`. The maximum absolute value for a number in a Q7.9 fixed-point format is approximately `2^7 - 1 + (1 - 2^-9)`, which we need to accommodate in our range.

The process includes three distinct steps: scaling, rounding, and saturation. Scaling involves multiplying the `float64` value by the scale factor. Rounding addresses how the resulting scaled number is converted to an integer, minimizing quantization errors (I prefer to use standard rounding rules rather than simple truncation). Saturation ensures we don't overflow the range of our 16-bit fixed-point representation. Values that are too large or small need to be capped to the minimum and maximum representable values, avoiding wraparound.

Here are three illustrative Python code examples demonstrating different aspects of this process:

```python
import numpy as np

def float_to_fixed_q7_9(float_val):
    """Converts a float64 value to a Q7.9 fixed-point int16.

    Args:
      float_val: The float64 value to convert.

    Returns:
      The int16 representation of the fixed-point value.
    """

    scale_factor = 2**9  # 2^9 to shift the fractional part into the integer

    scaled_value = float_val * scale_factor

    # Apply rounding using np.round to nearest integer.
    rounded_value = int(np.round(scaled_value))

    # Saturate to ensure int16 range [-32768, 32767]
    max_int16 = 2**15 - 1
    min_int16 = -2**15
    saturated_value = max(min(rounded_value, max_int16), min_int16)
    
    return np.int16(saturated_value) # Explicit cast

# Example Usage
float_val = 5.125
fixed_val = float_to_fixed_q7_9(float_val)
print(f"Float: {float_val}, Fixed: {fixed_val}, Q7.9 Representation: {fixed_val / (2**9)}")

float_val = -10.875
fixed_val = float_to_fixed_q7_9(float_val)
print(f"Float: {float_val}, Fixed: {fixed_val}, Q7.9 Representation: {fixed_val / (2**9)}")

float_val = 120.0 # Value outside of Q7.9 range
fixed_val = float_to_fixed_q7_9(float_val)
print(f"Float: {float_val}, Fixed: {fixed_val}, Q7.9 Representation: {fixed_val / (2**9)}")


```
The `float_to_fixed_q7_9` function demonstrates a basic implementation. It multiplies the `float64` input by the appropriate scale factor, rounds the result to the nearest integer, and saturates to a 16-bit range using `np.int16`. The output shows the conversion of valid and saturated values. Notice how the number outside of the valid range (120.0) gets saturated at the maximum positive number for the representation. The inclusion of `np.int16` ensures correct interpretation in PYNQ by enforcing data type explicitly.

Next, I’ll modify the code to support custom Q-formats for more flexibility:

```python
import numpy as np

def float_to_fixed_custom(float_val, integer_bits, fractional_bits):
    """Converts a float64 value to a custom fixed-point int16 format.

    Args:
      float_val: The float64 value to convert.
      integer_bits: The number of bits to use for integer portion.
      fractional_bits: The number of bits to use for fractional portion.

    Returns:
      The int16 representation of the fixed-point value.
    """

    scale_factor = 2**fractional_bits
    scaled_value = float_val * scale_factor
    rounded_value = int(np.round(scaled_value))

    max_value = (2**(integer_bits)) - 1
    min_value = -(2**(integer_bits))

    max_int16 = 2**15 - 1
    min_int16 = -2**15
    
    # Ensure that the number is within the 16-bit limits
    saturated_value = max(min(rounded_value, max_int16), min_int16)

    # Ensure we don't overflow the defined number representation range.
    saturated_value = max(min(saturated_value, max_value*scale_factor), min_value*scale_factor)

    return np.int16(saturated_value)

# Example Usage
float_val = 2.5
fixed_val = float_to_fixed_custom(float_val, 6, 10)
print(f"Float: {float_val}, Fixed: {fixed_val}, Q6.10 Representation: {fixed_val / (2**10)}")

float_val = -5.25
fixed_val = float_to_fixed_custom(float_val, 5, 11)
print(f"Float: {float_val}, Fixed: {fixed_val}, Q5.11 Representation: {fixed_val / (2**11)}")

float_val = 120.0
fixed_val = float_to_fixed_custom(float_val, 4, 12)
print(f"Float: {float_val}, Fixed: {fixed_val}, Q4.12 Representation: {fixed_val / (2**12)}")

```
The `float_to_fixed_custom` function is more flexible, allowing the caller to specify the number of bits for the integer and fractional components. It performs the same three fundamental steps: scaling, rounding, and saturation, but it adds a check to enforce the defined dynamic range for the specified Q-format.

Finally, let's consider vectorization for processing multiple values at once, a situation that might arise in signal processing applications.

```python
import numpy as np

def float_array_to_fixed(float_array, integer_bits, fractional_bits):
    """Converts an array of float64 values to fixed-point int16 format.

    Args:
      float_array: NumPy array of float64 values.
      integer_bits: Integer part bits
      fractional_bits: Fractional part bits.

    Returns:
      NumPy array of int16 representing the fixed-point values.
    """
    scale_factor = 2**fractional_bits
    scaled_values = float_array * scale_factor
    rounded_values = np.round(scaled_values).astype(int)
    max_int16 = 2**15 - 1
    min_int16 = -2**15
    max_value = (2**(integer_bits)) - 1
    min_value = -(2**(integer_bits))

    
    saturated_values = np.clip(rounded_values,min_int16,max_int16)
    saturated_values = np.clip(saturated_values, min_value*scale_factor, max_value*scale_factor)

    return saturated_values.astype(np.int16)

# Example Usage
float_array = np.array([1.2, -0.5, 3.7, 8.9])
fixed_array = float_array_to_fixed(float_array, 7, 9)
print(f"Float array: {float_array}")
print(f"Fixed array: {fixed_array}")
print(f"Q7.9 array Representation: {fixed_array / (2**9)}")


float_array = np.array([-2, 0.5, 10.75, -11])
fixed_array = float_array_to_fixed(float_array, 4, 12)
print(f"Float array: {float_array}")
print(f"Fixed array: {fixed_array}")
print(f"Q4.12 array Representation: {fixed_array / (2**12)}")

```

The `float_array_to_fixed` function accepts a NumPy array of `float64` values. By using NumPy’s vectorized operations, the code avoids explicit loops, resulting in a performance improvement. Notice that saturation using `np.clip` applies to arrays as well as single numbers.

When dealing with such data conversion, additional reference material will greatly improve implementation and development. A careful study of number representation, specifically fixed-point, from resources covering digital design will provide a deeper comprehension of the underlying mathematics and hardware constraints. Books on signal processing and digital filters frequently feature discussions on different number formats used in embedded systems. For a detailed investigation into practical coding, the documentation for numerical computing libraries such as NumPy and SciPy is invaluable. I find that comparing alternative implementations from different projects is extremely useful when creating my conversion implementations.
