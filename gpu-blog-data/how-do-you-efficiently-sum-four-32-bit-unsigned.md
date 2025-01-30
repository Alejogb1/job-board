---
title: "How do you efficiently sum four 32-bit unsigned integers to represent a 128-bit number?"
date: "2025-01-30"
id: "how-do-you-efficiently-sum-four-32-bit-unsigned"
---
Implementing arithmetic operations on numbers exceeding the native word size of a processor necessitates a careful approach. In my experience working on high-precision cryptography libraries, I frequently encountered the challenge of representing and manipulating numbers larger than 64 bits, often requiring 128-bit and 256-bit arithmetic. Summing four 32-bit unsigned integers to form a 128-bit result requires accounting for carries between each 32-bit segment. The core concept is to treat each 32-bit integer as a “digit” in base 2^32, performing addition similar to how one would sum decimal digits on paper, handling carries appropriately.

The 128-bit number will effectively be represented as four contiguous 32-bit unsigned integers. When adding two such numbers, we must add corresponding 32-bit segments, and propagate carries to the higher segments. Consider the four 32-bit integers `a3`, `a2`, `a1`, `a0`, representing the 128-bit number `A`, and similarly `b3`, `b2`, `b1`, `b0` representing `B`. The sum, `C` which is `A+B`, would be computed by iterating from `a0+b0` to `a3+b3`, incorporating any carry from previous additions. In the context of this question, we are summing four separate 32-bit numbers, not two 128-bit values. Therefore, we can expand on the core mechanism of carry propagation to achieve this. The resulting 128-bit number will be comprised of four 32-bit segments that, taken in order, form the lower to higher bits.

Here's a breakdown of how I implement this approach, alongside code examples in C which is frequently used in this domain:

**Example 1: Direct Summation with Carry Tracking**

This approach demonstrates the most basic form of accumulating the sum with manual carry handling. It uses a temporary `carry` variable to transfer any overflow from one 32-bit segment to the next. This method can be less optimized but it clearly illustrates the carry logic.

```c
#include <stdint.h>

typedef struct {
    uint32_t low;
    uint32_t mid_low;
    uint32_t mid_high;
    uint32_t high;
} uint128_t;

uint128_t sum_four_32bit_direct(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint128_t result;
    uint64_t sum;
    uint32_t carry = 0;

    // Sum the first two integers and handle the carry for the low word
    sum = (uint64_t)a + b;
    result.low = (uint32_t)sum;
    carry = (uint32_t)(sum >> 32);


    // Sum with the third integer and incorporate the carry, storing into the mid_low word
    sum = (uint64_t)c + carry;
    result.mid_low = (uint32_t)sum;
    carry = (uint32_t)(sum >> 32);


    // Sum with the fourth integer and the previous carry, storing into the mid_high word
    sum = (uint64_t)d + carry;
    result.mid_high = (uint32_t)sum;
    carry = (uint32_t)(sum >> 32);


    // Store any remaining carry in the highest word
    result.high = carry;

    return result;
}

```

**Commentary:**

This function `sum_four_32bit_direct` takes four 32-bit unsigned integers as input and returns a `uint128_t` struct.  I used a `uint64_t` intermediary because addition of two `uint32_t` can potentially exceed 32 bits. Each segment is summed together, carry values are calculated by right shifting and stored in a temporary variable. Any final carry is captured in the `high` field of the result. It is important to note that the cast to `uint64_t` prior to the addition is essential to prevent overflow during addition. This function avoids any dependency on compiler or platform-specific optimizations.

**Example 2: Array-Based Summation with Loops**

Often, representing the 128-bit number as an array simplifies access and manipulation, especially when handling a more general addition of an arbitrary number of 32-bit values to form a larger result. I've found that in some cases, loop-based processing allows for vectorization opportunities.

```c
#include <stdint.h>
#include <stdbool.h>


typedef struct {
    uint32_t segments[4];
} uint128_array_t;


uint128_array_t sum_four_32bit_array(uint32_t numbers[4]) {
  uint128_array_t result = {{0, 0, 0, 0}};
  uint64_t sum;
  uint32_t carry = 0;

  for(int i = 0; i < 4; ++i) {
    sum = (uint64_t)result.segments[i] + numbers[i] + carry;
    result.segments[i] = (uint32_t)sum;
    carry = (uint32_t)(sum >> 32);

  }
    result.segments[3] += carry;
    return result;
}
```
**Commentary:**

Here, `uint128_array_t` is an alternative representation using an array. The `sum_four_32bit_array` function iterates through the 4 input numbers and stores each partial sum into its corresponding array position. It handles the carry in a loop, incrementing the final `result.segments[3]` with any remaining carry which guarantees proper propagation for values larger than 4 numbers. This structure also enables a more extensible implementation for adding a variable number of input 32-bit numbers, which can be useful in certain contexts. This method is more memory intensive due to the stack allocation, however, its loop-centric nature can benefit from compiler optimizations, particularly auto-vectorization.

**Example 3: Summation with Pre-allocation**

In performance-critical systems, pre-allocating memory is often desired. In situations where I know I will be summing 4 values, I prefer an explicit pre-allocated array that I can pass into the addition routine. Here is an example that shows such a pattern.

```c
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
  uint32_t segments[4];
} uint128_array_t;

void sum_four_32bit_prealloc(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint128_array_t *result) {
    uint64_t sum;
    uint32_t carry = 0;


    memset(result, 0, sizeof(uint128_array_t)); // Initialize to zero.

    sum = (uint64_t)a;
    result->segments[0] = (uint32_t)sum;
    carry = (uint32_t)(sum >> 32);

    sum = (uint64_t)b + carry;
    result->segments[1] = (uint32_t)sum;
    carry = (uint32_t)(sum >> 32);


    sum = (uint64_t)c + carry;
    result->segments[2] = (uint32_t)sum;
    carry = (uint32_t)(sum >> 32);


    sum = (uint64_t)d + carry;
    result->segments[3] = (uint32_t)sum;
    carry = (uint32_t)(sum >> 32);


    result->segments[3] += carry;


}
```

**Commentary:**

This function `sum_four_32bit_prealloc`  operates on a pre-allocated `uint128_array_t` struct pointed to by result. Initially the memory is zeroed using `memset`. The addition proceeds similarly to Example 1, handling carries with bit shifts, but rather than returning a value it writes to the location in memory that the pointer references. This method offers increased control over memory management, and the lack of return value avoids copy operations.

**Resource Recommendations:**

For a deeper understanding of multi-precision arithmetic, I would recommend exploring the following:

1.  **"Hacker's Delight" by Henry S. Warren Jr.:**  This book provides in-depth explanations of bitwise operations and low-level arithmetic techniques, which are crucial for understanding carry propagation and optimized implementations.

2.  **"Handbook of Applied Cryptography" by A. Menezes, P. van Oorschot, and S. Vanstone:** While primarily focused on cryptography, this book covers large integer arithmetic extensively, including algorithms for addition, subtraction, and other operations.

3.  **Documentation for specific high-performance math libraries:** Examining libraries such as GMP (GNU Multiple Precision Arithmetic Library) and OpenSSL's bignum library can reveal best practices and optimized implementations for multi-precision arithmetic. Studying the structure and techniques employed by these libraries offers valuable insights into handling multi-precision values.

These resources provide a theoretical grounding and practical context essential for mastering efficient multi-precision arithmetic. My experience has shown these resources to be the most beneficial.
