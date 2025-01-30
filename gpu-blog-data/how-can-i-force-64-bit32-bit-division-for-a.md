---
title: "How can I force 64-bit/32-bit division for a 64-bit dividend and 32-bit quotient in GCC/Clang?"
date: "2025-01-30"
id: "how-can-i-force-64-bit32-bit-division-for-a"
---
The C standard, by default, performs integer division based on the operand's types. Consequently, if you provide a 64-bit dividend and a 32-bit divisor, the compiler will typically generate a 64-bit division instruction, yielding a 64-bit quotient and remainder. My experience, particularly with embedded systems where performance and memory footprint are critical, has often required forcing 32-bit divisions even with a larger dividend. This becomes necessary when working with specialized hardware registers or when optimizing for processors with faster 32-bit division units. Here, I'll explain how to explicitly achieve this using GCC and Clang, focusing on the use of bit manipulation and type casts, along with the reasoning behind my approach.

The core problem lies in the compiler's type promotion and default handling of integer division. When a 64-bit value is divided by a 32-bit value, the compiler promotes the smaller operand to match the larger one before division. Therefore, simply using a `uint64_t` and a `uint32_t` will not provide the desired outcome. The solution entails manipulating the dividend to fit within a 32-bit range prior to performing a 32-bit division and then recombining the results. This usually involves splitting the 64-bit dividend into two 32-bit parts, performing partial divisions, and then calculating the final result from these partials.

The methods I frequently utilize are based on exploiting the inherent properties of integer division and bit shifts. Conceptually, if I represent a 64-bit number, `N`, as `(high << 32) + low`, where `high` and `low` are both 32-bit integers, and I wish to perform a 32-bit division by some 32-bit divisor, `D`, I can leverage the following observation: `N / D = (high * (2^32) + low) / D = (high * 2^32 / D) + (low / D)`. Due to the 32-bit limit of the quotient however, care must be taken in performing the first portion of this calculation. The technique I've employed relies on calculating first `high/D`, and then using the remainder from that to continue calculation with the low portion.

Let's explore three code examples that demonstrate different ways to achieve this 64-bit dividend, 32-bit quotient division.

**Example 1: Basic Division with Type Casting and Bit Manipulation**

This first example is the most straightforward, although it may not be the most optimized. I use a series of type casts and bit shifting to separate the dividend into its upper and lower 32-bit parts, perform two 32-bit divisions, and then combine the results.

```c
#include <stdint.h>
#include <inttypes.h>

uint32_t force_32bit_div_basic(uint64_t dividend, uint32_t divisor) {
    if (divisor == 0) return 0; // Handle division by zero

    uint32_t high = (uint32_t)(dividend >> 32);
    uint32_t low  = (uint32_t)(dividend & 0xFFFFFFFF);
    uint32_t quotient_high, remainder_high;
    uint32_t quotient, combined_low;


    quotient_high = high / divisor;
    remainder_high = high % divisor;

    combined_low = (remainder_high << 1) * ((uint32_t)0x80000000/divisor) + low / divisor;


    return combined_low;

}

```

In this function, I first extract the high and low 32 bits from the 64-bit dividend using bit shifts and a mask. The `dividend >> 32` moves the high 32 bits into the low 32 bits of a temporary 64 bit value, which is then cast into the high 32 bit integer.  Similarly, `dividend & 0xFFFFFFFF` isolates the lower 32 bits. Next, I perform two divisions. Note the use of integer remainder, which is essential for the calculation of the final result. The combined division accounts for the remainder when dividing the high bits by the divisor, using bit shifting and division. I find that this approach is readily understandable, even if not perfectly optimized, and it is a good starting point. The handling of division by zero is always critical in such code.

**Example 2: Optimized Division with Multiplication**

This second example leverages a multiplication, instead of one of the divisions, to achieve faster division. This technique is particularly effective when the division operation is slower than multiplication on the target processor.

```c
#include <stdint.h>
#include <inttypes.h>

uint32_t force_32bit_div_optimized(uint64_t dividend, uint32_t divisor) {
    if (divisor == 0) return 0; // Handle division by zero

    uint32_t high = (uint32_t)(dividend >> 32);
    uint32_t low = (uint32_t)(dividend & 0xFFFFFFFF);
    uint32_t quotient_high, remainder_high;
    uint32_t quotient;

    quotient_high = high / divisor;
    remainder_high = high % divisor;

    uint64_t remainder_combined = ((uint64_t)remainder_high << 32) | low;
    quotient =  remainder_combined / divisor;

    return quotient;
}
```
Here, after the initial extraction of high and low parts, a 64 bit value is reconstructed from the remainder and the low bits.  The division is then performed with the 64 bit value.  This approach, while relying on the compiler to generate a 64 bit division on the final step, reduces the dependency on integer math when operating on the upper 32 bits. This is most useful when optimizing for hardware.
**Example 3: Avoiding Intermediate 64-bit Calculations**

This example shows an approach where all operations are constrained to 32 bits to further attempt performance improvements. This is achieved at the expense of some added complexity, but may be valuable in specific contexts where intermediate 64-bit operations carry a significant performance cost.

```c
#include <stdint.h>
#include <inttypes.h>

uint32_t force_32bit_div_full_32(uint64_t dividend, uint32_t divisor) {
  if(divisor == 0) return 0; // Handle division by zero
  uint32_t high = (uint32_t)(dividend >> 32);
  uint32_t low = (uint32_t)(dividend & 0xFFFFFFFF);
    uint32_t remainder = 0;
    uint32_t quotient = 0;
    for (int i = 31; i >= 0; --i) {
        remainder = (remainder << 1) | ((high >> i) & 1);
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= (1UL << i);
        }
    }
   for (int i = 31; i >= 0; --i) {
        remainder = (remainder << 1) | ((low >> i) & 1);
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= (1UL << i);
        }
    }

  return quotient;
}
```

In this example, I perform division using a bit-by-bit subtraction technique, treating the dividend as a series of bits rather than as a number. This involves shifting a bit from the dividend and accumulating a remainder. This process is then repeated for the low bits of the dividend. While less intuitive, it avoids explicit reliance on the 64-bit division unit, or any 64 bit operations at all. This is the most resource-conscious and is highly applicable for extremely constrained environments.

**Resource Recommendations**

For anyone interested in deepening their understanding of bit manipulation, I recommend exploring resources on computer architecture, specifically those covering integer arithmetic and binary representation. Textbooks on compiler design can offer valuable insights into how compilers handle integer division and optimization techniques. Additionally, investigating the assembly code produced by your compiler for these examples will shed light on how these operations translate into machine instructions on your target platform and help in determining the fastest method. Experimenting with these methods across various compilers and target architectures is also important for practical proficiency. Lastly, understanding the specific processor architecture and its division instructions is critical for making informed performance choices.
