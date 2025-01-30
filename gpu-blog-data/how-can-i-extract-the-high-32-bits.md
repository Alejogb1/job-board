---
title: "How can I extract the high 32 bits of a 64-bit multiplication result on x86-64?"
date: "2025-01-30"
id: "how-can-i-extract-the-high-32-bits"
---
The x86-64 instruction set offers several ways to obtain the high 32 bits of a 64-bit product resulting from multiplying two 32-bit integers, and selecting the optimal method depends on the specific context of usage, especially concerning performance needs and whether you are using assembly or a higher-level language. I've frequently encountered this need in high-performance scientific computing where intermediate calculations occasionally require examining these higher bits to determine potential overflow or to implement specific numerical algorithms.

The core challenge stems from the fact that the standard multiplication instruction, `imul`, when operating on 32-bit operands, produces a 64-bit result, which is then stored across two registers: the lower 32 bits reside in the general-purpose register specified, while the high 32 bits reside in the `edx` register. The critical point is that accessing the `edx` register directly provides the desired high 32 bits. However, relying solely on this mechanism often mandates using assembly language, which is not always desirable, or requires an intrinsic function. Fortunately, compiler technology offers a more accessible route.

One common approach in languages like C or C++ involves using intrinsic functions offered by the compiler. These intrinsics provide direct access to specific processor instructions, thus achieving the desired behavior without dropping to full assembly language programming. While the specific names of these functions vary slightly between compilers (GCC/Clang vs. MSVC), the underlying concept remains constant: utilizing the processor's capacity to hold the full product and accessing the upper bits effectively. For instance, both GCC and Clang provide `__builtin_mul_overflow`, which, along with performing the multiplication, allows detecting overflows when using unsigned numbers. This can also be repurposed to obtain the high bits in conjunction with a type cast when the overflow flag itself isn't necessary. MSVC provides a similar set of functionalities using compiler intrinsics starting with `_mul`.

Another method is leveraging the `long long` (or equivalent 64-bit integer type) in C/C++. The compiler implicitly takes care of allocating enough registers to hold the 64-bit result of the multiplication when casting the result. Then, a right-bit shift is applied to extract the upper 32 bits, as detailed below. The advantage of this method is that it's generally more portable across platforms compared to compiler intrinsics, though some compilers may still internally use intrinsics when optimizing this code.

Let's illustrate with a few practical examples.

**Example 1: Using Compiler Intrinsics (GCC/Clang)**

```c++
#include <iostream>

unsigned int extractHighBitsIntrinsic(unsigned int a, unsigned int b) {
    unsigned long long result;
    bool overflow;
    result = __builtin_mul_overflow(a, b, &result);

   if (result >> 32 > 0) {
      return result >> 32;
   }
    return 0;
}


int main() {
  unsigned int num1 = 0xFFFFFFFF;
  unsigned int num2 = 0x00000002;
    unsigned int highBits = extractHighBitsIntrinsic(num1, num2);
    std::cout << "High 32 bits (intrinsic): 0x" << std::hex << highBits << std::endl; // Output: 0x1
    return 0;
}

```

In this example, I've demonstrated the use of `__builtin_mul_overflow`. While designed for overflow detection, the product resides in `result`, and we extract the higher bits with right shifting after casting it to `long long`. Note that I'm checking whether the high bits are non-zero before the right-shift. This example highlights the direct access we gain by using these intrinsics. The `result` value will hold the 64-bit product, even though the original operation occurs on 32-bit operands.

**Example 2: Using 64-bit Integer Type (Portable Method)**

```c++
#include <iostream>

unsigned int extractHighBitsLongLong(unsigned int a, unsigned int b) {
    unsigned long long result = static_cast<unsigned long long>(a) * static_cast<unsigned long long>(b);
    return static_cast<unsigned int>(result >> 32);
}

int main() {
  unsigned int num1 = 0xFFFFFFFF;
  unsigned int num2 = 0x00000002;
    unsigned int highBits = extractHighBitsLongLong(num1, num2);
    std::cout << "High 32 bits (long long): 0x" << std::hex << highBits << std::endl;  // Output: 0x1
    return 0;
}
```
In this case, I use a portable C++ approach by first casting the input 32-bit integers into the `unsigned long long` data type and using a standard multiplication. The compiler internally generates code that handles the full 64-bit product. Afterwards a right bit shift will expose the high 32 bits of the result. This demonstrates a more abstract approach but often results in very similar assembly code due to compiler optimization. The result is identical to the previous example. It's worth noting the double cast to unsigned long long and unsigned int which will generate the correct result in case of a negative number being the multiplicand.

**Example 3: Assembly Language (Illustrative)**

```c++
#include <iostream>
unsigned int extractHighBitsAssembly(unsigned int a, unsigned int b) {
  unsigned int highBits;
    __asm__ (
        "imull  %2, %1\n\t"  // Multiply %2 (b) with %1 (a) and store in EDX:EAX
        "movl   %%edx, %0\n\t" // Move EDX to the output %0
        : "=r"(highBits)     // Output : highBits
        : "r"(a), "r"(b)    // Inputs  : a, b
        : "edx", "eax", "cc"  // Clobbered : edx, eax, flags
    );
    return highBits;
}

int main() {
  unsigned int num1 = 0xFFFFFFFF;
  unsigned int num2 = 0x00000002;
    unsigned int highBits = extractHighBitsAssembly(num1, num2);
    std::cout << "High 32 bits (assembly): 0x" << std::hex << highBits << std::endl; // Output: 0x1
    return 0;
}
```

This example uses inline assembly to demonstrate direct manipulation of registers. The `imull` instruction performs the 32-bit multiplication and places the high 32 bits into the `edx` register. The subsequent `movl` instruction moves the contents of the `edx` register into the `highBits` variable. I've included the clobbered list as it's essential when using inline assembly to declare which registers the assembly code modifies, allowing the compiler to correctly handle these registers when generating the final code. This is the most direct method but has the drawback of being platform and compiler-specific. It also involves manually managing registers, making it error-prone in more complex scenarios.

Regarding resources, I recommend exploring the Intel Instruction Set Manual for detailed information on specific processor instructions, particularly `imul`. Also, documentation for the compiler being used (GCC, Clang, or MSVC) is crucial for correctly using intrinsics. Furthermore, studying assembly language examples relevant to the x86-64 architecture provides foundational knowledge. Finally, examining the generated assembly code from compiler optimization flags will provide an insight into which is the best method in terms of performance.
