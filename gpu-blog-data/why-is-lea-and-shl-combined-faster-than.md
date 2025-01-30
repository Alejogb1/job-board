---
title: "Why is `lea` and `shl` combined faster than `imul` for constant multiplication, according to GCC -O2?"
date: "2025-01-30"
id: "why-is-lea-and-shl-combined-faster-than"
---
The optimization of integer multiplication using `lea` (load effective address) and `shl` (shift left) instead of `imul` (integer multiplication) stems from the architecture-specific cost of operations and the compiler's ability to recognize multiplication by a constant as a series of address calculations and bit shifts. My experience with low-level optimization on x86 platforms has frequently exposed this optimization strategy, particularly when targeting performance-critical sections.

The fundamental principle is that `imul`, while a single instruction, is often a more complex operation for the processor's execution unit, requiring multiple clock cycles for computation and potentially stalling the instruction pipeline. In contrast, `lea` is designed for address calculation and typically executes in a single clock cycle. `shl`, which implements bitwise left shift, also typically executes quickly, often on the same pipeline stage as address calculation. When a multiplication by a small, power-of-two constant or a constant that can be expressed as a combination of shifts and additions, can be constructed using `lea` and `shl`, the resulting execution is often faster and utilizes fewer processor resources. GCC, when given the `-O2` optimization level, exploits these cost differentials to substitute `imul` with an equivalent sequence of `lea` and `shl` when advantageous.

Let’s consider a practical scenario where we need to multiply an integer by 10. Multiplication by 10 cannot be directly represented by a single shift but can be expressed as (value * 2) + (value * 8). This can be further translated into a series of shifts and additions, effectively utilizing `lea` for the addition aspect.

**Example 1: Multiplication by 10**

```c
int multiply_by_ten(int x) {
  return x * 10;
}
```

When compiling this with GCC using `-O2` for x86-64, the generated assembly output (using `objdump -d` on the compiled object file) might be similar to this:

```assembly
_multiply_by_ten:
  mov  eax, edi       ; Move input 'x' to eax register
  lea  eax, [rax+rax*4] ; eax = x + x*4 (x*5) using lea address calculation
  shl  eax, 1         ; eax = (x*5) << 1, which is x * 10 (x*5*2)
  ret                 ; Return the result in eax
```

**Commentary:**

*   The input integer `x` is first copied into the `eax` register.
*   The `lea` instruction calculates `x + x*4` without modifying memory. This effectively multiplies x by 5 (x + 4x = 5x). It achieves this multiplication by only manipulating the address calculation logic of the CPU, not using the general multiplication functional unit.
*   `shl eax, 1` performs a bitwise left shift of `eax` by one bit. This is equivalent to multiplying `eax` by 2, resulting in a final value of `x * 10`.
*   The function then returns the result in the `eax` register, according to the x86-64 calling convention.

Observe that the compiler avoided the use of the `imul` instruction entirely. This substitution is advantageous because address calculations and shifts are typically faster than full multiplication instructions on x86 architectures.

Now, consider another case involving multiplication by 36.  This is not a simple power of two, but we can express it as (x * 32) + (x * 4). This utilizes the same principles as the previous example.

**Example 2: Multiplication by 36**

```c
int multiply_by_thirty_six(int x) {
    return x * 36;
}
```

Compiling this with GCC -O2 gives a sequence similar to:

```assembly
_multiply_by_thirty_six:
  mov  eax, edi       ; move x into eax
  lea  eax, [rdi+rdi*8] ; eax = x + x * 8 (x*9) using lea
  shl  eax, 2          ; eax = (x*9) << 2 which is equivalent to x * 36 (x*9*4)
  ret                ; Return the result
```
**Commentary:**
* The input `x` is again moved to register `eax`.
* An `lea` calculates x + 8x, which equals 9x. Again, no general-purpose multiplier was used.
* Then, `shl eax, 2` is used to perform the shift left by 2 bits, which multiplies by 4. This yields a total multiplication of 9 * 4 or 36.
* The result is then returned in the eax register.

Let's consider a slightly more complex example: multiplication by a constant that's not a sum of two powers of 2.  Let's use 15. This can be expressed as (16*x) - x which translates to shifting left by four and then subtracting. However, the compiler will recognize that this subtraction can be better handled by using address calculations.

**Example 3: Multiplication by 15**

```c
int multiply_by_fifteen(int x) {
    return x * 15;
}
```
The generated assembly by GCC -O2:
```assembly
_multiply_by_fifteen:
  mov  eax, edi      ; Move input x into eax
  lea  eax, [rdi+rdi*2]  ; eax = x+x*2 or 3x
  lea eax, [rax+rax*4]  ; eax = 3x + 12x or 15x
  ret                 ; Return the result
```
**Commentary:**

* The value `x` is moved into the `eax` register.
* The `lea` instruction calculates `x + x*2`, or `3x`, without using multiplication instructions.
* The second `lea` calculates `3x + 3x*4` or `15x`, again without multiplication instructions.
*  The result is returned in `eax`.

These examples demonstrate how the compiler intelligently chooses less expensive operations. The `imul` instruction would have been a single instruction, but it’s architecturally slower than the sequences using `lea` and `shl`, particularly for small constants or constants that can be constructed by address computations and shifts.

It's crucial to recognize that not all multiplication by constants will be optimized this way. The complexity of the constant, target architecture, and optimization level all play roles in deciding when to prefer shifts and `lea` to `imul`. If the constant is sufficiently complex such that generating such a sequence is longer or as complex as a multiplication instruction (or if the target architecture’s multiplication unit is very fast, relative to the address calculation unit), then the compiler may choose `imul`. This choice is specific to the target processor's microarchitecture, as it is influenced by the latency of each unit. Therefore, the decision to substitute `imul` is not always straightforward.

For further study into compiler optimizations and low-level processor instructions, I recommend focusing on resources discussing:

1.  **x86-64 Assembly Language Programming:** Understanding the instruction set is critical for recognizing these optimization patterns. Reference manuals and books focusing on assembly language are a must.
2.  **Compiler Theory and Design:** Books and courses covering compiler optimization techniques, particularly instruction scheduling and peephole optimization, offer valuable insight into how such transformations are generated.
3.  **Computer Architecture Texts:** Resources detailing processor pipelines, instruction execution units, and memory access will aid in understanding the performance benefits of various instruction choices on specific architectures.
4.  **GCC documentation:** Reading through documentation specific to GCC optimization options provides insight into when different transformations are performed.

Understanding these areas has significantly improved my own ability to debug, optimize, and diagnose low-level performance issues. Examining assembly code is often necessary to grasp the true impact of high-level code, especially when attempting to exploit performance at the limits.
