---
title: "c multiplication assembly code example?"
date: "2024-12-13"
id: "c-multiplication-assembly-code-example"
---

Alright so you're looking for C multiplication translated to assembly right been there done that plenty of times let me break it down for ya I've wrestled with this kinda thing since way back when I was fiddling with my old 8086 trying to make it do anything besides blink a cursor on a green screen good times good times

Essentially what happens when you write a multiplication in C like say `int result = a * b` is that the compiler doesn't magically make multiplication happen it actually translates that to a specific set of assembly instructions that the processor understands natively and those instructions vary slightly based on the processor architecture we're talking about x86 ARM RISC-V each has their own flavor but the core principle remains

I'll give you examples using x86 since that's what I'm most comfy with and I'm assuming you're also dealing with a fairly standard setup like maybe a Linux or Windows environment using a GCC or Clang compiler

First off a simple unsigned multiplication using 32 bit integers in C looks like this

```c
#include <stdio.h>
#include <stdint.h>

int main() {
  uint32_t a = 10;
  uint32_t b = 20;
  uint32_t result = a * b;
  printf("Result: %u\n", result);
  return 0;
}
```

Now when the compiler gets its hands on this the relevant part turns into something along the lines of this x86 assembly its a simplified version but you get the gist

```assembly
mov eax, 10       ; Move the value 10 into the EAX register (a)
mov ebx, 20      ; Move the value 20 into the EBX register (b)
mul ebx         ; Multiply EAX by EBX the result is stored in EDX:EAX
mov ecx, eax      ; Move the lower 32 bits of result from EAX into ECX which is where 'result' is held
;... some other stuff for printing and stuff ...
```

See the magic instruction `mul` there that's the workhorse of multiplication on x86 processors  the thing is though `mul` is a bit special on x86 for unsigned multiplication of 32-bit integers it actually calculates a 64 bit result and stores the lower 32 bits in the `EAX` register and the higher 32 bits in the `EDX` register We just moved the EAX value into ecx in this case which held the result. Also if you see `imul` instead of `mul` that is for signed multiplication.

Here's a more detailed example with both unsigned and signed operations and it also shows how the assembly changes based on the data types

```c
#include <stdio.h>
#include <stdint.h>

int main() {
    uint16_t a_u = 10;
    uint16_t b_u = 20;
    uint16_t result_u = a_u * b_u;

    int16_t a_s = -10;
    int16_t b_s = 20;
    int16_t result_s = a_s * b_s;

    printf("Unsigned result: %u\n", result_u);
    printf("Signed result: %d\n", result_s);

    return 0;
}
```

And here is the equivalent simplified assembly code for x86 when you compile it

```assembly
; unsigned multiplication
mov ax, 10      ; Move 10 into AX (lower 16 bits of EAX) (a_u)
mov bx, 20      ; Move 20 into BX (b_u)
mul bx          ; Multiply AX by BX result in DX:AX lower 16 bits in AX
mov cx, ax      ; Move lower 16 bits of result into CX which is where 'result_u' lives

;signed multiplication
mov ax, -10       ; Move -10 into AX (a_s)
mov bx, 20       ; Move 20 into BX (b_s)
imul bx       ; Multiply AX by BX Result is in AX
mov dx, ax      ; Move the result into DX which is where 'result_s' lives
;... other printing stuff
```

So you see how the assembly changes a little based on signed or unsigned data types for signed we used `imul` and in both cases we just moved the resulting lower bits of the result to the memory location where the result variables are stored. Sometimes the compiler might do more complicated stuff if you have say a multiplication of variables of different types it would cast them to the bigger type but the core idea remains the same in assembly land.

Now sometimes you can get more complicated with assembly optimizations the compiler might try to do shifts instead of actual multiplication for certain kinds of values if it recognizes it can do that for example multiplying by 2 is equivalent to a left bit shift ( `x << 1`) and this can be much faster. But those are compiler specific optimizations and are not part of the core of the multiplication instruction and will depend on compilation settings and the specific target architecture. I once spent 3 days pulling my hair out trying to debug assembly code that I assumed was using mul and it turns out that the compiler was actually doing bit shifts on me it was a fun time not.

And if you're looking at even bigger multiplications things get interesting when we go beyond 64bit that's a whole different kettle of fish and would involve multiple multiplication operations and addition and handling the carry bits manually but thats for a different day maybe another stack overflow question.

For a good dive into this I would check out "Computer Organization and Design: The Hardware/Software Interface" by Patterson and Hennessy that's where I learned all the low level stuff and it explains how assembly works for different architectures. Also for x86 specific stuff Intel's instruction set reference manuals are a must. You can find them on Intel's website and they are massive and it is very dry but you should give it a look if you want a full in depth view of the instructions.

And last but not least here is an example using 64-bit integers using C and then the corresponding simplified x86_64 assembly

```c
#include <stdio.h>
#include <stdint.h>

int main() {
  uint64_t a = 1000000;
  uint64_t b = 2000000;
  uint64_t result = a * b;

  printf("Result: %llu\n", result);
  return 0;
}
```

```assembly
mov rax, 1000000        ; Move 1000000 into RAX
mov rbx, 2000000      ; Move 2000000 into RBX
mul rbx             ; Multiply RAX by RBX result is in RDX:RAX
mov rcx, rax        ; Move lower 64 bits of the result to RCX (result variable)
;... other printing stuff ...
```

The x86-64 instruction set has a lot more registers than x86 so we can use `rax` `rbx` and `rcx` for our 64 bit variables and the result will be a 128 bit number where the lower 64-bits are in `rax` and the higher 64 bits are in `rdx`. Again like before with 32-bit integers we are just concerned with the lower 64 bits since the variable in C is 64 bit which we store in rcx.

So there you have it a quick rundown of C multiplication and the low level assembly that makes it work. I hope this helps you out and remember always dive deep into the docs if you want to understand something properly that is the best way to do it. Happy coding!
