---
title: "inline experiment arm assembly code?"
date: "2024-12-13"
id: "inline-experiment-arm-assembly-code"
---

Okay so inline experiment arm assembly right I've been down this rabbit hole more times than I care to admit lets break it down like we're debugging a stubborn kernel module

First off what do you even mean by inline arm assembly Are you talking about embedding assembly directly into your C or C++ code using some kind of compiler intrinsic or are you trying to inject raw machine code into a running program's memory I'm going to assume the former because the latter is a whole different level of pain and not exactly something you'd casually ask about but hey both are possible

I mean I've spent weeks debugging some really weird code paths on embedded systems back in my early career days and trust me inline assembly can be a lifesaver when you're squeezing every last bit of performance out of a resource constrained microcontroller but its also a double edged sword and you need to be damn careful with it

So we're talking about writing something like this right

```c
#include <stdio.h>

int main() {
  int a = 10;
  int b = 20;
  int result;

  asm volatile (
      "add %0, %1, %2\n"
      : "=r" (result)
      : "r" (a), "r" (b)
  );

  printf("Result: %d\n", result);
  return 0;
}

```

This is your basic hello world of inline assembly using GCC's `asm` keyword The `volatile` part tells the compiler to not be too smart and optimize away this assembly code you really do want it executed right the string "add %0 %1 %2" is the actual arm assembly instruction that adds two registers the `%0 %1 and %2` are placeholders that will be substituted with C variables like `a` `b` and `result`

The next part `: "=r" (result)` is the output constraint it tells the compiler the variable `result` is going to hold the output of the assembly instruction and it's a register variable the `=r` specifies this that it is going to be written to and it should be in a register

Then the part `: "r" (a), "r" (b)` these are the input constraints the variables `a` and `b` are inputs to the assembly again they should be loaded into registers its crucial to understand these constraints to be able to use assembly in C or C++ code you need to tell the compiler what registers to use and how it interfaces with your C variable

This will compile and run you can use `gcc yourfile.c -o yourfile` and it should work but it is not as portable as you may expect its compiler specific and target specific

Now the tricky part is experimenting with more complex stuff lets say you want to try something a little fancier like loading a value from memory using inline assembly well here is another piece of code for that

```c
#include <stdio.h>
#include <stdint.h>

int main() {
  uint32_t data[] = {0x12345678, 0x9abcdef0};
  uint32_t value;

  asm volatile (
      "ldr %0, [%1]\n"
      : "=r" (value)
      : "r" (&data[1])
  );

  printf("Value: 0x%x\n", value);
  return 0;
}

```

In this example the `ldr` instruction loads data from a memory address into a register the `[%1]` specifies that the address of `data[1]` which is `0x9abcdef0` is loaded into register which is then copied into `value`

Now if you are running on bare metal environments you may want to disable interrupts before and after the assembly operation this example is simplified for illustration purposes

```c
#include <stdio.h>

int main() {
  int my_variable = 42;

  asm volatile (
      "mrs r0, cpsr\n"   // Read CPSR into r0
      "orr r0, r0, #0x80\n"  // Set the I bit to disable interrupts
      "msr cpsr_c, r0\n"  // Write the new CPSR value back
      : /* No outputs */
      : /* No inputs */
  );

  // Your critical code here, for example the following variable access
    my_variable++;

  asm volatile (
      "mrs r0, cpsr\n"   // Read CPSR into r0
      "bic r0, r0, #0x80\n"  // Clear the I bit to enable interrupts
      "msr cpsr_c, r0\n"  // Write the new CPSR value back
      : /* No outputs */
      : /* No inputs */
  );

  printf("my variable is: %d\n", my_variable);

  return 0;
}

```

This is for disabling interrupts and its a bit more involved since it directly modifies the processor status register CPSR you should understand that accessing and modifying the status registers are very powerful but also dangerous and you can crash your system if you don't know what you are doing this is a good example of the power and pitfalls of using the inline assembly

Now here is something very important when you mess with inline assembly stuff can go sideways very fast especially when you start working with compiler optimizations and register allocation It’s easy to write code that seems right but then generates unexpected behavior due to the way compiler works So always test your code carefully especially if you are dealing with real time systems

Another thing to keep in mind is that inline assembly is not portable it ties your code directly to the target architecture and compiler This can make your code harder to maintain if you need to support multiple architectures If you need portability consider using other abstractions layers

Now resources I would recommend diving deeper into this topic is the ARM Architecture Reference Manual for the specific ARM version you're targeting it's a dry read but its the definitive source for all things ARM assembly And also for GNU Assembler you can check the documentation on GNU websites it provides all the information about how the compiler treats assembly code and the different assembly dialects

Also the book "ARM System Developer’s Guide" by Andrew N Sloss Dominic Symes and Chris Wright provides a good introduction to ARM architecture and how to program it with a good explanation of inline assembly and its implications in C and C++ systems

Finally a funny note you know debugging inline assembly is like trying to solve a jigsaw puzzle with half the pieces missing you think you've got it then bam segmentation fault the joys of low level programming

So to recap inline assembly is powerful but requires a deep understanding of both the target architecture and the compiler behavior be careful and test your code thoroughly before deploying it in critical applications and if you are new try with simple examples then move to more complex experiments only when you are comfortable with the basics.
