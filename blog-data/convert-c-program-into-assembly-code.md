---
title: "convert c program into assembly code?"
date: "2024-12-13"
id: "convert-c-program-into-assembly-code"
---

Okay so you want to convert C code to assembly huh Been there done that a million times It’s a rite of passage for anyone who gets serious about systems level stuff I remember back in uni we had this professor old school guy total wizard with low level but terrible with anything resembling user interfaces He made us write entire operating systems in assembly using C only for basic bootstrapping talk about a crash course in the reality of computer architectures

First thing’s first understand that assembly isn't some magic black box it's just a human readable representation of machine code Machine code is what the CPU actually executes assembly is our way of talking to it without using binary strings Remember those days trying to punch in binary instructions on punch cards good times no actually horrible times I try to block that out

The conversion process you are talking about is called compilation and specifically the part you are interested in is usually done by a compiler and assembler working together The C compiler transforms your high level C into something called intermediate representation IR or assembly code itself depending on the flags you use An assembler then takes the assembly and transforms it into machine code which is executable by your CPU

Now the specific assembly dialect you end up with depends heavily on your target architecture Are you on x86 x64 ARM RISC-V you get the idea Each architecture has its own instruction set and therefore its own assembly language This can be painful but also enlightening When I was doing some reverse engineering a while back I spent days tracing through x86 assembly trying to figure out some obscure malware function that was fun not really though but educational

Okay lets get practical Let’s say we have this simple C program

```c
#include <stdio.h>

int main() {
  int x = 10;
  int y = 20;
  int sum = x + y;
  printf("The sum is: %d\n", sum);
  return 0;
}
```
Pretty basic right we are adding two numbers and printing the sum to the console

Now let's see how to get assembly code from that using GCC the common compiler I'd use First compile with the `-S` flag and the `-o` flag to tell it to produce a `sum.s` file

```bash
gcc -S sum.c -o sum.s
```

This produces an assembly file `sum.s` If you open it you'll likely see some boilerplate stuff plus the assembly instructions generated for your C program The generated assembly output is usually quite verbose with lots of compiler optimizations stuff that might make your head spin at first

 Here is an example of the relevant x86 64 bit assembly code that will be part of the `sum.s` file you just compiled

```assembly
  movl  $10, -4(%rbp)    # Store the integer 10 to the variable x
  movl  $20, -8(%rbp)    # Store the integer 20 to the variable y
  movl  -4(%rbp), %eax  # Move the value of x into the eax register
  addl  -8(%rbp), %eax  # Add the value of y to the value in eax
  movl  %eax, -12(%rbp) # Store the result (in eax) to the variable sum

  movl  -12(%rbp), %esi # move the sum to the register for printf
  movl  $.LC0, %edi
  movl  $0x0, %eax
  call  printf
```

Okay so what is going on here We have `movl` for moving data `addl` for adding The  `%rbp` register represents the base pointer for the stack `eax` `esi` are general purpose registers used to perform computations and move arguments to and from function calls  and the strange `-4(%rbp)`  `-8(%rbp)`  `-12(%rbp)` are stack memory locations allocated for our local variables x y and sum respectively `$.LC0` is the string literal for printf. In short this low-level stuff is managing and manipulating data directly in memory and registers It's very close to the hardware and that's where the true understanding of how computers operate happens

Now you might be thinking oh great I have to hand write this garbage every time I want to write a simple C program No thankfully no The compiler handles all of this for us That’s why they make the big bucks but I digress Understanding assembly is invaluable for debugging performance analysis reverse engineering and getting a more profound sense of what your high-level code is doing under the hood Also sometimes when you are debugging a low-level memory access fault it will come up in the assembly this is something every system programmer needs to be very comfortable with

Lets look at another example say we have some pointer stuff in C

```c
#include <stdio.h>

int main() {
  int numbers[] = {1, 2, 3, 4, 5};
  int *ptr = numbers;

  for (int i = 0; i < 5; i++) {
    printf("%d ", *ptr);
    ptr++;
  }
  printf("\n");
  return 0;
}
```

This is simple array and pointer access Now lets see the corresponding x86 64 bit assembly

```assembly
  movl  $1, -20(%rbp)    # array[0] = 1
  movl  $2, -16(%rbp)    # array[1] = 2
  movl  $3, -12(%rbp)    # array[2] = 3
  movl  $4, -8(%rbp)     # array[3] = 4
  movl  $5, -4(%rbp)     # array[4] = 5
  leaq  -20(%rbp), %rax # load the address of array[0] to rax this will be pointer to start of array
  movq  %rax, -24(%rbp) # copy address in rax to a pointer in the stack
  movl  $0, -28(%rbp)    # initialize i = 0

.L3: # Start of the loop
  movq -24(%rbp), %rax    # load pointer to array element to rax
  movl (%rax), %esi       # dereference the pointer to get value to esi register
  movl $.LC0, %edi       # format string
  movl $0x0, %eax
  call  printf             # call printf
  addq  $4, -24(%rbp)     # add 4 to the pointer ie pointer++
  addl $1, -28(%rbp)      # add 1 to the i loop variable
  cmpl $4, -28(%rbp)       # compare i to 4
  jle .L3 # jump if i<=4
```

Here `leaq` is used to load the address of the first element of the array into a register which we then store at the memory location where the `ptr` is stored. Then inside the loop you can see the address is fetched then dereferenced using `(%rax)` to get the actual data that pointer is pointing at and finally the pointer is incremented by 4. It’s all memory manipulation at this point

One more example a simple function call and return

```c
#include <stdio.h>

int add(int a, int b) {
  return a + b;
}

int main() {
  int x = 5;
  int y = 7;
  int result = add(x, y);
  printf("The result is: %d\n", result);
  return 0;
}
```

And here the relevant x86_64 assembly

```assembly
  movl  $5, -4(%rbp)      # x = 5
  movl  $7, -8(%rbp)      # y = 7
  movl  -4(%rbp), %eax   # Move the value of x into the eax register
  movl  -8(%rbp), %edx   # Move the value of y into the edx register
  movl  %edx, %esi        # y is moved to esi as second parameter
  movl  %eax, %edi        # x is moved to edi as first parameter
  call  add              # call function
  movl %eax, -12(%rbp)   # store result of add in register at stack

  movl  -12(%rbp), %esi  # result is moved to esi for print
  movl  $.LC0, %edi
  movl  $0x0, %eax
  call  printf

.LFE1:
  .size add, .-add
  .globl add
  .type  add, @function
add:
.LFB2:
  .cfi_startproc
  pushq %rbp    # Save base pointer
  .cfi_def_cfa_offset 16
  .cfi_offset 6, -16
  movq %rsp, %rbp # Make rbp current stack point
  .cfi_def_cfa_register 6
  movl %edi, -4(%rbp) # move first parameter to local variable a
  movl %esi, -8(%rbp) # move second parameter to local variable b
  movl -4(%rbp), %eax  # move a to eax
  addl -8(%rbp), %eax  # add b to eax so result of sum
  popq %rbp # restore base pointer
  .cfi_def_cfa 7, 8
  ret # return from function
```

Here the arguments to the `add` function are placed in the `edi` `esi` registers The function call is done via `call add` and the return value is placed into the `eax` register This is a common way of passing arguments and return values in assembly you will see more of it the more you dive deep

Now you're probably wondering how all this works with different instruction sets How the heck does ARM assembly look like or what is this RISC-V thing I heard about

For x86 Intel’s documentation is a good place to start I remember spending way too many hours in their manuals but the documentation is the best you can get. For ARM the ARM architecture reference manuals are the definitive guide. RISC-V has its specification documentation which is also freely available All these resources will give you a full deep dive into their instruction sets, addressing modes and more and of course if you are really serious a book like Computer Organization and Design by Patterson and Hennessy is an absolute must have a bible for any system programmer

Remember assembly programming is not about writing every single program in it It's about understanding how your code interacts with the processor It helps you write more efficient code debug issues at the lowest level and lets you understand the limitations of the architecture which is very important it also helps you reverse engineer things (sometimes even if that thing is just that code you wrote 6 months ago and forgot how that worked)

Oh and for those who say assembly is dead well I will point out its like saying you do not need to know basic arithmetic to perform rocket science its always good to know how to add 2 + 2 because at some point you will need that kind of math in the lower level part of the rocket science. There is no magic to it just a lot of logic and understanding the hardware
