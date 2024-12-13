---
title: "convert c code to assembly language?"
date: "2024-12-13"
id: "convert-c-code-to-assembly-language"
---

Okay so you want to go from C to assembly eh I get it I've been there plenty of times It's like looking at a high-level description of a building and wanting to see the nitty-gritty blueprints the actual nuts and bolts stuff I mean you can read the manual and know what the function does but sometimes you want to know *how* it does it right I remember back in my early days struggling with this exact thing I was working on this embedded system project something with a tiny microcontroller and I needed to optimize the heck out of the code for speed and power Turns out the C compiler was doing some weird things that were far from optimal so I had to dive deep into assembly to see where the bottlenecks were and rewrite parts of it by hand It's a rite of passage for any serious low-level programmer

So let’s start with the basic premise C code is essentially a set of instructions that a computer needs to execute but those instructions aren’t what the hardware actually understands They need to be translated to a language the processor can directly understand and that's assembly language Assembly is low-level language directly corresponding to processor instructions Its like a middleman its closer to the hardware than C ever will be Its a human readable representation of machine code and different processors have different assembly languages ARM x86 MIPS etc Each assembly instruction typically corresponds to a single machine instruction which makes it good for understanding what the processor is doing at a very detailed level

Now the question is how do you actually translate C to assembly Right there are a few ways

First the easiest approach is to let the compiler do the heavy lifting Almost all C compilers like GCC or Clang can produce assembly code for you as an intermediary step They do that every time you compile but you normally dont see this output you have to ask nicely for it Now for example if you are using gcc there's a flag for that -S it generates an assembly file and you can see it if you add that flag

```c
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
  int x = 5;
  int y = 10;
  int result = add(x, y);
  printf("The sum is %d\n", result);
  return 0;
}
```

If you save this as `test.c` then you can get its assembly with command `gcc -S test.c` then you get `test.s` in the same directory If you open that file you'll find an assembly output like this. And if you compile for ARM instead then you get an assembly file that is completely different showing what the underlying target architecture is doing. Now this specific example will vary depending on your architecture and compiler but its a good start to see how functions and variables map to assembly instructions

```assembly
	.file	"test.c"
	.text
	.globl	add
	.type	add, @function
add:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	addl	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	add, .-add
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	$5, -4(%rbp)
	movl	$10, -8(%rbp)
	movl	-8(%rbp), %esi
	movl	-4(%rbp), %edi
	call	add
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %esi
	leaq	.LC0(%rip), %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
.LC0:
	.string	"The sum is %d\n"
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
```
Notice how the C code is broken down into assembly instructions such as `pushq`, `movl`, `addl` etc These instructions directly manipulate the processor's registers and stack In the `add` function you can see how the two arguments are moved from the register to the stack the addition is done then the result is moved to the return register. The main function also does the same pushes the data to stack calls add and saves the return value before passing everything to printf

Now a second method is for more advanced stuff when you want to be able to compile to assembly as part of your build process. Now lets say you have a simple file `helper.c`

```c
int increment(int x) {
    return x + 1;
}
```
and you want to compile it to an object file without linking it yet

then the command `gcc -c helper.c -o helper.o` will do the trick and you can examine the assembly generated inside object file with command `objdump -d helper.o` which yields an output similar to this. It will be very similar to what you get with `-S` but its stored inside the object file itself which you later can use with your linker

```assembly
helper.o:     file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <increment>:
   0:	55                   	push   %rbp
   1:	48 89 e5             	mov    %rsp,%rbp
   4:	89 7d fc             	mov    %edi,-0x4(%rbp)
   7:	8b 45 fc             	mov    -0x4(%rbp),%eax
   a:	83 c0 01             	add    $0x1,%eax
   d:	5d                   	pop    %rbp
   e:	c3                   	ret    
```
You can also use compilers from different vendors like intel's icc or Microsoft's cl Each have different options that accomplish roughly the same purpose of inspecting generated assembly output. In the case of Microsoft cl you can use the `/Fa` flag and similarly the corresponding commands to view the output can differ

Now these first two methods rely on a compiler to generate the assembly for you. But lets say that you want to translate the assembly by hand because you want to become truly enlightened in low-level programming This is the hard way but its the most rewarding and it will let you really understand how a processor actually works

You’ll need to carefully think about memory layout register usage and the target processor instruction set. I remember working on an old processor and having to write a matrix multiplication entirely by hand in assembly. I spent days staring at manuals and instruction sets debugging instruction by instruction It wasn’t pretty but it was a very humbling experience

Now let's do an example to understand the low-level nature better Lets say you want to write a function that does a simple bit manipulation for a specific processor the function is just setting the nth bit of some integer and we will not use a compiler at all rather we will write the assembly directly.

Here is how the C version would look like

```c
int set_bit(int value, int bit_number) {
    return value | (1 << bit_number);
}
```

And here’s how it could look like in an x86 assembly using a tool like an assembler like nasm on a Linux-like system to see the output

```assembly
section .text
    global set_bit
set_bit:
    push rbp
    mov rbp, rsp
    mov edi, [rbp+16]  ; value is the first argument moved to edi register
    mov esi, [rbp+24]  ; bit_number is the second argument moved to esi register
    mov eax, 1         ; eax will hold the bit value that we want to set
    mov ecx, esi       ; now move esi into ecx because we are going to shift it
    shl eax, cl        ; this is the actual left shift instruction
    or edi, eax        ; now we OR the original number with the shifted value
    mov eax, edi       ; move the result to eax which is where function will return
    pop rbp
    ret
```
As you can see its a very different beast than the C code directly You can see that registers are involved in computations and memory locations are accessed explicitly This level of detail is useful for performance tuning or working with very specific hardware

Now for learning materials on this I would recommend *Computer Organization and Design: The Hardware/Software Interface* by David Patterson and John Hennessy it gives great detailed insight into all computer architecture concepts and hardware operations. Also for deeper assembly language specifics there are many books depending on your target architecture like for example *Assembly Language for x84 Processors* by Kip Irvine for x86 and for ARM specific processor architecture manuals directly from ARM

I think that should be more than enough to get you started converting C code to assembly or at least understanding better how to do it. Now if you excuse me I need to go back to debugging this race condition it’s getting personal you know like when two threads try to access the same data and things start going haywire its like a dance-off between threads and nobody can agree on who goes first or when its their turn you get me? Good luck and have fun playing with assembly remember the key is practice practice practice.
