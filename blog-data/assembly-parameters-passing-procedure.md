---
title: "assembly parameters passing procedure?"
date: "2024-12-13"
id: "assembly-parameters-passing-procedure"
---

Okay so you're asking about assembly parameter passing right Been there done that got the t-shirt and probably a few extra grey hairs Let me lay it all out for you like I usually do on stack overflow after banging my head against the wall for hours.

First off assembly is not like your high-level languages where you just casually throw parameters into a function call and it all magically works No sirree bob. In assembly you are down in the trenches dealing with registers memory locations and the raw instruction set architecture of the processor itself Its beautiful chaos if you know what I mean.

Let's talk parameter passing specifically In general there are several ways we do it and the exact method depends on the specific architecture like x86 x64 ARM etc and also on the calling convention which is the agreed-upon standard for how functions pass arguments. It’s like agreeing on which side of the road to drive on you don't want to end up in a collision.

One of the most common methods especially with older 32-bit systems is to use the stack. It is a last-in first-out data structure. Before the function is called the caller pushes the arguments onto the stack in reverse order If you have function called foo(int a int b int c) the stack would look something like c then b then a at the time of call and this is something you will probably see a lot in older code bases. The called function then retrieves the parameters off the stack usually using offsets relative to the stack pointer register esp in x86. Keep in mind this is a very simplified view and modern compilers do tons of optimisations to the code.

Here is an example of how this might look in x86 assembly it’s super simplified though

```assembly
;Caller code
push 3 ;Push argument c
push 2 ;Push argument b
push 1 ;Push argument a
call foo ;Call the function foo
add esp 12 ;Clean up the stack after the call

;foo function code
foo:
push ebp ;Standard prologue save the base pointer
mov ebp esp ;Set current base pointer
mov eax [ebp + 8] ;Get the value of a from the stack
mov ebx [ebp + 12] ;Get the value of b from the stack
mov ecx [ebp + 16] ;Get the value of c from the stack
; Do some calculation with a b and c
mov esp ebp ;Standard epilogue
pop ebp ;Restore the original base pointer
ret ;Return
```

As you can see in the first snippet the caller code is pushing the parameters 3 2 1 in reverse order and then calls foo. In the second snippet we have the foo function.

A key thing to remember here is how the offsets work. The stack pointer esp points to the top of the stack so if we want to access the parameters we have to access the stack pointer plus the offsets and this will lead us to the first pushed parameters. In the case of foo function the parameters a b c are respectively 8 12 and 16 offsets from ebp. This is because there is also return address that is also pushed in to the stack when we call foo with the instruction `call foo`. It is also important to set and restore the base pointer and stack pointer in each function call.

Another very common method especially in modern architectures like x64 and arm is to use registers. Registers are super fast on-chip memory locations that are used to perform calculations and access data. Usually the first few parameters are passed directly through the registers. This is way faster than going to the memory stack and hence the compiler try to optimise as much as it can in that way. The x64 calling convention for example uses registers like rdi rsi rdx rcx r8 r9 for the first six integer or pointer type arguments and any additional arguments are usually pushed onto the stack. We use a different set of registers in other calling conventions such as the ARM64 which uses x0 to x7 for the first eight parameters.

Here is a simplified example of parameter passing using registers in x64

```assembly
;Caller code
mov rdi 1 ;Move the first parameter a into rdi
mov rsi 2 ;Move the second parameter b into rsi
mov rdx 3 ;Move the third parameter c into rdx
call foo; Call the function foo
;No stack cleanup needed for integer parameters

;foo function code
foo:
mov rax rdi; Copy a from rdi to rax
mov rbx rsi; Copy b from rsi to rbx
mov rcx rdx; Copy c from rdx to rcx
; Do some calculation using rax rbx rcx which stores a b and c respectively
ret ; Return
```

See the difference from the stack method? No push and pop this is way faster if we have enough registers to pass our parameters. Keep in mind that the usage of registers is also defined by the calling convention of your processor so it can slightly vary based on different OS and processors.

Now a little detail which might cause some headache sometimes the size of the parameter matters. If you have a 64-bit parameter you'll usually need a 64-bit register on x64 systems such as rax or rdi or you might need to push 8 bytes into the stack. If you have a floating point number for instance you may need to use another type of registers such as xmm0-xmm7 on x64 or other floating point registers available on your target machine. You need to consult your architecture manual to understand this well.

There is also a situation with structures where instead of just pushing the whole struct onto the stack the structure is passed by copy or by reference. Copying means that the compiler creates a copy of the structure in the stack or registers which makes the function to operate on the copy rather than the original structure. When you pass by reference the address of the structure is passed which is usually a pointer so that the function can operate on the original data. Copying struct might be more computationally expensive so passing them by reference might be preferred in some instances. This can vary a lot by architecture and compiler.

Now you might ask if all of this was complex so what should I do. Don't you worry you don't usually write raw assembly code anymore not unless you are working in a very specific niche. It’s more about understanding what’s going on under the hood when the compiler does its thing. Compilers today are extremely advanced and they will automatically optimise your code to use the most efficient method of parameter passing. You will mostly deal with it when writing a kernel or low level drivers or when trying to reverse engineer code. Its very useful to know what is happening but you probably wont code your whole program in assembly. However you need a proper grounding to understand these kind of stuff.

This is not all there is a more advanced topic such as calling conventions and how to call c or c++ functions from assembly or vice versa which is another beast on its own. There is also the issue with calling other assembly functions that might not be using same calling conventions so it can cause some problems if you are not careful. If that happens to you and your program start behaving weirdly that will probably be the problem.

I once spent a whole week debugging a problem where I was accidentally mixing calling conventions It was not a pleasant week to say the least and after several late nights I realised the culprit was that I was not saving registers which would mean my registers would be overwritten. This was back in my early days and I was pulling out my hair at the time. I was quite green back then. It feels like ages ago. You live you learn haha.

I would definitely recommend checking out the Intel architecture manual if you are working with x86 or x64. They have tons of useful details about registers calling conventions and low level hardware information in them. Same goes for ARM architecture manual that is also available online for free. Also the book "Computer Organization and Design" by Patterson and Hennessy is quite good to understanding the basics and if you wanna deep dive into compiler stuff the dragon book the famous "Compilers Principles Techniques and Tools" by Aho Lam Sethi and Ullman is a must read.

So there you have it parameter passing in assembly it’s complex but it makes you a better programmer after understanding it. It is important to understand that different processor architectures may have different approaches and the calling convention also plays a significant role. It’s all about the details.

Remember understanding these low level details helps to write better higher level code.
