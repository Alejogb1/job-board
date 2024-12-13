---
title: "shadow space assembly code explanation?"
date: "2024-12-13"
id: "shadow-space-assembly-code-explanation"
---

Okay so shadow space assembly code huh Been there done that Got the t-shirt and probably a few bug reports logged against my name because of it Let me lay it all out for you from a purely practical perspective like I've seen this play out a bunch of times

First off understand what we mean by shadow space in assembly context Think of it as a designated area on the stack Specifically it’s space reserved by the calling function for the callee to use for passing arguments not through registers This is particularly vital in systems following calling conventions like the System V AMD64 ABI where the first six integer arguments are typically passed through registers rdi rsi rdx rcx r8 and r9 Any additional arguments or arguments that are too large to fit into a register get pushed onto the stack but not quite at the top they are pushed into this shadow space

Now why this whole shadow space thing exists is because compilers don’t want to assume that registers will be preserved across function calls This is a safe approach I had the misfortune of working on a system where someone thought they were clever with register usage across function boundaries it became a debugging nightmare it made me wanna pull my hairs and I have little to start with So they give the called function this space to work with its own little private playground on the stack for arguments that registers can't handle or when they're just plain out of registers

Let's get to the concrete stuff I remember once debugging a particularly annoying issue on a embedded linux platform It wasn't a seg fault more like a weird value corruption when passing structs I was pulling my hairs for hours before i even thought to look at the assembly and realize the compiler was using this shadow space and there was a misunderstanding about struct alignment and size between the calling and the called functions The called function assumed a larger struct than what the caller actually passed on the stack causing this corruption.

The usual culprits when you see shadow space issues are ABI mismatches incorrect stack pointer management or just plain misunderstanding of calling conventions And trust me I’ve seen more than my fair share of each of those Usually if it works it's because of luck and usually that luck does not last long so always good to explicitly check this stuff with debugging in assembly level.

Here's a quick snippet of what you might see in assembly code when dealing with shadow space This is very simplified assume its a 64-bit x86 system with a call convention similar to System V AMD64. I'm not gonna write out the full prologue or epilogue just the interesting bit where arguments are passed on stack

```assembly
; Caller function
mov rdi, 1   ; First argument
mov rsi, 2   ; Second argument
mov rdx, 3   ; Third argument
mov rcx, 4   ; Fourth argument
mov r8, 5   ; Fifth argument
mov r9, 6   ; Sixth argument
mov [rsp - 8] , 7  ; Seventh argument on shadow space
mov [rsp - 16], 8 ; Eighth argument on shadow space
sub rsp 16 ; Allocate shadow space
call my_function ; Function call
add rsp, 16 ; Deallocate the shadow space
```

Here the stack pointer rsp is decremented to allocate shadow space *before* the call and incremented back after return This is crucial the called function expects this space and not decrementing or not incrementing it accordingly leads to all sorts of chaos I wish people would follow conventions more closely its always easier to read correct code then debugging incorrect code.

And here is how the called function might use the passed arguments

```assembly
; Called function (my_function)
; Assume the function uses the registers
mov rax, rdi ; use the first argument
add rax, rsi; use the second argument
add rax, rdx; use the third argument
add rax, rcx; use the fourth argument
add rax, r8; use the fifth argument
add rax, r9; use the sixth argument
mov rbx, [rsp + 8]; move from shadow space the 7th argument
add rax, rbx; do something with the 7th argument
mov rbx, [rsp + 16]; move from shadow space the 8th argument
add rax, rbx; do something with the 8th argument
; Function logic here using rax and other registers
ret ; Return
```

Notice how the called function accesses the stack using offsets from rsp It’s not using negative offsets which would have happened if shadow space wasn't allocated prior to the function call These offsets are relative to the stack pointer at the start of the function which is exactly where we put the extra args in the calling function before the call instruction

A common gotcha I've seen a lot is when a function expects arguments passed on the stack but the caller function fails to allocate the space or allocates it incorrectly It's like expecting a delivery but having the delivery driver not have the correct address or not having allocated space for all packages it's a disaster waiting to happen This is very typical for systems which mix different architectures or different operating systems or with different compilers or with different versions of same compilers.

Now if you are having trouble debugging this stuff and not sure what the compiler is doing this is another trick i use often: Force the compiler to pass everything through stack or disable register-based argument passing by changing compiler flags this can force everything onto the stack, making it easier to trace what's going on at the cost of performance But for debugging its ok. It will be slower sure but this will save you many hours of debugging.

Here's how you can do it on gcc

```bash
gcc -mno-red-zone -fno-omit-frame-pointer your_code.c -S -o your_code.s
```

The `-mno-red-zone` and `-fno-omit-frame-pointer` options are also useful to get a more detailed view of stack usage and frame setup Also `-S` will save the assembly code in a `your_code.s` file for you to analyze I use this all the time to check what the compiler is doing or not doing.

Some resources I would strongly suggest you check out are the System V ABI documentation for your specific architecture It’s the bible on calling conventions And if you like a more hands on approach the book "Programming from the Ground Up" by Jonathan Bartlett is pretty good to learn assembly and how the stack works. Also check any architecture specific manuals from Intel or AMD or Arm. They will provide details about their instruction set including the registers and calling conventions for those architectures

Understanding shadow space really comes down to understanding how the stack is used during function calls It's one of those low-level things you have to grasp to avoid weird heisenbugs and understand compiler's decision making process. You don't need to become an assembly guru but having this basic knowledge will help you avoid lots of problems and you will be a far better programmer because of it.

Oh and one last thing if you get really lost just remember the golden rule of debugging “always check your assumptions” especially when dealing with low level stuff like shadow space and calling conventions It's so true it's almost like a bad joke right haha.
