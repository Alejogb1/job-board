---
title: "why i do get cannot find bound of current function when i overwrite the ret ad?"
date: "2024-12-13"
id: "why-i-do-get-cannot-find-bound-of-current-function-when-i-overwrite-the-ret-ad"
---

Okay so you're hitting a classic and frankly annoying issue you're messing with the return address on the stack and getting tripped up by the compiler or maybe even the hardware itself let me break down why that's happening and how I've personally battled this demon over the years

So first off the “cannot find bound of current function” message you’re seeing that's essentially the system’s way of saying "hey I have no freaking idea where I’m supposed to go back to after this function finishes" It’s like trying to use a GPS but the starting point is nowhere and the ending point is a random pile of bits You've overwritten the crucial piece of data the return address

Now lets dig a bit into why this is happening and how I've seen it go down I've been down this rabbit hole more times than I care to remember especially back in my early days of trying to do some low level exploit work or even just dabbling with custom stack frames before we had better debugging tools

Basically when a function is called the processor does a few things It pushes the current instruction pointer which is the address of the instruction that’s about to happen onto the stack as the return address so when the function is done the processor knows where to continue executing the program from This return address is absolutely vital its the breadcrumb trail that keeps your program’s execution order correct

Now when you overwrite this return address well all hell breaks loose The program is gonna try to return to some completely random spot in memory and that's where you get segmentation faults crashes and that lovely "cannot find bound" error your debugging system whether it’s a debugger or just the operating system is trying to unwind the stack to find where it was supposed to be coming from and its looking at a corrupted return address so it's completely lost

Okay code examples lets look at a super simplified view I'm not gonna dive into actual stack frames here cause that’s way more platform specific but I will show how the principle looks

**Code Example 1: Simple C function with return address overwrite**

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void vulnerable_function(char *input) {
 char buffer[10];
  strcpy(buffer, input); // BAD practice buffer overflow waiting to happen
  // Now imagine buffer[10] being the return address itself you overwrote it
}

int main() {
 char *attacker_input = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
 vulnerable_function(attacker_input);
 printf("Shouldn't get here");
 return 0;
}
```

Here we've got a classic buffer overflow strcpy doesn't know when to stop copying so if your `input` is bigger than buffer it starts overwriting the stack beyond `buffer` and importantly this can overwrite the return address. This can lead to very unpredictable behavior and usually a crash at the function return if the overwritten address is invalid

Okay thats the classic example but its not always just a buffer overflow sometimes you might be manually manipulating the stack for some reason or another like in assembly code

**Code Example 2: Assembly example of manual stack manipulation and return address overwrite (x86_64 like)**

```assembly
section .text
 global _start

_start:
  ; setup stack frame
  push rbp
  mov rbp rsp
  sub rsp 16    ; space for local variable (not used here in this case but needed for alignment)
  
  ; intentionally overwrite return address
  mov rax, 0x4141414141414141  ; Some random address lets pretend "A"
  mov [rbp+8], rax ; this overwrites return address in this simplified example

  ; Pretend we did some work with the stack
  ; and reach function return
  leave
  ret
  
```

Here we’re directly manipulating the stack setting a specific address to overwrite the return address and this code is simplified its for demonstration and yes it crashes as soon as that return instruction executes

I've seen this done before to intentionally redirect program execution usually as part of a more complex exploit but in that case its being done to jump to other function within memory

Now sometimes this isn't a direct overwrite its a result of stack corruption issues I remember once I spent a whole weekend tracking down a problem where a complex data structure on the stack had a padding issue meaning some random memory writes were happening close to the return address corrupting it by accident which leads to a crash after the function returns its important to pay attention to structure packing sometimes that can be a killer

And lastly sometimes this isn't about you overwriting it yourself it can also happen if you've got a compiler bug or bad optimization settings that are moving things around in a weird way I’ve had one or two instances where aggressive compiler optimizations introduced stack alignment issues and corrupted return addresses that way This is where low level debugging and actually watching the stack layout comes into play

**Code Example 3: C++ with explicit stack frame control through assembly and its corruption**

```cpp
#include <iostream>

extern "C" void my_assembly_function(void *target_address);

void cpp_function(int x) {
    std::cout << "Inside cpp_function before asm" << std::endl;
    void *return_address;
    asm(
        "movq (%%rbp), %0;" // get return address in this simple example
        : "=r"(return_address)
        :
    );
    std::cout << "Original Return address" << return_address << std::endl;
    my_assembly_function(nullptr); // Pass null address to be overwritten

    std::cout << "This wont print because ret address is probably overwritten" << std::endl;

}

extern "C" void my_assembly_function(void *target_address){

   asm volatile(
        "movq %%rbp, %%rax;" // save rbp in rax
        "movq 8(%%rax), %%rcx;" // get return address in rcx
        "movq %0, 8(%%rax);" // overwrite the return address with null (or another address)
        : // no output registers
        : "r"(target_address) // inputs
        : "rax","rcx" // clobbered register

   );
   //return address will be overwritten in here
}


int main() {
    cpp_function(10);
    std::cout << "This will not be called as ret address is overwritten in cpp_function" << std::endl;
    return 0;
}
```

Here we’re getting the return address in cpp_function then passing a null pointer to my_assembly_function which we are then using to overwrite the return address again. So after my_assembly_function returns then the program will jump to null and crash. This can be used to jump to malicious code and also this demonstrates how the address can be overwritten

Now what to do about it and to avoid these issues in the future

First thing use a good debugger like gdb or lldb set breakpoints step through code and watch the stack it’s the only way to see exactly what’s going on This means really knowing how stack frames work on your specific architecture x86 ARM MIPS whatever

Second practice safe coding don't use `strcpy` `gets` or other functions that don't check buffer boundaries use `strncpy` `fgets` or even better use safer alternatives from libraries like `std::string` from C++

Third always be careful when manipulating assembly directly that means you really have to know stack frame layouts and register usage or otherwise it will be a wild ride

Fourth check your compiler settings aggressive optimization sometimes leads to unpredictable issues so testing and being careful helps a lot

For resources I always recommend “Modern Operating Systems” by Andrew S Tanenbaum for a solid grounding on operating system principles including how stacks work at a system level also “Computer Organization and Design” by David Patterson and John Hennessy gives a good view on the low level architecture that dictates stack behaviour and for more detailed platform specific stuff you can dive into the Intel manual for x86 or ARM architecture manuals all of these resources are dense but they give you a rock-solid foundation to troubleshoot these types of problems
Remember one time a colleague spent 2 days tracking down this sort of issue and it turns out the return address overwrite was caused by a single bit flip in memory due to a faulty ram chip so yeah not every problem is in the code sometimes it’s just plain hardware acting up so in conclusion keep a calm mind do not panic you can do this and remember the stack is your friend (until it isn't that's the joke)
