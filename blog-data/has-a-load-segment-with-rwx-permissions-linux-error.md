---
title: "has a load segment with rwx permissions linux error?"
date: "2024-12-13"
id: "has-a-load-segment-with-rwx-permissions-linux-error"
---

Okay so you're seeing a load segment with rwx permissions on Linux huh been there done that so many times it's practically a rite of passage for a kernel dev or someone messing with low-level stuff like me Let's break this down real quick.

First off rwx permissions that's read write execute all in one place not exactly something you see every day on purpose Usually we're talking about code or data segments and they have stricter permissions. Code generally has rx and data usually rw or just r but rwx is kinda a red flag. The kernel doesn't like it because it's basically an open invitation for all sorts of security problems if you’re dealing with untrusted input. Think of it as a hacker's dream basically.

Now how does this even happen? Well plenty of ways let me tell you from my own experiences with this. The most common scenario I've seen is when a poorly written or malicious program or a badly configured linker does something funny with memory mapping. You might see this when you are creating or loading dynamically linked libraries or shared objects or when the program tries to mmap a file with improper flags or maybe when you’re fiddling with custom memory allocators that are not fully thought through.

A common instance is a vulnerability i came across a while back, a program was using a custom allocator and it wasn't being too careful about how it was setting up memory. It ended up allocating a memory region using mmap with the wrong protection flags like PROT_READ | PROT_WRITE | PROT_EXEC. Then for some reason the program decided to store the data and then later tried to execute the data as code yeah i know that's stupid. Well, that was a problem because a malicious user could inject some code into that region and get it executed. I spent a solid two days debugging this memory corruption problem until i found it. It was a really long debug session with gdb if i remember correctly.

Also, another classic I faced was in a build system, some build scripts weren't setting the proper segment flags during the linking process. So, some parts of the executable ended up with rwx instead of rx. These build issues are sometimes really hard to catch since most of the build systems these days use macros and templating to generate linker scripts and sometimes the output isn’t what you were expecting.

What about shared libraries? Oh yes, that was fun too. It wasn't code I was actually responsible for. Some shared library was trying to be "clever" and using a memory region for both data and code, mmap-ing the segment as rwx and then doing some on the fly code manipulation thing for reasons I still can’t figure out. You might find that one in some legacy C++ code or some embedded systems code sometimes, or sometimes some really old python packages that uses shared libraries underneath.

So yeah, this is usually not intended behavior. If you're seeing this consistently it's almost certainly a bug somewhere. If you are an Operating system kernel developer this is something that you might have seen at least once. Or if you were a student doing some low level programing or if you were like me just a poor guy who just wanted to build something with c and got into this mess.

Okay let's see some code examples here to illustrate.

First a simple example how *not* to do it:

```c
#include <sys/mman.h>
#include <stdio.h>
#include <string.h>

int main() {
    size_t size = 1024;
    void *mem = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap failed");
        return 1;
    }

    char* code = (char *)mem;
    strcpy(code, "\x48\xb8\x01\x00\x00\x00\x00\x00\x00\x00\xff\xe0");
    // this is "mov rax, 1; jmp rax"
    ((void(*)())mem)();

    munmap(mem, size);
    return 0;
}
```

This code maps a memory region with rwx permissions, copies some machine code into it, and then attempts to execute it. This is an example of what we’re trying to avoid.

Here is another example more complex one in a shared object

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <dlfcn.h>

void* execute_jit_code(const char* code, size_t code_len){
    size_t page_size = sysconf(_SC_PAGE_SIZE);
    size_t aligned_len = (code_len + page_size - 1) & ~(page_size-1);

    void *mem = mmap(NULL, aligned_len, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap failed");
        return NULL;
    }

    memcpy(mem, code, code_len);

    return mem;
}

void cleanup_jit_code(void* mem, size_t code_len){
    size_t page_size = sysconf(_SC_PAGE_SIZE);
    size_t aligned_len = (code_len + page_size - 1) & ~(page_size-1);
    munmap(mem, aligned_len);
}

int add(int a, int b) {
    const char code[] = {
        0x48, 0x89, 0xf8,       // mov    rax,rdi
        0x48, 0x01, 0xf0,       // add    rax,rsi
        0xc3                    // ret
    };
     void* jit_code = execute_jit_code(code, sizeof(code));
    if(jit_code == NULL){
      return 0;
    }

    int (*func)(int, int) = (int(*)(int, int))jit_code;
    int result = func(a, b);
    cleanup_jit_code(jit_code, sizeof(code));

    return result;
}

```
This example code allocates a rwx buffer and copies some x86 assembly into it and then executes it, and then cleans up the memory. In normal situations the shared object should have the code in a read only execute memory region and data in read write region.

Now, a more acceptable way to do things:

```c
#include <sys/mman.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main() {
    size_t size = 1024;
    size_t page_size = sysconf(_SC_PAGE_SIZE);
    void *mem = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap failed");
        return 1;
    }

    char* code = (char *)mem;
   strcpy(code, "This is some text data");

    if (mprotect(mem, size, PROT_READ) == -1) {
        perror("mprotect failed");
	munmap(mem, size);
        return 1;
    }


    printf("%s\n", (char*)mem);

    munmap(mem, size);
    return 0;
}

```

This code maps a memory region with read write permissions and copies some data, then changes the permission to read only with `mprotect`. Then reads the data and unmaps it. This is a safer and more standard practice. Now, if I try to execute this code as a shellcode as the previous example it will crash due to a segmentation fault.

So how to fix the rwx problem you see? First and most important thing is to find where this is happening. If it is your code or a program you’re developing then you need to check your memory mappings and linker configurations. Check all your mmap calls and all of your dynamic loading of libraries and memory region allocations. If it is a third party library or a program that is exhibiting this behavior then you need to file a bug to the vendor or the developers. If the memory is not meant to be executed then you should use mprotect to remove the execute permission.

Now, what to learn more on the topic you might ask. Well if you’re serious about this stuff, I highly suggest you do some reading on Operating system concepts, and memory management in linux. I'd strongly recommend "Operating System Concepts" by Silberschatz Galvin and Gagne, that should cover all you need to know about Operating System concepts. For Linux specifics, "Understanding the Linux Kernel" by Daniel P. Bovet and Marco Cesati, gives a good deep dive into the kernel's memory management system and all its components. There's also the linux man pages for the relevant syscalls of course mmap mprotect which is a must. They are all very important resources.

Oh and one last thing, you know why they call it debugging? Because sometimes you end up feeling like you’re finding bugs on an insect, one tiny microscopic thing messing up the entire thing hahaha… but honestly sometimes it’s a small thing messing up.

Anyways, I hope this helps let me know if anything is still unclear or if you have more issues.
