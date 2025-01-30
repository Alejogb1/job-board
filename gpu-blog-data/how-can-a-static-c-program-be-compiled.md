---
title: "How can a static C program be compiled using only necessary functions and variables?"
date: "2025-01-30"
id: "how-can-a-static-c-program-be-compiled"
---
Achieving minimal binary size in static C programs necessitates meticulous control over which functions and variables are ultimately included in the final executable. This process, commonly referred to as link-time garbage collection or dead code elimination, relies heavily on the interplay between the compiler, the linker, and specific build flags. The default behavior often includes substantial portions of the standard C library, even if the program utilizes only a small subset of its features.

The key principle is to ensure that only object files containing utilized functions and variables are included in the final link. This prevents the linker from incorporating entire libraries or object files that are only partially required, leading to bloated binaries. The compiler translates source code into object files, each typically containing the code and data for one or more functions. The linker then combines these object files and necessary libraries to produce the executable. It's during this linking phase that we can significantly impact binary size.

The challenge lies in convincing the linker to disregard unused symbols—functions and global variables—defined in included object files or libraries. This optimization depends on a combination of factors, including the compiler’s ability to generate object files with precise symbol information, the linker’s support for dead code elimination, and the correct flags used during compilation and linking. When linking with static libraries, such as the standard C library, the process is typically straightforward, provided we've used the appropriate flags to encourage selective linking. However, dynamic linking complicates this considerably, as the linker must include entire shared object files (.so or .dll files) to ensure resolution at runtime, making this technique less applicable for dynamic binaries.

I've found, in my experience optimizing embedded systems, that aggressive dead code elimination requires a specific build toolchain and a nuanced understanding of linker flags. For instance, GCC and Clang, common compilers in the C ecosystem, offer several relevant command-line arguments. Notably, `-ffunction-sections` and `-fdata-sections` instruct the compiler to place each function and global variable in its own unique section within the object file. This partitioning allows the linker to selectively include or exclude these individual sections, enabling fine-grained control over the final executable's composition. Furthermore, the linker flag `-Wl,--gc-sections` enables garbage collection of unused sections. Without the `-ffunction-sections` and `-fdata-sections` flags, the linker treats the whole object file as one unit, and it can't discard individual functions and data objects.

Here are three code examples with explanations of the build process:

**Example 1: Minimalist Program with Full Library Inclusion**

```c
#include <stdio.h>

int main() {
  printf("Hello, World!\n");
  return 0;
}
```

Compilation and linking without any size optimization, using standard `gcc` settings, results in a relatively large executable. The `printf` function, while simple in this context, drags in a substantial portion of the standard input/output library. This is because, by default, the linker doesn't discard the unused symbols in the object file for `libc.a`.

```bash
gcc -o hello_world hello_world.c
ls -l hello_world
```

The generated executable will be larger than required by the program's content itself. This baseline demonstrates the impact of default compilation behaviors.

**Example 2: Function Sectioning and Garbage Collection**

```c
#include <stdio.h>

void unused_function() {
    int x = 10;
    x++;
}

int main() {
    printf("Hello, Optimized World!\n");
    return 0;
}
```

In this example, we introduce an `unused_function`. Using the `-ffunction-sections` and `-fdata-sections` flags in the compiler and the `-Wl,--gc-sections` flag in the linker, we can eliminate this unused function. Here's the command:

```bash
gcc -ffunction-sections -fdata-sections -o hello_optimized hello_optimized.c -Wl,--gc-sections
ls -l hello_optimized
```
The `-ffunction-sections` and `-fdata-sections` flags, tell the compiler to put each function and global variable in its own unique section in the .o file. The `-Wl,--gc-sections` flag tells the linker to discard unused sections. The resulting executable should be smaller compared to the previous compilation, as the linker has effectively removed the object code corresponding to `unused_function` and other unused sections in `libc.a`.

**Example 3: Replacing printf with a System Call**

```c
#include <unistd.h>

int main() {
  write(1, "Hello, System Call!\n", 19);
  return 0;
}
```

This version completely replaces the `printf` function with a direct system call, `write`. The benefit here is that we bypass much of the standard I/O library, relying instead on a much smaller, kernel-level interface. While slightly less portable, this allows for drastically reduced binaries. We still use the same flags as before to optimize:

```bash
gcc -ffunction-sections -fdata-sections -o hello_syscall hello_syscall.c -Wl,--gc-sections
ls -l hello_syscall
```
The output binary will be noticeably smaller, exhibiting the impact of both garbage collection and reduced reliance on high-level libraries. Direct system call usage reduces dependency on the standard C library and results in a much smaller binary footprint. Note, that for portability, a proper wrapper around `write` would need to be constructed; for example, this needs to handle various types of files or redirection.

Beyond the flags, the way you structure your program impacts final size. For example, utilizing only statically linked libraries or using a minimal version of the standard C library, like `newlib`, is essential for embedded systems where space constraints are a major factor. Building without standard I/O is common in the very tightest resource constraints.

Resource recommendations for further exploration include compiler documentation for GCC and Clang, specifically the sections discussing code generation and link-time optimizations. Textbooks on compiler design and system programming often have comprehensive information regarding object file formats and linking, providing insights into the process of building and linking code. The ELF (Executable and Linkable Format) specification is useful for a deep dive into how object files and executables are structured, but this is a more technical document. Additionally, examining the build process in embedded development environments can illuminate techniques for minimizing code and data sizes when dealing with limited resources. While I've focused on C, many similar concepts apply to other compiled languages and their respective toolchains.
