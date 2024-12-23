---
title: "How to build a RISC-V 32I GCC compiler for uncompressed instructions?"
date: "2024-12-23"
id: "how-to-build-a-risc-v-32i-gcc-compiler-for-uncompressed-instructions"
---

Alright, let's delve into the process of crafting a RISC-V 32I gcc compiler tailored for uncompressed instructions. I’ve actually navigated this territory before, back when I was working on a low-power embedded system where instruction set size wasn't a critical bottleneck; raw performance was paramount. Compressing instructions wasn't on the table; we needed every cycle to count. This requires a nuanced approach, ensuring that the compiler toolchain understands the specific parameters and generates only 32-bit instructions.

Essentially, building a gcc compiler, or any compiler for that matter, involves configuring a complex suite of software components, and modifying it to produce the desired output. In this case, it means disabling the generation of compressed instructions within the RISC-V architecture. Here's a structured breakdown of the steps I've found effective over the years, along with practical code examples.

First, let's examine the key components that are necessary. A RISC-V toolchain generally consists of several crucial tools: *binutils*, for assembly, linking, and binary manipulation; *gcc*, the core compiler; and optionally, the *newlib* or *glibc* libraries, providing standard C functionalities. We will use the common *newlib* since we are likely targeting a simple embedded environment. We will focus our attention primarily on *gcc* and *binutils*, as they handle the crucial parts of target specification.

The initial step is to obtain the sources of these tools. You can get the official repositories from GNU or SiFive: *binutils* (`git://sourceware.org/git/binutils-gdb.git`), *gcc* (`git://gcc.gnu.org/git/gcc.git`), and *newlib* (`git://sourceware.org/git/newlib-cygwin.git`). It's crucial to use specific releases or branch tags to ensure the components are compatible. For our illustration, let's assume we’re targeting the `release/12` branch for *gcc* and *binutils* and the current `master` for *newlib*.

Once you have the sources, the next stage involves configuring the build environment. A separate build directory is recommended to keep the source clean and organized. This will prevent conflicts in the source tree and allows multiple build configurations. Let's create a directory named `riscv32i-toolchain`. We'll work from inside this directory.
```bash
mkdir riscv32i-toolchain
cd riscv32i-toolchain
mkdir build-binutils build-gcc build-newlib install
```
Here, `install` is where the resulting compiler toolchain will reside.

Now, let's tackle *binutils*. We'll use the following configuration, focusing on a 32-bit RISC-V target:
```bash
cd build-binutils
../binutils/configure --prefix=../install --target=riscv32-unknown-elf --disable-nls --enable-multilib
make -j$(nproc)
make install
cd ..
```
The `--target=riscv32-unknown-elf` option ensures that binutils will generate code for a generic 32-bit RISC-V architecture. The `--disable-nls` option speeds up the build by skipping localization features, and `--enable-multilib` allows building for multiple architectures if needed. *It's important to understand the implications of each of these options and tailor them to your specific requirements.*

Next up is *gcc*. Here's where the crucial configuration to disable compressed instructions happens. We use `-march=rv32i` which forces the compiler to only generate standard base instruction set:
```bash
cd build-gcc
../gcc/configure --prefix=../install --target=riscv32-unknown-elf --disable-nls --disable-multilib --enable-languages=c,c++ --with-newlib --disable-decimal-float --disable-libgomp --disable-libmudflap --disable-libssp --disable-libquadmath --disable-libsanitizer --with-pkgversion=custom-uncompressed --with-arch=rv32i --with-abi=ilp32
make -j$(nproc) all-gcc
make install-gcc
cd ..
```

Here, `--target=riscv32-unknown-elf` continues to specify the RISC-V target. The `--enable-languages=c,c++` enables support for C and C++. `--with-newlib` integrates newlib as our standard C library. Most other options are disabled to reduce the build size and complexity. *Crucially, the `--with-arch=rv32i` and `--with-abi=ilp32` flags are what enforce the use of the RISC-V 32I base instruction set and corresponding Application Binary Interface (ABI), explicitly excluding compressed extensions.* The `--with-pkgversion=custom-uncompressed` just gives our compiler a name so we can identify it.
It’s critical to understand that *this explicitly tells the compiler to only use the standard 32-bit instructions.*

Finally, let's build the *newlib*. We need to build it with the same target as the compiler.
```bash
cd build-newlib
../newlib/configure --prefix=../install --target=riscv32-unknown-elf --disable-nls
make -j$(nproc)
make install
cd ..
```

With all components compiled, the toolchain should be available in the `install` directory. We need to add this path to our environment, this will allow us to use the newly built toolchain from the command line.
```bash
export PATH=$PWD/install/bin:$PATH
```

Now, to verify that the compiler is indeed generating only uncompressed 32-bit instructions, we can write a simple C program and compile it. Consider this snippet named `test.c`:
```c
#include <stdio.h>

int main() {
  int a = 10;
  int b = 20;
  int c = a + b;
  printf("The sum is: %d\n", c);
  return 0;
}

```
Compile this code using the newly built compiler and disassemble the output.
```bash
riscv32-unknown-elf-gcc -o test test.c
riscv32-unknown-elf-objdump -d test
```
Inspecting the disassembled code, you'll observe that all instructions are indeed 32-bit long and are part of the rv32i instruction set.

This method ensures the generated executables will not use any compressed extensions. For deeper understanding of the RISC-V architecture and its instruction formats, I recommend consulting the official *RISC-V instruction set manuals*. For compiler specifics, “*Modern Compiler Implementation in C*” by Andrew W. Appel offers a comprehensive treatment of compiler construction. Also, the *GNU Compiler Collection Internals* documentation provides insight into the workings of gcc.

In conclusion, building a RISC-V 32I gcc compiler that generates only uncompressed instructions involves careful configuration of the toolchain, particularly specifying the target architecture using `--with-arch=rv32i` and ABI using `--with-abi=ilp32` during gcc configuration. This approach, rooted in my experience, provides a reliable solution for those requiring full control over the generated code for specific hardware requirements. You need to make sure that when you have a working compiler, you verify the output to ensure it aligns with your expectations. This methodology focuses on clarity and directness, which, from my experience, are the most effective tools for software development.
