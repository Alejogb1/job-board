---
title: "How to build an ARM GNU cross compiler?"
date: "2025-01-30"
id: "how-to-build-an-arm-gnu-cross-compiler"
---
The challenge in constructing an ARM GNU cross compiler lies not just in the complexity of the toolchain itself, but in ensuring its target architecture matches the intended embedded platform’s specific ARM variant, ABI, and operating system (or lack thereof). This necessitates a meticulous configuration and build process, deviating significantly from compiling for the host machine.

My experience has involved building custom cross-compilers for a variety of embedded ARM systems, ranging from Cortex-M microcontrollers to more powerful Cortex-A processors, each presenting unique requirements. Fundamentally, a cross-compiler enables the creation of executables on one architecture (the host, typically x86_64), designed to execute on a different architecture (the target, in this case, ARM). This requires specialized versions of core toolchain components, including the compiler (gcc), assembler (as), linker (ld), and the C standard library (libc), all tailored for the specific ARM architecture.

Building a cross-compiler generally proceeds in a multi-stage process, each phase contributing to the overall functionality. First, we must acquire the source code for the tools. This typically means downloading the latest versions of GNU binutils, gcc, and glibc (or newlib for embedded bare metal systems). Binutils, providing core low-level tools, are fundamental. Next, gcc, based on the binutils, is built as a stage 1 compiler. Finally, glibc (or newlib) completes the toolchain when a stage 2 compiler is built. The core challenge resides in configuring each component with correct target parameters, utilizing the `--target` flag consistently across all stages.

The first component built is binutils, providing the assembler, linker and other utilities. This step is relatively straightforward. Specifying the `--target` parameter correctly is of paramount importance. Building this step is a non-recursive process. Here's a basic configure and make sequence for targeting `arm-none-eabi` (a common target for embedded systems without an operating system).

```bash
#!/bin/bash

BUILD_DIR=build_binutils
SRC_DIR=~/src/binutils-2.40

mkdir -p $BUILD_DIR
cd $BUILD_DIR

../$SRC_DIR/configure \
    --target=arm-none-eabi \
    --prefix=/opt/arm-none-eabi \
    --disable-nls \
    --disable-werror

make -j$(nproc)
make install
```

In this script, `BUILD_DIR` defines where the build process occurs, keeping source files separate. `SRC_DIR` specifies the binutils source directory. The configure script is invoked with critical options. `--target=arm-none-eabi` is pivotal, specifying the architecture. `--prefix=/opt/arm-none-eabi` defines the installation location; avoid installing into system locations. `--disable-nls` turns off native language support and `--disable-werror` disables treating warnings as errors, which can be useful when first starting. `make -j$(nproc)` compiles with multiple threads for speed, and `make install` installs to the specified prefix. Proper setting of the `prefix` allows isolation from system toolchains.

Following the successful compilation of binutils, the next critical step involves building gcc. This is a recursive compilation process, and more complex.  A simple 'make all' does not work. First, a 'stage 1' compiler is built, which is used to build the C library.  The second stage compiler then has the full library. In this example, we assume a bare-metal target where a C library is newlib.

```bash
#!/bin/bash

BUILD_DIR=build_gcc
SRC_DIR=~/src/gcc-13.2.0
NEWLIB_DIR=~/src/newlib-4.3.0

mkdir -p $BUILD_DIR
cd $BUILD_DIR

../$SRC_DIR/configure \
    --target=arm-none-eabi \
    --prefix=/opt/arm-none-eabi \
    --disable-nls \
    --disable-werror \
    --enable-languages="c,c++" \
    --with-newlib-include-dir=$NEWLIB_DIR/newlib/libc/include \
    --with-sysroot=/opt/arm-none-eabi/arm-none-eabi

make all-gcc -j$(nproc)
make all-target-libgcc -j$(nproc)
make install-gcc
make install-target-libgcc
```

Again, `BUILD_DIR` and `SRC_DIR` are defined, and additionally `NEWLIB_DIR` to specify location of the newlib source.  `--enable-languages="c,c++"` specifies the supported languages. The `--with-newlib-include-dir` specifies the location of the newlib include files.  `--with-sysroot` specifies a system directory, which can be left as /opt/arm-none-eabi/arm-none-eabi to install the library to this location.  `make all-gcc` builds the compiler, followed by `make all-target-libgcc`, which builds the core support library. Corresponding 'install' commands are used to copy to the correct place.

Finally, the C standard library itself (glibc or newlib) needs to be built. In a bare-metal or embedded environment, newlib is usually preferable. However, for a more capable system running, for example, a minimal embedded Linux, glibc may be more appropriate. In this example, it is not necessary to build newlib, since it is only necessary to install the headers, but to demonstrate, here is the example build.

```bash
#!/bin/bash

BUILD_DIR=build_newlib
SRC_DIR=~/src/newlib-4.3.0

mkdir -p $BUILD_DIR
cd $BUILD_DIR

../$SRC_DIR/configure \
    --target=arm-none-eabi \
    --prefix=/opt/arm-none-eabi \
    --disable-nls \
    --disable-werror

make -j$(nproc)
make install
```

Again, we follow similar construction principles, setting `BUILD_DIR` and `SRC_DIR`. The same principles apply for configuration and installation with the appropriate `--target` and `--prefix` options.

Upon successful execution of these three stages, a functional cross-compiler will be present in `/opt/arm-none-eabi`. Testing the compiler with a basic "Hello World" program is crucial. This entails compiling the program using `arm-none-eabi-gcc` and then transferring the output to the target device for verification.

Troubleshooting typically revolves around misconfigured paths, incorrect `--target` settings, or missing dependencies. Examining the build output carefully for error messages is essential. In situations requiring customized library functionalities, building and linking against specific board support packages (BSPs) becomes a critical step; but is outside the scope of this document.

For further in-depth information, I recommend consulting the official GNU documentation for Binutils, GCC, and Glibc. Additionally, resources such as "Embedded Systems Building Blocks" and "Modern Embedded Computing" provide a more practical, applied approach to toolchain usage and integration. Furthermore, a deep understanding of the ARM Architecture Reference Manual and the specific processor implementation details are required for fine tuning and optimization.
