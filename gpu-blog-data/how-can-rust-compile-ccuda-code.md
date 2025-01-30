---
title: "How can Rust compile C/CUDA code?"
date: "2025-01-30"
id: "how-can-rust-compile-ccuda-code"
---
Rust's capacity to integrate with C and CUDA code stems from its Foreign Function Interface (FFI) and dedicated tooling, allowing it to leverage existing libraries and hardware capabilities. Specifically, it does not directly *compile* C or CUDA; rather, it compiles Rust code that *interacts* with pre-compiled or concurrently compiled C/CUDA libraries. I've encountered this frequently in embedded systems work and high-performance computing, where integrating legacy C code or utilizing GPUs for specific tasks was a necessity.

The core mechanism for C integration is Rust's FFI. FFI allows Rust to call functions defined in other languages (primarily C, but also C++) and vice versa, and this is accomplished using the `extern` keyword and the `#[link]` attribute. When Rust compiles code utilizing FFI, it generates calls that adhere to the Application Binary Interface (ABI) of the target architecture, allowing seamless communication with native libraries compiled through standard C/C++ compilers. This interoperability is not automatic; careful handling of data types, memory management, and error handling becomes essential.

In general terms, the interaction involves these steps:

1.  **C/C++ Code Compilation:** C/C++ code is compiled into a shared library (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows) or a static library (`.a` on Linux/macOS, `.lib` on Windows) using a traditional C/C++ compiler like GCC or Clang. These libraries contain the compiled machine code and metadata needed for linking.

2.  **FFI Declaration in Rust:** Rust code declares the functions and data structures of the C/C++ library that it intends to use using the `extern` keyword followed by the linkage name (e.g., `extern "C"` for C libraries). This declaration specifies the function signature with the appropriate type mappings between Rust and C.

3.  **Linking:** The Rust compiler links the compiled Rust code with the pre-compiled C/C++ library during the linking phase. The linker resolves references to external symbols (functions and variables) by connecting the Rust code's function calls with the corresponding functions in the C/C++ library.

4.  **Runtime Interaction:** Once compiled and linked, the Rust application can call functions from the linked C/C++ library as if they were native Rust functions. Data transfer between Rust and C/C++ must adhere to the FFI rules.

For CUDA, the approach is slightly more complex but conceptually similar. The CUDA SDK provides a compiler, `nvcc`, which compiles CUDA code (typically written in a C++ dialect) into a `.cubin` file or other intermediate representation for the specific GPU target. Rust code then uses CUDA's C API (or higher-level bindings like the `cuda-rs` crate) to interact with the compiled CUDA kernel. Similar to C interaction, Rust does not compile CUDA code, it utilizes the compiled `.cubin` files or CUDA runtime API to upload the kernels to the GPU and execute them.

Let's look at some examples illustrating these concepts:

**Example 1: Simple C Library Integration**

Suppose you have a C file named `add.c` containing the following code:

```c
// add.c
int add(int a, int b) {
  return a + b;
}
```

You would first compile this to a shared library: `gcc -shared -o libadd.so add.c`. In a Rust program, you can call this `add` function:

```rust
// src/main.rs
extern "C" {
    fn add(a: i32, b: i32) -> i32;
}

fn main() {
    let result = unsafe { add(5, 3) }; // Calling the C function requires 'unsafe'
    println!("The result of 5 + 3 from C: {}", result);
}
```

**Explanation:** The `extern "C"` block declares the `add` function with its expected C signature. `unsafe` is needed because Rust cannot guarantee the safety of code interacting with C. During compilation, the Rust compiler will need to be linked to the `libadd.so` library; this is achieved either during the build using a build script or through direct linking via `-l` command line argument. Specifically, `rustc src/main.rs -l add -L .` will do the job when the shared library is in the current directory.

**Example 2:  C Library with Structs**

Let’s enhance the previous example by using a C structure:

```c
// person.h
typedef struct {
    char name[50];
    int age;
} Person;

void print_person(const Person* person);
```

```c
// person.c
#include "person.h"
#include <stdio.h>
#include <string.h>

void print_person(const Person* person) {
  printf("Name: %s, Age: %d\n", person->name, person->age);
}
```
Compile: `gcc -shared -o libperson.so person.c`.  Now, the Rust counterpart:

```rust
// src/main.rs
#[repr(C)]
struct Person {
    name: [u8; 50],
    age: i32,
}

extern "C" {
    fn print_person(person: *const Person);
}

fn main() {
    let person = Person {
        name: *b"Alice\0                                         ",
        age: 30,
    };

    unsafe {
        print_person(&person);
    }
}
```

**Explanation:** The `#[repr(C)]` attribute ensures that Rust's `Person` struct has the same memory layout as the C struct, essential for FFI compatibility. Note that the name has to be padded with zeros up to 50 bytes since C strings are null-terminated. A pointer is passed, rather than the struct itself since that matches the C signature.

**Example 3: Simple CUDA Interaction**

For a simplified CUDA example, consider a CUDA kernel defined in `kernel.cu`:

```cpp
// kernel.cu
#include <cuda.h>
__global__ void addArrays(int *a, int *b, int *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
       c[i] = a[i] + b[i];
    }
}
```
Compile this with `nvcc -ptx kernel.cu -o kernel.ptx`. To interact with it from Rust, you will need a CUDA crate. Here is a minimal example using the `cuda-rs` crate.

```rust
use cuda_rs::{Device, LaunchConfig};
use std::ffi::CString;

fn main() {
    let device = Device::new(0).expect("Failed to get device");
    let ptx_bytes = std::fs::read("kernel.ptx").expect("Failed to read ptx file");
    let ptx_cstr = CString::new(ptx_bytes).expect("Failed to construct C string from ptx");
    let module = device.load_ptx(&ptx_cstr, Some(CString::new("addArrays").unwrap())).expect("Failed to load ptx");

    let size = 1024;
    let a = vec![1; size];
    let b = vec![2; size];
    let mut c = vec![0; size];

    let a_gpu = device.copy_to_device(&a).expect("Failed to copy to device");
    let b_gpu = device.copy_to_device(&b).expect("Failed to copy to device");
    let mut c_gpu = device.copy_to_device(&c).expect("Failed to copy to device");

    let launch_config = LaunchConfig {
        block_size: 256,
        grid_size: (size + 255) / 256
    };


   let kernel_function = module.get_function("addArrays").expect("Failed to get the kernel function");

   unsafe {kernel_function.launch(launch_config, &mut a_gpu, &mut b_gpu, &mut c_gpu, &size).expect("Failed to launch kernel")};

    device.copy_from_device(&mut c, &c_gpu).expect("Failed to copy from device");


    println!("Result from GPU {:?}", &c[0..10]); // Display the first 10 elements
}
```

**Explanation:** This example loads the compiled `kernel.ptx`, copies input data to the GPU, executes the kernel, and copies the result back to the host. The `cuda-rs` crate provides a more idiomatic way of interaction compared to using CUDA C APIs directly but under the hood, it does the same thing. The `unsafe` block is used since the GPU interaction is considered outside the Rust's memory safety model.

For further understanding and practical usage, I recommend focusing on specific documentation related to Rust's FFI.  For CUDA-specific integration, explore the `cuda-rs` crate repository and the official NVIDIA CUDA documentation. Examining real-world projects utilizing FFI and CUDA in GitHub can offer insightful implementation patterns. Good introductory books on Rust also cover FFI extensively, and can help build a conceptual foundation.  Furthermore, the "Rust and WebAssembly" book offers a good view on the general concept of foreign function interfaces. A deep understanding of C/C++ memory management, pointer manipulation, and ABIs is essential for avoiding common pitfalls when interacting with external libraries from Rust.
