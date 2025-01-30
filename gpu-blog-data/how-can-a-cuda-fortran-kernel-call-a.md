---
title: "How can a CUDA Fortran kernel call a CUDA C device function?"
date: "2025-01-30"
id: "how-can-a-cuda-fortran-kernel-call-a"
---
Interoperability between CUDA Fortran and CUDA C code, specifically when involving device functions called within kernels, requires explicit linking and careful management of compilation units. Fortran and C have distinct memory layout conventions and calling mechanisms; thus, a straightforward function call across language boundaries during GPU execution is not possible without deliberate setup.

The fundamental issue stems from the separate compilation processes of Fortran and C/C++. When a Fortran kernel needs to utilize a device function defined in C/C++, the CUDA compiler, `nvcc`, needs to be aware of both the Fortran kernel's code as well as the compiled C/C++ device function. This awareness is achieved by separating the compilation into intermediate object files (.o or .obj) and then linking them into a final executable.

The challenge is further complicated by Fortran's name mangling conventions, which differ from C/C++. Name mangling alters the name of functions during compilation to include additional information such as the function's argument types, its scope, and whether it belongs to a module. This ensures that the linker can correctly match function calls to their definitions. Because Fortran and C use disparate mangling schemes, a direct call to a C function from Fortran would result in the linker being unable to locate the necessary symbol. We circumvent this issue using the Fortran `bind(C)` attribute, explicitly specifying that a Fortran procedure adheres to C linking conventions. Similarly, on the C side, we declare these linked functions using `extern "C"` to suppress C++ name mangling if compiled with a C++ compiler.

My experience in large-scale simulations has repeatedly demonstrated that this interoperability is essential for integrating existing C/C++ libraries into Fortran projects. For instance, I’ve encountered situations where highly optimized C libraries for specific numerical algorithms were available, requiring their integration with Fortran-based parallel simulations. Successfully integrating these libraries often depended upon a robust understanding of inter-language communication within the CUDA context.

Here’s a breakdown of a typical workflow, along with supporting code examples:

**1. Defining the CUDA C Device Function:**

First, the CUDA C device function must be defined in a separate source file, for instance, `device_functions.cu`. Here’s a basic example of a C device function designed to add a constant to a value:

```c++
// device_functions.cu
#include <cuda_runtime.h>

__device__ extern "C" float add_constant(float input, float constant) {
    return input + constant;
}
```

The `extern "C"` declaration ensures C-style linkage and suppresses C++ name mangling. The function is annotated with `__device__`, making it callable from a CUDA kernel.

**2. Defining the Fortran Kernel:**

Next, we define the Fortran CUDA kernel in a separate file, e.g., `kernel.f90`.  Crucially, we declare the C device function interface using the Fortran `bind(C)` attribute:

```fortran
! kernel.f90
module cuda_module
  use cudafor
  implicit none

  interface
    function add_constant(input, constant) bind(C,name="add_constant")
      use iso_c_binding, only: c_float
      real(c_float), value :: input
      real(c_float), value :: constant
      real(c_float) :: add_constant
    end function add_constant
  end interface

contains
  attributes(global) subroutine my_kernel(d_out, d_in, size, constant)
    real(c_float), device, intent(inout) :: d_out(:)
    real(c_float), device, intent(in) :: d_in(:)
    integer, value :: size
    real(c_float), value :: constant
    integer :: i

    i = blockIdx%x*blockDim%x + threadIdx%x
    if (i < size) then
      d_out(i) = add_constant(d_in(i), constant)
    end if
  end subroutine my_kernel
end module cuda_module

program main
    use cuda_module
    use iso_c_binding
    implicit none
    integer, parameter :: N = 1024
    real(c_float), allocatable :: h_in(:), h_out(:)
    real(c_float), device, allocatable :: d_in(:), d_out(:)
    real(c_float) :: constant
    integer :: i, status

    allocate(h_in(N), h_out(N))
    allocate(d_in(N), d_out(N), stat=status)

    do i = 1, N
      h_in(i) = real(i,c_float)
    end do
    constant = 2.0_c_float
    h_out = 0.0_c_float

    d_in = h_in
    d_out = h_out

    call my_kernel<<< (N+255)/256, 256 >>>(d_out, d_in, N, constant)
    status = cudaDeviceSynchronize()

    h_out = d_out

    do i = 1, 10
       print *, h_out(i)
    end do
end program main
```

Here, we define the interface to `add_constant`, specifying the correct C-style function name, argument types using `iso_c_binding` for type compatibility, and the `bind(C)` attribute. This interface enables Fortran to call the C device function.

**3. Compiling and Linking:**

The final step involves compiling both files and linking them.  Crucially, this involves separate compilation and then linking with `nvcc` to generate the executable:

First compile the C code:

```bash
nvcc -c device_functions.cu -o device_functions.o
```

Then compile the fortran code:

```bash
nvfortran -c kernel.f90 -o kernel.o
```

Finally, link both object files:

```bash
nvcc kernel.o device_functions.o -o my_executable -lcudart -lcuda
```

The above commands compile the C code to an object file, then the Fortran to another object file, and finally links both object files into an executable, along with required libraries. The linking step is the crucial part that creates an executable with both the Fortran kernel and the C device function, resolving the cross-language function call. This order of compiling C/C++ then Fortran avoids Fortran linking issues in my experience with nvfortran.

**Additional Notes on Function Names**: When calling C functions from Fortran, the `name` parameter of `bind(C)` should be identical to the C function’s name as defined in the C source file; that is, before any name mangling is done. If the C function was in a C++ source file, the name would still be `add_constant` if the `extern "C"` linkage is specified, but otherwise would need to include the mangled name.

**Resource Recommendations:**

For a deeper understanding of CUDA programming:

*   The official NVIDIA CUDA documentation provides a comprehensive guide to all CUDA APIs.
*   The programming guides associated with CUDA development tools often include details of best practices.
*   Several online courses focusing on parallel programming and GPU computing are available, often covering topics such as memory management, optimization, and inter-language calls.
*   Advanced Fortran textbooks are helpful for understanding `bind(C)` and its interplay with C/C++ interfaces.
*   Example repositories that use cross-language device code are good sources for practical examples and for better understanding the build process.

By carefully structuring code and build processes, you can successfully bridge Fortran and C/C++ within CUDA, opening up possibilities for integrating heterogeneous libraries and taking advantage of the strengths of both languages in your GPU applications.
