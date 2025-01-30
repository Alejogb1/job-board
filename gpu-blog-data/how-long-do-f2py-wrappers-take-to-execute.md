---
title: "How long do f2py wrappers take to execute?"
date: "2025-01-30"
id: "how-long-do-f2py-wrappers-take-to-execute"
---
F2py execution time is not a single, fixed value; rather, it's a composite influenced by several factors, primarily the complexity of the Fortran subroutine being wrapped, the data transfer overhead between Python and Fortran, and the specific hardware and software configuration. My experience across multiple high-performance computing projects, where I routinely integrated legacy Fortran code into Python workflows, has shown that while the actual Fortran computation within the f2py-generated wrapper can be exceedingly fast, the cost of data marshaling frequently dominates overall execution time.

The process of wrapping a Fortran subroutine with f2py creates an interface through which Python can call the compiled Fortran code. This interface necessitates the conversion of Python’s data structures (typically NumPy arrays) into their Fortran equivalents before invoking the Fortran routine, and then back to Python objects when the routine has completed. This data conversion is not a zero-cost operation, particularly with large arrays. The fundamental reason is the difference in memory layout: Fortran typically stores arrays in column-major order, while NumPy arrays are by default in row-major order. F2py must bridge this gap, often by creating temporary copies of the data with transposed axes if the array is not C-contiguous, impacting performance. Moreover, the data transfer process between the Python interpreter and the compiled Fortran extension module also carries inherent overhead.

To understand these effects in practice, consider three scenarios with varying complexities. The first involves a simple Fortran routine that performs an element-wise addition on a pair of floating-point arrays. The core Fortran code, `add_arrays.f90`, might look like this:

```fortran
subroutine add_arrays(a, b, c, n)
  implicit none
  real(8), intent(in) :: a(:), b(:)
  real(8), intent(out) :: c(:)
  integer, intent(in) :: n
  integer :: i

  do i = 1, n
    c(i) = a(i) + b(i)
  end do

end subroutine add_arrays
```

This subroutine accepts two input arrays `a` and `b`, sums them element-wise, storing results in `c`, and takes `n` as the array size. Here's the Python code to test the f2py wrapper performance:

```python
import numpy as np
import time
from numpy.f2py import f2py2e

# Compile the Fortran code
f2py2e.run_main(['-c','-m','add_mod','add_arrays.f90'])
from add_mod import add_arrays

n = 10**6
a = np.random.rand(n)
b = np.random.rand(n)
c = np.zeros(n)

start_time = time.time()
add_arrays(a, b, c, n)
end_time = time.time()

print(f"Execution time for simple addition with {n} elements: {end_time - start_time:.6f} seconds")
```

The execution time here primarily reflects the data transfer for the 3 million doubles and the basic arithmetic within the Fortran routine. This operation is relatively fast, often executing in milliseconds even for large arrays.

Now, let's examine a more computationally intensive operation. Consider a Fortran routine that performs a matrix multiplication (`matmul.f90`):

```fortran
subroutine matmul(a, b, c, n)
  implicit none
  real(8), intent(in) :: a(:,:), b(:,:)
  real(8), intent(out) :: c(:,:)
  integer, intent(in) :: n
  integer :: i,j,k

  do i=1,n
    do j=1,n
      c(i,j) = 0.0
      do k=1,n
        c(i,j) = c(i,j) + a(i,k) * b(k,j)
      end do
    end do
  end do

end subroutine matmul
```

The Python usage is as follows:

```python
import numpy as np
import time
from numpy.f2py import f2py2e

# Compile the Fortran code
f2py2e.run_main(['-c','-m','matmul_mod','matmul.f90'])
from matmul_mod import matmul

n = 100
a = np.random.rand(n, n)
b = np.random.rand(n, n)
c = np.zeros((n, n))


start_time = time.time()
matmul(a, b, c, n)
end_time = time.time()

print(f"Execution time for matrix multiplication ({n}x{n}): {end_time - start_time:.6f} seconds")
```

Here, the computation cost of the matrix multiplication is significantly higher than the element-wise addition. The data transfer overhead still exists, but it becomes a smaller fraction of the total execution time. This highlights that the nature of the Fortran subroutine plays a critical role in f2py performance. The time complexity of the algorithm has a large effect.

Finally, let's consider a scenario where memory layout considerations become more prominent. We use the same `add_arrays.f90`, but now explicitly create a non-contiguous array by transposing a copy of array `a`.

```python
import numpy as np
import time
from numpy.f2py import f2py2e

# Compile the Fortran code
f2py2e.run_main(['-c','-m','add_mod','add_arrays.f90'])
from add_mod import add_arrays


n = 10**6
a = np.random.rand(n).reshape(1,n)
b = np.random.rand(n)
c = np.zeros(n)

# Transpose a copy of a to create non-contiguous array
a_noncont = a.T.copy()

start_time = time.time()
add_arrays(a_noncont, b, c, n)
end_time = time.time()

print(f"Execution time for addition with non-contiguous array: {end_time - start_time:.6f} seconds")

start_time = time.time()
add_arrays(a.reshape(n),b,c,n)
end_time = time.time()

print(f"Execution time for addition with contiguous array: {end_time - start_time:.6f} seconds")
```

In this example, the non-contiguous array `a_noncont` forces f2py to create a copy with transposed axes behind the scenes. This adds significant overhead compared to the contiguous case. The performance difference can be very noticeable, especially with large arrays. If the Fortran subroutine requires a contiguous array, the data transfer phase will be expensive if you don't input a contiguous array.

In summary, f2py wrapper execution time is heavily context-dependent. Simple routines may be primarily limited by data transfer overhead while complex computations reduce the relative cost of this overhead. Array contiguity is also vital.

For those aiming to optimize f2py performance, several resources provide valuable guidance. I'd recommend exploring documentation related to NumPy array contiguity, specifically how to create and manipulate arrays with optimal memory layout for efficient data transfer. Compiler optimization settings for Fortran can also have a considerable impact; consult your specific compiler manual to explore flags relevant to performance. Resources on multi-dimensional arrays and performance impacts in Fortran (and NumPy) are also valuable. Finally, articles and texts discussing the overheads involved in crossing language boundaries with foreign function interfaces (FFIs) can provide a more general understanding of the fundamental challenges.
