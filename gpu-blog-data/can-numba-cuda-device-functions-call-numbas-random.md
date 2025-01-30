---
title: "Can numba CUDA device functions call numba's random number generators?"
date: "2025-01-30"
id: "can-numba-cuda-device-functions-call-numbas-random"
---
Numba CUDA device functions cannot directly invoke Numba's host-side random number generators. This stems from the fundamental architectural separation between the CPU (where host code runs) and the GPU (where device code executes) within a CUDA environment. Numba's standard random number generation functions, like those found in `numba.np.random`, are designed for execution on the host and are not compiled for or accessible from the GPU device. Attempting such calls will result in compilation errors or unexpected runtime behavior.

The challenge lies in the need for random number generation within device kernels for simulations, Monte Carlo methods, and similar applications. To overcome this, we must utilize methods explicitly designed for GPU execution. Numba provides facilities for utilizing CUDA-specific random number generator libraries, allowing for high-performance generation within device code.

There are primarily two avenues for generating random numbers within CUDA device functions using Numba. The first utilizes the cuRAND library, a high-performance random number generation library provided by NVIDIA. The second uses a less performant but sometimes more flexible method using the XORShift algorithm directly implemented in Numba device code. The cuRAND method is preferred when performance and a wide variety of distributions are important. However, the XORShift method can be useful when explicit control over the random state is paramount.

Let's consider the cuRAND library first. To utilize it, you must instantiate a `numba.cuda.random.xoroshiro128p_normal` (or similar) object within the device code. This instance will maintain a random state per thread. Crucially, you must initialize this object with a unique seed value for each thread; otherwise, identical sequences will be generated across threads, producing useless output. We can accomplish this using the thread ID available through `cuda.grid(1)`. Here's an example:

```python
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal

@cuda.jit
def device_random_curand(out, rng_states):
    i = cuda.grid(1)
    if i < out.size:
      rng = xoroshiro128p_normal(rng_states, i)
      out[i] = rng

def test_curand_generation(size, seed):
  threadsperblock = 32
  blockspergrid = (size + (threadsperblock - 1)) // threadsperblock

  rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=seed)
  out = cuda.to_device(np.empty(size, dtype=np.float32))

  device_random_curand[blockspergrid, threadsperblock](out, rng_states)
  return out.copy_to_host()


size = 1000
seed = 123
random_numbers = test_curand_generation(size, seed)
print(random_numbers[:10])
```

In this example, `create_xoroshiro128p_states` allocates and initializes the random number generator states on the device. Inside the `device_random_curand` kernel, `xoroshiro128p_normal` generates a standard normal random number, utilizing the appropriate state corresponding to thread `i`. The thread ID is crucial to use in the index for the state array, and if we omit `i`, all threads will access the same index and will output the same random values. It is also vital to understand that the `rng_states` variable needs to be passed to the device through a variable and not created locally.  The function `test_curand_generation` prepares all necessary steps to call the kernel function, such as moving the data to the GPU and creating the required random state. This code efficiently distributes random number generation across GPU threads, avoiding CPU interaction.

Now, let’s consider using the XORShift algorithm implemented directly within Numba. This approach allows for a higher degree of customization but requires more manual effort and is generally not as performant as using cuRAND. It also requires the programmer to manually pass around a state. In essence, you are creating your random number generator using bitwise operations in the device function. Here is the implementation in the kernel.

```python
import numpy as np
from numba import cuda

@cuda.jit
def xor_shift_32(state):
    x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x

@cuda.jit
def device_random_xor(out, state):
  i = cuda.grid(1)
  if i < out.size:
    rand_int = xor_shift_32(state[i:i+1])
    rand_float = float(rand_int) / (2**32-1)
    out[i] = rand_float

def test_xor_generation(size, seed):
    threadsperblock = 32
    blockspergrid = (size + (threadsperblock - 1)) // threadsperblock

    state = cuda.to_device(np.array([seed * i for i in range(threadsperblock * blockspergrid)], dtype=np.uint32))

    out = cuda.to_device(np.empty(size, dtype=np.float32))

    device_random_xor[blockspergrid, threadsperblock](out, state)
    return out.copy_to_host()


size = 1000
seed = 123
random_numbers = test_xor_generation(size, seed)
print(random_numbers[:10])
```

Here, the `xor_shift_32` function implements the core algorithm, modifying the state directly. In the kernel, each thread calls this function to generate a random integer. This integer is then normalized to a float between 0 and 1 for the output. The `state` variable is an array where each thread has a unique state initialized from the seed.  This example showcases a more fundamental approach to random number generation on the device, offering maximum control at the expense of code complexity and performance. It is critical to note, that for this implementation, the `state` variable is modified by the function `xor_shift_32` and therefore it is required to be of type `array` on the device.

Finally, let's examine an alternative approach for Gaussian distributed numbers based on the Box-Muller transform, which could be relevant when you don't want to use the specific functions such as the `xoroshiro128p_normal` function.  This is less efficient than using cuRAND directly, but can be useful for demonstration purposes or when cuRAND is not viable. Here is an example of a device function that uses this technique:

```python
import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

@cuda.jit
def box_muller_transform(rng_states, i):
    u1 = xoroshiro128p_uniform_float32(rng_states, i)
    u2 = xoroshiro128p_uniform_float32(rng_states, i + 1 if i < rng_states.shape[0] -1 else 0)

    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return z0


@cuda.jit
def device_random_gaussian(out, rng_states):
  i = cuda.grid(1)
  if i < out.size:
    out[i] = box_muller_transform(rng_states, i)

def test_gaussian_generation(size, seed):
  threadsperblock = 32
  blockspergrid = (size + (threadsperblock - 1)) // threadsperblock

  rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=seed)

  out = cuda.to_device(np.empty(size, dtype=np.float32))

  device_random_gaussian[blockspergrid, threadsperblock](out, rng_states)
  return out.copy_to_host()


size = 1000
seed = 123
random_numbers = test_gaussian_generation(size, seed)
print(random_numbers[:10])
```

In this example, the `box_muller_transform` function uses two uniformly distributed random numbers to generate a standard normal distributed number. The kernel `device_random_gaussian` applies this function to each thread. Similar to the previous examples, the random state is created on the host and then sent to the device.

For further exploration, I recommend reviewing the Numba documentation, specifically the sections pertaining to `numba.cuda.random`. Additionally, the NVIDIA documentation on cuRAND will provide a comprehensive explanation of the underlying library and its capabilities. A good understanding of the Box-Muller transform will assist in understanding the alternative gaussian distributed number generation approach. The use of the thread ID to generate a unique random number for each thread is a cornerstone of parallel processing that is useful in a wide variety of applications outside of random number generation. I also advise against using global device random number generator states unless you have a concrete and specific justification for doing so.
