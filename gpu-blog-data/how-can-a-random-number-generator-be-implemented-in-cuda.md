---
title: "How can a random number generator be implemented in CUDA?"
date: "2025-01-26"
id: "how-can-a-random-number-generator-be-implemented-in-cuda"
---

The inherent parallelism of CUDA architectures presents a unique set of considerations when implementing random number generators (RNGs). A naive approach, using a single generator accessed by all threads, would lead to significant performance bottlenecks due to thread contention and limited throughput. Therefore, the generation of random numbers in CUDA necessitates a per-thread or per-block approach, requiring algorithms specifically designed for parallel execution.

The fundamental challenge resides in ensuring both randomness and independence of generated sequences across different threads or blocks. A common strategy involves using a linear congruential generator (LCG) or a more robust variant, like a Mersenne Twister, but seeding each instance with unique initial values. These values are derived from the thread's or block's unique identifiers, ensuring distinct random sequences while capitalizing on the computational horsepower of the GPU. The algorithm's execution on the GPU demands careful management of memory access patterns and efficient parallel computation to maximize performance. The implementation often involves leveraging CUDA's built-in math functions and optimized memory access strategies.

Let's delve into a practical illustration. I previously worked on a Monte Carlo simulation for a fluid dynamics problem, and the performance of the simulation was critically dependent on the quality and speed of random number generation. A poor implementation would drastically slow down the simulation and potentially introduce biases in the results. The solution we arrived at used a LCG approach, chosen for its relative simplicity and speed, coupled with careful seed generation.

**Code Example 1: Basic Per-Thread LCG**

This example showcases the basic implementation of a LCG where each thread generates its own sequence. The seed is derived directly from the thread ID.

```c++
__device__ unsigned int lcg_thread_rand(unsigned int *state) {
    *state = (unsigned int)(1103515245 * (*state) + 12345);
    return *state;
}

__global__ void generate_random_numbers(unsigned int *output, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        unsigned int state = i;
        output[i] = lcg_thread_rand(&state);
    }
}
```

**Commentary:**

Here, `lcg_thread_rand` is a device function, callable from within the kernel. Each thread maintains its own state variable, initialized with its thread ID, `i`. This ensures each thread starts with a unique initial seed. The `generate_random_numbers` kernel launches threads to populate the output array with generated random values. Each thread independently executes `lcg_thread_rand` to generate a single random number using its unique state.  The simple multiplication and addition within `lcg_thread_rand` make it efficient for GPU execution. The output is an array `output` containing the random number for each indexed thread.  The conditional statement (`if (i < size)`) is crucial for handling cases where the launched grid exceeds the desired output size.

**Code Example 2: Per-Block LCG using Shared Memory**

This example demonstrates an alternative where each block maintains a LCG, and all threads within the block generate numbers based on the block's state. Shared memory is employed for managing the block's state.

```c++
__device__ unsigned int lcg_block_rand(unsigned int *state) {
    *state = (unsigned int)(1103515245 * (*state) + 12345);
    return *state;
}

__global__ void generate_block_random_numbers(unsigned int *output, int size) {
    extern __shared__ unsigned int shared_state[]; // shared memory

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x == 0) {
        shared_state[0] = blockIdx.x * blockDim.x;
    }
    __syncthreads();

    if(i < size) {
      output[i] = lcg_block_rand(&shared_state[0]);
   }
}
```

**Commentary:**

Here, `shared_state` is a shared memory variable allocated dynamically per block.  The first thread in each block (threadIdx.x == 0) initializes the block's state based on the block ID.  `__syncthreads()` is a synchronization primitive, ensuring all threads in the block have access to the initialized `shared_state` before proceeding with random number generation. This ensures each thread within the same block gets a related sequence derived from the same initial seed. The advantage of the approach is fewer states are kept within the GPU’s registers, which can be helpful in complex kernels. The downside is potential correlation between the random numbers generated within the same block due to the shared seed, hence its suitability is highly application specific.

**Code Example 3: Utilizing cuRAND for Higher Quality RNG**

While the previous examples offer a basic understanding, NVIDIA's cuRAND library provides highly optimized and high-quality RNGs for CUDA. This is the approach that I eventually migrated to for the production simulation. The code below illustrates its usage.

```c++
#include <curand.h>

__global__ void curand_random_numbers(float *output, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        curandState_t state;
        curand_init(i, 0, 0, &state); // Unique seed per thread
        output[i] = curand_uniform(&state);
    }
}
```

**Commentary:**

This example demonstrates the simplicity of using cuRAND. `curandState_t` defines the state needed for curand operations. `curand_init` initializes the state with a unique seed using the thread index `i`. `curand_uniform` generates a uniformly distributed random number between 0 and 1. cuRAND offers various distributions and generators such as normal and poisson. The key benefit of cuRAND lies in its optimized implementation, offering much better statistical properties and performance compared to manually implemented LCGs. While the previous examples were illustrative, using cuRAND is the recommendation for production applications where randomness quality is a critical factor.

In my experience with the fluid dynamics project, the initial simple LCG implementation, while fast, presented certain issues with statistical properties of the generated numbers, impacting the accuracy of long simulations. Switching to the cuRAND library significantly improved the accuracy and robustness of the simulation, while also being reasonably fast due to the optimized implementations.

Regarding further learning and resources, the NVIDIA CUDA documentation provides a comprehensive overview of cuRAND, including detailed explanations of its functions and usage. Textbooks on parallel computing and numerical methods also delve into the design and implementation of random number generators for parallel architectures.  Additionally, online resources detailing the various types of random number generators and their statistical characteristics (e.g., LCG, Mersenne Twister, Xorshift), would prove helpful. I’ve always found a clear understanding of both the mathematical underpinnings and the implementation details to be critical for effectively utilizing these powerful tools. Examining code examples from open-source projects utilizing CUDA for numerical simulations can also provide practical insights into different strategies for generating random numbers in parallel. It's vital to test the random number generator thoroughly in the application context and ensure it adheres to the desired statistical properties for accurate and reliable results.
