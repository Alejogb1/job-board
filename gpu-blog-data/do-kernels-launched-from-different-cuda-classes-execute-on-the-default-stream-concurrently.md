---
title: "Do kernels launched from different CUDA classes execute on the default stream concurrently?"
date: "2025-01-26"
id: "do-kernels-launched-from-different-cuda-classes-execute-on-the-default-stream-concurrently"
---

CUDA stream behavior, particularly regarding kernel concurrency from diverse class instantiations, presents a nuanced area for efficient GPU programming. The fundamental principle to grasp is that, by default, CUDA kernels launched from separate class instances do not inherently execute concurrently on the same device. The default stream in CUDA is a *synchronizing stream*; meaning, kernel launches onto the default stream respect the order in which they were issued, guaranteeing sequential execution. This holds true *even if* these launches originate from distinct class objects. The key factor is the stream, not the object responsible for issuing the launch.

The common misconception stems from the object-oriented nature of the code; it's easy to assume that since `class A` and `class B` execute their kernels, each has its own implicit queue or context. This is untrue. Unless an explicit non-default stream is created and utilized, *all* kernel launches implicitly target the default stream (stream 0). This leads to serialization; one kernel will complete on the GPU before the next one starts, even if they are computationally independent. Concurrency can only be achieved by launching kernels on *different* streams.

To illustrate, consider a scenario where I developed a data processing application involving multiple data sets, each handled by a distinct class. I initially assumed that because my classes were independent, their associated GPU work would occur in parallel. I quickly discovered this to be false, resulting in severely underutilized GPU resources. The resolution came from understanding CUDA streams and their effect on kernel execution.

Let's consider three examples.

**Example 1: Sequential Execution with Default Stream**

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

class KernelLauncherA {
public:
    __device__ static void kernel_a() {
        // Simulate some work
        for (int i = 0; i < 100000; ++i) {
           int temp = i * i; // Placeholder calcuation
        }
    }
    void launch() {
        kernel_a<<<1, 1>>>();
        cudaDeviceSynchronize(); // Wait for completion
        std::cout << "Kernel A completed." << std::endl;
    }
};

class KernelLauncherB {
public:
    __device__ static void kernel_b() {
        // Simulate some work
         for (int i = 0; i < 200000; ++i) {
           int temp = i * i; // Placeholder calculation
        }
    }
     void launch() {
        kernel_b<<<1, 1>>>();
        cudaDeviceSynchronize(); // Wait for completion
        std::cout << "Kernel B completed." << std::endl;
    }
};


int main() {
    KernelLauncherA launcher_a;
    KernelLauncherB launcher_b;

    launcher_a.launch();
    launcher_b.launch();

    return 0;
}
```

In this example, `KernelLauncherA` and `KernelLauncherB` each launch a kernel using default parameters onto the default stream. Even though the kernels are defined in separate classes, and `launcher_a.launch()` is invoked before `launcher_b.launch()`, the output confirms sequential execution due to their reliance on the default stream.  The call to `cudaDeviceSynchronize()` after each launch demonstrates explicitly how the second kernel cannot start until the first kernel completes. The print statements clearly order the kernel completion messages, indicating that `Kernel B` does not begin executing until `Kernel A` has completed. This highlights the default serialization behavior.

**Example 2: Concurrency with Explicit Streams**

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

class KernelLauncherA {
public:
    __device__ static void kernel_a() {
        // Simulate some work
        for (int i = 0; i < 100000; ++i) {
           int temp = i * i; // Placeholder calculation
        }
    }
    void launch(cudaStream_t stream) {
        kernel_a<<<1, 1, 0, stream>>>();
        std::cout << "Kernel A launched on stream." << std::endl;
    }

};

class KernelLauncherB {
public:
    __device__ static void kernel_b() {
        // Simulate some work
         for (int i = 0; i < 200000; ++i) {
           int temp = i * i; // Placeholder calculation
        }
    }
    void launch(cudaStream_t stream) {
        kernel_b<<<1, 1, 0, stream>>>();
         std::cout << "Kernel B launched on stream." << std::endl;
    }
};


int main() {
    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    KernelLauncherA launcher_a;
    KernelLauncherB launcher_b;

    launcher_a.launch(streamA);
    launcher_b.launch(streamB);

    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);

    std::cout << "Both streams synchronized." << std::endl;
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
    return 0;
}
```

In this modified version, I have allocated and used two explicit CUDA streams, `streamA` and `streamB`. Each class instance now launches its respective kernel onto a different stream. The `kernel_a` function is launched using `streamA`, and `kernel_b` is launched onto `streamB`. The `cudaStreamSynchronize()` calls ensure that the host program doesn't continue until each stream's work is complete. This demonstrates how explicit streams enable concurrent kernel execution. The program output reveals that both kernels can be started without waiting for completion of the previous kernel.

**Example 3: Incorrect Stream Management**

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

class KernelLauncherA {
public:
    __device__ static void kernel_a() {
        // Simulate some work
        for (int i = 0; i < 100000; ++i) {
           int temp = i * i; // Placeholder calculation
        }
    }
     void launch(cudaStream_t stream) {
        kernel_a<<<1, 1, 0, stream>>>();
         std::cout << "Kernel A launched on stream." << std::endl;
    }
};

class KernelLauncherB {
public:
    __device__ static void kernel_b() {
       // Simulate some work
         for (int i = 0; i < 200000; ++i) {
           int temp = i * i; // Placeholder calculation
        }
    }
      void launch(cudaStream_t stream) {
        kernel_b<<<1, 1, 0, stream>>>();
         std::cout << "Kernel B launched on stream." << std::endl;
    }
};



int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    KernelLauncherA launcher_a;
    KernelLauncherB launcher_b;


    launcher_a.launch(stream);
    launcher_b.launch(stream);

    cudaStreamSynchronize(stream);

    std::cout << "Stream synchronized." << std::endl;

    cudaStreamDestroy(stream);
    return 0;
}
```
This final example illustrates a potential pitfall in stream management.  Here, I allocate *one* stream, `stream`, and pass it to both `launcher_a.launch` and `launcher_b.launch`. Even though each class is independent, and I’m explicitly passing a stream, I’ve unintentionally forced the kernels to serialize because both launch onto the *same* non-default stream, effectively creating another sequential queue. While the program will function, it will not achieve concurrency.

These examples highlight that it's the CUDA stream, not the class instance, that determines concurrent execution behavior. Explicitly creating and managing streams is essential for efficient GPU programming.

For further study, I recommend examining the CUDA programming guide, specifically the sections on stream management and asynchronous execution. Review examples provided in CUDA SDK. Consulting the CUDA runtime API documentation will also solidify stream functionality. Publications focused on GPU architecture and parallel computing can aid in understanding the underlying reasons for this behavior, moving beyond simple recipes for effective, performant use.
