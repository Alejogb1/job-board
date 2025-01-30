---
title: "How do I determine the SM version of my GPU?"
date: "2025-01-30"
id: "how-do-i-determine-the-sm-version-of"
---
The Streaming Multiprocessor (SM) version of a GPU is a critical piece of information when optimizing CUDA applications, as it dictates the available features and performance characteristics of the device. Identifying this version allows developers to tailor their code for specific hardware, leveraging the most efficient instruction sets and memory access patterns. Neglecting this can lead to suboptimal performance, even code incompatibility across different NVIDIA GPU generations.

Specifically, the SM version, often represented by a compute capability number (e.g., 7.0, 8.6, 9.0), indicates the architectural features of the multiprocessor. Newer SM versions introduce enhancements such as increased shared memory per block, new instruction sets (like Tensor Cores), and improved memory access patterns that significantly impact performance. Knowing this information allows code to be optimized using the `#pragma unroll` directives at compilation, for example, allowing a function to be unrolled as many times as possible, provided the compute capability is high enough to make this optimization beneficial.

I have found myself in situations many times where code optimized for a newer generation card was failing due to the differences in the architecture. In these cases, accurately identifying the SM version is not merely informative but crucial.

The process of determining the SM version can be achieved programmatically using the CUDA API. This is generally preferred over relying on external tools or driver information, as this approach integrates directly into the code and can adapt to dynamic changes in the runtime environment.

Here are three illustrative code examples using C++ and the CUDA runtime API, demonstrating methods to retrieve and interpret the SM version.

**Example 1: Basic Retrieval using `cudaDeviceGetAttribute`**

This example demonstrates the most direct method to get the compute capability via CUDA API. It fetches the major and minor version components separately and then constructs the SM version from them.

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    std::cerr << "No CUDA-enabled devices found." << std::endl;
    return 1;
  }

  int deviceId = 0; // Assumes we will check the first device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);

  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceId);


  std::cout << "GPU Name: " << deviceProp.name << std::endl;
  std::cout << "Compute Capability: " << major << "." << minor << std::endl;

  return 0;
}
```

In this example:

1.  `cudaGetDeviceCount` determines the number of available CUDA devices.
2.  `cudaGetDeviceProperties` retrieves device properties, including the GPU name.
3.  `cudaDeviceGetAttribute` with `cudaDevAttrComputeCapabilityMajor` and `cudaDevAttrComputeCapabilityMinor` fetches the major and minor versions, respectively.
4.  The output combines the major and minor versions into a standard dot-notation representation of the compute capability.

**Example 2: Error Handling and Multiple Devices**

This example expands on the first one by incorporating error handling and the ability to iterate through all available CUDA devices. This is useful for systems with multiple GPUs.

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error getting device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-enabled devices found." << std::endl;
        return 0;
    }

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, deviceId);
        if(err != cudaSuccess){
           std::cerr << "CUDA error getting device properties: " << cudaGetErrorString(err) << std::endl;
           continue;
        }
        int major, minor;
        err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId);
        if(err != cudaSuccess){
             std::cerr << "CUDA error getting compute capability major: " << cudaGetErrorString(err) << std::endl;
             continue;
        }
        err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceId);
        if(err != cudaSuccess){
            std::cerr << "CUDA error getting compute capability minor: " << cudaGetErrorString(err) << std::endl;
             continue;
        }

        std::cout << "Device " << deviceId << ": " << deviceProp.name << ", Compute Capability: " << major << "." << minor << std::endl;
    }

    return 0;
}
```

Here, each API call is checked for an error code. If an error is encountered, an error message including the specific error string is printed, and the loop continues to the next device. This provides robustness and informs the developer if something goes wrong. This prevents the program from crashing due to a minor or unexpected issue when trying to retrieve device information.

**Example 3: Using a Struct and a Helper Function**

This example encapsulates the logic of SM version retrieval into a helper function, returning a struct that holds the device name and the SM version. This promotes code reusability and clarity.

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

struct DeviceInfo {
    std::string name;
    float smVersion;
};

DeviceInfo getDeviceInfo(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceId);

    float smVersion = static_cast<float>(major) + static_cast<float>(minor) / 10.0f;
    return {deviceProp.name, smVersion};
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled devices found." << std::endl;
        return 1;
    }

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
         DeviceInfo info = getDeviceInfo(deviceId);
         std::cout << "Device " << deviceId << ": " << info.name << ", Compute Capability: " << info.smVersion << std::endl;
    }
    return 0;
}
```

This example shows the SM version as a floating-point number instead of integers separated by a period. This illustrates how to convert the integers into the float format. Structs can group the device name and SM version in a clean way.

The decision of which approach to use depends on the specific needs. The first example is sufficient for many basic applications, the second is essential for production applications where robust error handling is a necessity, and the third demonstrates how to write reusable and easier-to-read code.

When creating CUDA programs that need to work across multiple devices with different architectures it is important to detect the compute capability dynamically. This allows the program to create conditional execution based on the architecture which is necessary when optimizing for performance on multiple GPU generations.

Further exploration of NVIDIA documentation related to CUDA programming and the CUDA Driver API Reference can prove invaluable. These resources provide comprehensive details on all available functions and attributes and are essential to master for serious CUDA development. Additionally, resources dedicated to CUDA best practices, such as NVIDIA's developer website and programming guides are beneficial to learn how to best utilize GPU capabilities. Lastly, specialized texts on parallel programming with CUDA can deepen one’s knowledge of architectural nuances that impact GPU performance.
