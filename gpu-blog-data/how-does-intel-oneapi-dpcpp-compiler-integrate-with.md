---
title: "How does Intel oneAPI dpcpp compiler integrate with Google Test?"
date: "2025-01-30"
id: "how-does-intel-oneapi-dpcpp-compiler-integrate-with"
---
Intel oneAPI's DPC++ compiler, *dpcpp*, integrates with Google Test (gtest) primarily through standard C++ compilation and linking practices. This compatibility hinges on the fact that DPC++ generates standard object files that can be linked against libraries compiled by other C++ compilers, including those used to build gtest. My experience in porting a legacy C++ physics simulation to SYCL has revealed this to be a robust, albeit occasionally intricate, process. The key lies in ensuring that the compilation environment is correctly configured and that device code is handled distinctly from the host code.

The integration process is fundamentally about separating the testing framework (gtest) which executes on the host, from the computationally intensive kernels that may execute on Intel GPUs or CPUs via SYCL. The tests themselves are coded using standard C++, leveraging the gtest API. These tests then invoke functions that, in turn, enqueue SYCL kernels for execution. Therefore, the interaction isn't a direct link between the gtest framework and the DPC++ compiler’s internals, but rather between the gtest tests and the SYCL runtime environment, which has been built with DPC++.

Essentially, a build pipeline will typically include two compilation steps: one for the test binaries using a standard C++ compiler (like g++) that includes gtest, and another using *dpcpp* to produce object files for the kernels and host-side functions which rely on DPC++. Then, a final linking stage combines these artifacts. This multi-step approach isolates the complexities of SYCL compilation while still providing a framework for verifying the correctness of the compute kernels.

Let's consider specific examples. Suppose we have a basic function that adds two numbers using SYCL for acceleration. The following examples demonstrate how gtest would integrate with this using DPC++.

**Example 1: Simple SYCL Addition Test**

First, let's define the SYCL component. We'll put this in `sycl_add.cpp`:

```cpp
#include <CL/sycl.hpp>
#include <vector>

void sycl_add(std::vector<int>& a, std::vector<int>& b, std::vector<int>& result, size_t size) {
    sycl::queue q;
    {
        sycl::buffer<int, 1> buf_a(a.data(), sycl::range<1>(size));
        sycl::buffer<int, 1> buf_b(b.data(), sycl::range<1>(size));
        sycl::buffer<int, 1> buf_result(result.data(), sycl::range<1>(size));

        q.submit([&](sycl::handler& h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_result = buf_result.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
                acc_result[i] = acc_a[i] + acc_b[i];
            });
        });
    }
    q.wait();
}
```

This function defines a basic SYCL kernel that adds two vectors of integers. Note that it's not directly coupled to any testing framework.

Now, in `test_sycl_add.cpp`, we'll create a Google Test case that will verify the correctness of the `sycl_add` function:

```cpp
#include "gtest/gtest.h"
#include "sycl_add.h"
#include <vector>

TEST(SyclAdditionTest, SimpleAddition) {
    size_t size = 5;
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {6, 7, 8, 9, 10};
    std::vector<int> result(size);

    sycl_add(a, b, result, size);

    std::vector<int> expected = {7, 9, 11, 13, 15};
    ASSERT_EQ(result, expected);
}
```

Here, the test constructs vectors, calls our SYCL function, and then asserts that the results match the expected values. The `sycl_add` function executes the SYCL kernel, and the test verifies its correctness within the gtest framework.

**Compilation Example 1 (Simplified):**

```bash
# Compile sycl_add.cpp using dpcpp
dpcpp -c sycl_add.cpp -o sycl_add.o

# Compile test_sycl_add.cpp using g++
g++ -c test_sycl_add.cpp -I/path/to/gtest/include -o test_sycl_add.o

# Link everything together
g++ test_sycl_add.o sycl_add.o -L/path/to/gtest/lib -lgtest -lgtest_main -lsycl -o test_executable
```

(Replace `/path/to/gtest/include` and `/path/to/gtest/lib` with your actual gtest directories.)

This compilation sequence first uses the *dpcpp* to compile the SYCL related file which is responsible for the calculation, and then compiles the gtest file with g++ linking everything together. The `-lsycl` flag is essential to link the necessary SYCL libraries which will allow the compiled program to access SYCL devices.

**Example 2: SYCL Kernel with Data Initialization and Verification**

Often, kernels require initializing data before processing. Consider this extension:

`sycl_init_add.cpp`:

```cpp
#include <CL/sycl.hpp>
#include <vector>
#include <numeric> // for std::iota

void sycl_init_add(std::vector<int>& result, size_t size, int offset) {
    sycl::queue q;
    {
        sycl::buffer<int, 1> buf_result(result.data(), sycl::range<1>(size));

        q.submit([&](sycl::handler& h) {
            auto acc_result = buf_result.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
                acc_result[i] = static_cast<int>(i) + offset;
            });
        });
    }
    q.wait();
}
```

`test_sycl_init_add.cpp`:

```cpp
#include "gtest/gtest.h"
#include "sycl_init_add.h"
#include <vector>
#include <numeric> // for std::iota

TEST(SyclInitAddTest, InitializedAddition) {
    size_t size = 10;
    std::vector<int> result(size);
    int offset = 5;

    sycl_init_add(result, size, offset);

    std::vector<int> expected(size);
    std::iota(expected.begin(), expected.end(), offset);

    ASSERT_EQ(result, expected);
}
```

Here, the SYCL kernel initializes data, adding an offset. The test verifies if it initializes correctly.

**Example 3: Using Multiple SYCL Devices**

It is also possible to test code that is designed to run across multiple devices. A SYCL queue can be targeted to specific devices, although for this test we will only check for at least one device.

`sycl_device.cpp`:

```cpp
#include <CL/sycl.hpp>

bool is_device_available() {
    try {
        sycl::queue q;
        return true;
    }
    catch (sycl::exception const& e) {
       return false;
    }
}
```

`test_sycl_device.cpp`:

```cpp
#include "gtest/gtest.h"
#include "sycl_device.h"

TEST(SyclDeviceTest, DeviceAvailability) {
    ASSERT_TRUE(is_device_available());
}
```

Here, the code attempts to create a queue. This is a fairly simple check, but it allows us to demonstrate the use of a device.

**Resource Recommendations:**

For further study, I recommend:

1. **Intel oneAPI Programming Guide:** This official document provides comprehensive details about the DPC++ language and its compilation process.
2. **SYCL Specification:** The formal SYCL specification outlines the standard that DPC++ implements.
3. **Google Test Documentation:** This provides extensive resources for using the Google Test framework, covering topics such as test fixtures, parameterized tests, and test suites.
4. **Examples provided in the oneAPI toolkit:** Intel ships a wide range of DPC++ examples which are located in the `<oneapi-install-path>/samples` directory. This provides practical demonstrations of DPC++ usage.

In conclusion, integrating Intel oneAPI’s *dpcpp* compiler with Google Test is a process that largely relies on standard C++ build procedures, taking advantage of the compiler's ability to generate standard object files that can be linked with libraries compiled by traditional C++ compilers. The critical aspect is to separate host testing from device execution. By understanding the compilation process, one can effectively verify SYCL based kernels using robust testing frameworks like Google Test.
