---
title: "How can CUDA Thrust be used to perform a maximum-reduce operation with a mask?"
date: "2025-01-30"
id: "how-can-cuda-thrust-be-used-to-perform"
---
CUDA Thrust's strength lies in its ability to elegantly express parallel algorithms using a high-level interface.  While a straightforward `thrust::reduce` operation doesn't directly incorporate masking, achieving a masked maximum-reduce requires a slightly more nuanced approach leveraging the library's flexibility.  My experience optimizing particle simulations for fluid dynamics heavily relied on precisely this technique, handling irregular data distributions efficiently.  The key is to combine `thrust::transform` with `thrust::reduce`, effectively filtering out elements based on the mask before performing the reduction.


**1. Clear Explanation:**

The masked maximum-reduce operation aims to find the maximum value within a dataset, but only considering elements where a corresponding mask element is true (typically 1 or true; 0 or false otherwise).  A naive approach might involve a serial loop, which is highly inefficient on a GPU.  Thrust offers a far superior solution.  We utilize `thrust::transform` to create a new sequence where elements are modified based on the mask. Specifically, elements corresponding to false mask values are replaced with a value guaranteed to be smaller than any potential maximum in the original dataset (e.g., negative infinity for floating-point types).  This effectively removes them from consideration during the subsequent `thrust::reduce` operation.  The `thrust::reduce` then operates on the transformed sequence, yielding the desired masked maximum.  The choice of the 'replacement' value needs careful consideration depending on the data type and potential range of valid values.


**2. Code Examples with Commentary:**

**Example 1:  Floating-point Data with a Boolean Mask**

```c++
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <limits> // Required for numeric_limits


// Custom function to apply the mask
struct masked_value {
  template <typename T>
  __host__ __device__
  T operator()(const T& val, const bool& mask) {
    return mask ? val : -std::numeric_limits<T>::infinity();
  }
};

int main() {
  thrust::host_vector<float> h_data = {1.5f, 2.7f, 3.1f, 4.2f, 0.9f};
  thrust::host_vector<bool> h_mask = {true, false, true, true, false};

  thrust::device_vector<float> d_data = h_data;
  thrust::device_vector<bool> d_mask = h_mask;

  // Transform the data based on the mask
  thrust::device_vector<float> masked_data(d_data.size());
  thrust::transform(d_data.begin(), d_data.end(), d_mask.begin(), masked_data.begin(), masked_value());

  // Perform the reduction
  float max_val = thrust::reduce(masked_data.begin(), masked_data.end(), -std::numeric_limits<float>::infinity(), thrust::maximum<float>());

  //Print result (for demonstration only; avoid in production code unless essential)
  printf("Masked Maximum: %f\n", max_val); 
  return 0;
}
```

This example demonstrates a straightforward application to floating-point data.  The `masked_value` functor applies the masking logic, replacing unmasked values with negative infinity.  `thrust::reduce` then correctly identifies the maximum among the remaining values.  Note the use of `std::numeric_limits` for a portable method of obtaining negative infinity.


**Example 2: Integer Data with an Integer Mask**

```c++
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <limits>


struct masked_value_int {
  template <typename T>
  __host__ __device__
  T operator()(const T& val, const int& mask) {
    return mask ? val : std::numeric_limits<T>::min();
  }
};

int main() {
  thrust::host_vector<int> h_data = {10, 20, 30, 40, 50};
  thrust::host_vector<int> h_mask = {1, 0, 1, 1, 0};

  thrust::device_vector<int> d_data = h_data;
  thrust::device_vector<int> d_mask = h_mask;

  thrust::device_vector<int> masked_data(d_data.size());
  thrust::transform(d_data.begin(), d_data.end(), d_mask.begin(), masked_data.begin(), masked_value_int());

  int max_val = thrust::reduce(masked_data.begin(), masked_data.end(), std::numeric_limits<int>::min(), thrust::maximum<int>());

  printf("Masked Maximum: %d\n", max_val);
  return 0;
}

```

This example adapts the technique for integer data, utilizing `std::numeric_limits<T>::min()` as the replacement value. This ensures that any masked-out element will be smaller than any valid integer.


**Example 3: Handling Custom Data Types**

```c++
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct MyData {
  int value;
  bool valid;
};

struct masked_max_custom {
  template <typename T>
  __host__ __device__
  T operator()(const T& a, const T& b) {
    return a.valid && ( (!b.valid) || (a.value > b.value) ) ? a : b;
  }
};


int main() {
    thrust::host_vector<MyData> h_data = { {10, true}, {20, false}, {30, true}, {40, true} };

    thrust::device_vector<MyData> d_data = h_data;

    MyData initial_value = {std::numeric_limits<int>::min(), false}; //Handle potentially invalid initial state
    MyData result = thrust::reduce(d_data.begin(), d_data.end(), initial_value, masked_max_custom());

    printf("Masked Maximum: %d\n", result.value);
    return 0;
}
```

This demonstrates handling a custom data structure. Here, the `MyData` struct contains both the value and a validity flag. The `masked_max_custom` functor directly compares and selects the maximum based on the `valid` flag, avoiding the need for a separate mask vector. This is more memory-efficient for custom types where the validity is inherently part of the data structure.  Careful consideration is needed for the initial value to handle the case where all elements are considered invalid.


**3. Resource Recommendations:**

*   **CUDA Programming Guide:**  Thorough understanding of CUDA fundamentals is crucial.
*   **Thrust Documentation:**  Essential for detailed information on Thrust's functions and capabilities.
*   **NVIDIA's CUDA Samples:**  Exploring relevant examples provides practical insights and implementation strategies.  Pay close attention to examples related to reduction and custom functors.
*   **A good C++ textbook:**  Solid C++ knowledge is paramount.  Pay particular attention to template metaprogramming.


Remember to always profile your code and consider different approaches for optimal performance based on the specific characteristics of your data and hardware.  The efficiency of these methods heavily depends on data size and the sparsity of the mask.  For extremely sparse masks, alternative approaches might be more efficient.  These examples provide a strong foundation for building upon and adapting to your specific needs.
