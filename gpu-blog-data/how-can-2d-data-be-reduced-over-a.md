---
title: "How can 2D data be reduced over a single dimension using Thrust?"
date: "2025-01-30"
id: "how-can-2d-data-be-reduced-over-a"
---
The challenge of efficiently reducing 2D data along a single dimension is commonly encountered in image processing, scientific simulations, and data analysis. Thrust, a C++ template library for CUDA, offers highly optimized primitives that can drastically simplify this task, leading to substantial performance gains compared to a naive implementation. From personal experience with particle simulation post-processing, I've observed that hand-rolled CUDA kernels for such operations often become unwieldy to maintain and optimize. Thrust abstracts much of the complexity away while retaining high performance, allowing developers to focus on their problem rather than low-level GPU details.

The core idea revolves around treating the 2D data, stored in a linear fashion in device memory, as a sequence of rows or columns, depending on the intended reduction direction. Thrust’s `reduce_by_key` or `transform_reduce` algorithms are particularly suitable for this. The choice depends largely on the nature of the reduction and the desired output format. `reduce_by_key` is beneficial when you want the result to be associated with a key, often a row or column index. `transform_reduce` is ideal when you need to apply a transformation before the reduction or simply compute an aggregated value for each row/column.

Let's explore this with examples. We'll consider the common case of reducing along rows, meaning we’ll be producing an output where each element is the reduction of a corresponding row in the input data. The input 2D data is conceptually arranged as a matrix of dimensions *rows* x *cols*. In memory, this is stored as a linear array of size *rows* \* *cols*, typically in row-major order.

**Example 1: Row-wise Summation using `reduce_by_key`**

This example demonstrates how to sum all the elements within each row using `thrust::reduce_by_key`. We'll assume the input data is stored in a `thrust::device_vector<float>` named `input_data`, and that `rows` and `cols` are defined. We’ll produce the sum of each row as output in the `thrust::device_vector<float>` named `row_sums`.

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>

// Assume input_data, rows, cols are already defined and populated with values
thrust::device_vector<float> input_data;
size_t rows;
size_t cols;

thrust::device_vector<float> row_sums(rows);
thrust::device_vector<int> row_indices(rows * cols);
thrust::sequence(row_indices.begin(), row_indices.end(), 0); // Initialize row_indices

thrust::device_vector<float> keys(rows * cols);
for(size_t i = 0; i < rows; ++i) {
   thrust::sequence(keys.begin() + i * cols, keys.begin() + (i+1)*cols, i);
}

thrust::device_vector<float> values = input_data;
thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), thrust::make_discard_iterator(), row_sums.begin());
```

In this code, we first create a vector, `row_indices`, to facilitate assigning keys (row numbers). This sequence is then converted into row numbers stored in the `keys` vector. We then apply `reduce_by_key`, using `keys` as the source of keys, input_data as values, and `row_sums` as the destination. The result is stored in `row_sums`. The discarded iterator is used because we're not concerned with the keys after reduction.

**Example 2: Row-wise Maximum using `transform_reduce`**

This example showcases how to find the maximum value of each row using `thrust::transform_reduce`. Again, we assume that `input_data`, `rows`, and `cols` are defined. The maximum of each row will be stored in a `thrust::device_vector<float>` named `row_maxima`.

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <algorithm>

// Assume input_data, rows, cols are already defined and populated with values
thrust::device_vector<float> input_data;
size_t rows;
size_t cols;

thrust::device_vector<float> row_maxima(rows);

for (size_t i = 0; i < rows; ++i)
{
    row_maxima[i] = thrust::transform_reduce(
        input_data.begin() + i * cols,
        input_data.begin() + (i + 1) * cols,
        thrust::identity<float>(),
        thrust::maximum<float>(),
        -std::numeric_limits<float>::infinity()
    );
}
```

Here, we use a for loop to iterate through each row, extracting row-specific iterators into input_data using arithmetic offsets based on cols. `transform_reduce` takes the begin and end of each row as iterators, applies the identity function (since we need no transformation) and uses the `thrust::maximum` function as the reduction operator, effectively finding the maximum value in the specified row. The result is stored directly to `row_maxima` which contains the maximum element of each row. Note the initialization of the reduction to negative infinity.

**Example 3: Column-wise Averaging Using `transform_reduce` with Element-wise Division**

This example is more complex. It demonstrates column-wise averaging. Unlike the previous examples, we need an extra calculation to average each column after it has been summed. We'll introduce another device vector to store intermediate column sums before dividing them by the number of rows to get the column average.

```cpp
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

// Assume input_data, rows, cols are already defined and populated with values
thrust::device_vector<float> input_data;
size_t rows;
size_t cols;

thrust::device_vector<float> col_sums(cols, 0.0f);
thrust::device_vector<float> col_averages(cols);

for(size_t i = 0; i < rows; ++i) {
   thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(cols),
        col_sums.begin(),
        [=](const int& col_idx){
           return col_sums[col_idx] + input_data[i * cols + col_idx];
        }
    );
}
thrust::transform(col_sums.begin(), col_sums.end(), col_averages.begin(), [=](float val){return val/(float)rows;});

```
In this example, we iterate through each row of the matrix adding each element to the corresponding column's total. This generates the intermediate `col_sums` which is then divided by the number of rows to give column averages in `col_averages`.  Note the use of `make_counting_iterator` which provides the columns as a sequence. The Lambda capture statement, `[=]`, is crucial to properly access the row index and the current column sum for this operation.  This demonstrates the power of `transform` with element-wise custom functions.

These examples show the flexibility of Thrust for reducing data over a single dimension.  `reduce_by_key` is optimal for scenarios where grouped reduction is needed by key, whilst `transform_reduce` offers a very concise solution for a wider range of reduction operations. The combination of these tools empowers developers to handle these kinds of data manipulations without sacrificing performance.

For anyone seeking further understanding, I highly recommend examining the Thrust documentation and the associated CUDA samples, specifically those that explore `reduce_by_key`, and `transform_reduce` in detail.  Also, consider exploring books focused on GPU computing and parallel algorithms. Publications from Nvidia on CUDA optimization are also a valuable resource. Experimentation and profiling will remain critical for truly optimal results in specific use cases. Remember to always benchmark and profile your code on the target GPU architecture.
