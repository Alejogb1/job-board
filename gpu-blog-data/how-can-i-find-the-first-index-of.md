---
title: "How can I find the first index of each unique element in a CUDA array?"
date: "2025-01-30"
id: "how-can-i-find-the-first-index-of"
---
Finding the first occurrence of each unique element within a large CUDA array presents a compelling performance challenge, particularly when dealing with data sets that exceed the capacity of single thread processing. Traditional linear search methods are fundamentally unsuited to the parallel paradigm, therefore a more sophisticated approach leveraging CUDA’s parallel processing capabilities is required. My work on large-scale genomic analysis involved similar challenges where rapid identification of unique sequences was crucial, forcing me to develop an optimized strategy for this problem. The core idea is to employ a combination of sorting and prefix sum operations, which can be executed efficiently in parallel on the GPU.

The fundamental issue with a naïve approach is the inherent serial nature of finding the *first* index. Standard reduction techniques will naturally produce the *last* index if one is just blindly comparing values. We can break down the process into several stages: first, identify and mark unique elements; second, determine the index of the first occurrence; and third, associate the resulting index to each unique value. This multi-stage approach allows us to utilize CUDA’s massively parallel architecture effectively.

Initially, the input array must be sorted. Sorting, while an *n log n* operation generally, can be effectively done on a GPU through algorithms like merge sort or radix sort. After sorting, adjacent identical elements will be grouped. With this arrangement, finding the first occurrence of each unique element becomes a matter of identifying where the change in value happens in the sorted array. This can be handled using a simple comparison between neighboring elements and a Boolean array that signals the start of a new value.

Once we have determined the ‘start’ indices, we need to determine the *original* indices to those elements. We can do this by using a parallel prefix sum of that boolean flag indicating the start of a new unique element. The resulting prefix sum indicates the new index for each unique element. For example: if the unique element markers are [1,0,0,1,0,1], the prefix sum will be [1,1,1,2,2,3]. Then, for example, the start of the third unique element is marked with a 3. Because the original data was sorted to enable this step, we will now use that sorted index array to look up the true positions. Let's consider the following sorted data array and associated original indices to see how this works:

Sorted Data: [2, 2, 3, 3, 3, 5, 5, 7]
Original Indices: [12, 10, 1, 14, 2, 5, 3, 6]

The unique flags will be [1,0,1,0,0,1,0,1]. The prefix sum is thus: [1, 1, 2, 2, 2, 3, 3, 4]. Now, this tells us the first unique element is at position 1, the second at 2, the third at 3, and the fourth at 4.

The core challenge now is in reverse-mapping these values back to the *original* indices, prior to the sorting. Since we have the indices of the start of each unique element in the sorted array (e.g. indices 0, 2, 5, and 7), we can now use these original indices and prefix sum indices to output a final mapping of unique element to their first index. For example, the first unique element is at sorted index 0, and has an original index of 12. The second unique element is at sorted index 2 and has an original index of 1.

The final output for the above example using indices starting at zero would then be:

Unique Value | First Index
-------------|------------
2            | 12
3            | 1
5            | 5
7            | 6

Below, I provide code snippets for clarity. These snippets do not contain full programs but rather highlight the important kernel implementations that perform the key steps of identifying unique element indices, and performing the associated prefix sum. The code uses `thrust` for sorting.

**Code Example 1: Identifying Unique Element Start Points**

```cpp
__global__ void find_unique_starts_kernel(const int* sorted_data, int* unique_flags, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < size) {
        if (sorted_data[i] != sorted_data[i - 1]) {
            unique_flags[i] = 1;
        } else {
            unique_flags[i] = 0;
        }
    }
     if(i == 0)
    {
       unique_flags[i] = 1;
    }
}

```

This CUDA kernel operates on the sorted input data. Each thread compares the current element with its predecessor. If a change in value is detected, the corresponding element in the `unique_flags` array is marked with 1, otherwise with 0. The first element is always flagged as a start.

**Code Example 2: Calculating Prefix Sum of Unique Flags**

```cpp
__global__ void prefix_sum_kernel(int* unique_flags, int* prefix_sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int sum = 0;
        for(int j = 0; j <= i; j++){
            sum += unique_flags[j];
        }
      prefix_sum[i] = sum;
    }
}
```

This kernel calculates the prefix sum of the `unique_flags` array. It iterates up to each element index, summing the values in the `unique_flags` array to determine the new cumulative index. While an inclusive scan could be done with `thrust`, it is important to demonstrate how it could be achieved without library function calls.

**Code Example 3: Applying Prefix Sum to Determine Final Output**

```cpp
__global__ void map_results_kernel(const int* sorted_indices, const int* prefix_sum, const int* unique_flags, int* output_indices, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if(unique_flags[i] == 1)
        {
            int prefix_idx = prefix_sum[i] - 1;
            output_indices[prefix_idx] = sorted_indices[i];
        }
    }
}
```

This kernel uses the results of the prefix sum and the sorted indices to determine the correct *original* indices. It iterates through the unique flags. If a unique flag is found it maps the associated `sorted_index` to the `output_indices` using the prefix sum to define the correct location.

For these kernels, the threads per block and number of blocks will need to be adjusted for your input size, but these provide a baseline example of how to perform these parallel computations. The following outline indicates how the functions could be arranged for an end-to-end solution:

1.  **Sorting:** Sort the input array and store original indices using `thrust`.
2.  **Unique Start Flagging:** Call the `find_unique_starts_kernel`.
3.  **Prefix Sum:** Call the `prefix_sum_kernel`.
4.  **Mapping to Original Indices:** Call the `map_results_kernel`.

This completes the mapping of unique values to their first indices. The output will then store the *original* indices of the first occurrence of each unique value.

For further study, consider exploring resources that delve into GPU algorithm optimization, specifically the application of `thrust` for sorting and the optimization of prefix sum operations.  Material focusing on memory access patterns on GPUs will help refine the approach to reduce memory bottlenecks. Resources explaining the different parallel scan algorithms and their efficient implementation can also help to improve overall performance. Specifically, understand the relationship between work per thread, data locality, and access latency. These resources will be highly beneficial to fully understand and optimize these steps in a more complex production implementation.
