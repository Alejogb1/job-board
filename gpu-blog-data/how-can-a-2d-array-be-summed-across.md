---
title: "How can a 2D array be summed across multiple processors using MPI?"
date: "2025-01-30"
id: "how-can-a-2d-array-be-summed-across"
---
Distributing the summation of a 2D array across multiple processors using MPI necessitates careful consideration of data decomposition and communication. My experience optimizing large-scale numerical simulations has shown that efficiency hinges on minimizing inter-process communication while maintaining a balanced workload. A straightforward approach is to distribute the array rows amongst the available processors, calculate partial sums locally, and then reduce these partial sums to obtain the global sum.

The initial challenge is partitioning the array. Given an *m x n* array and *p* processors, I’ve found that assigning roughly *m/p* rows to each processor provides an adequate balance, provided *m* is considerably larger than *p*. In scenarios where *m* isn’t a multiple of *p*, some processors may end up with one extra row. While sophisticated load balancing algorithms exist, for many practical cases, this simple distribution works well. Each processor will then compute a local sum of its assigned elements. Crucially, this local summation step can be executed in parallel with no inter-process communication. The final step is to collect the partial sums and combine them to obtain the total sum; this reduction is achieved through MPI’s collective communication mechanisms.

Let me illustrate this process with some practical code examples. These examples will be in C++ using the MPI library, as that is what I have the most experience with.

**Example 1: Basic Row Distribution**

This first example outlines the basic approach where we partition the rows amongst the processes, compute partial sums, and then collect the final sum.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int m = 1000; // Number of rows
    const int n = 500;  // Number of columns
    std::vector<std::vector<int>> array;

    if (world_rank == 0) {
        // Only root process creates the array
        array.resize(m, std::vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              array[i][j] = i + j;  // arbitrary values
            }
        }
    }

    int rows_per_process = m / world_size;
    int remainder = m % world_size;
    int start_row = world_rank * rows_per_process + std::min(world_rank, remainder);
    int end_row = start_row + rows_per_process;
    if (world_rank < remainder) {
        end_row++;
    }

    std::vector<int> local_array;
    int local_size = (end_row - start_row) * n;

    if (world_rank == 0){
    for (int i = 0; i < world_size; i++){
        int local_rows = rows_per_process;
        int local_start = i * rows_per_process + std::min(i, remainder);
         if (i < remainder) {
             local_rows++;
          }
      
        if (i > 0){ //send chunks of array to each processor
            int chunk_size = local_rows * n;
            std::vector<int> temp_chunk;
            temp_chunk.reserve(chunk_size);
                for (int k = local_start; k < local_start + local_rows; k++)
                {
                    temp_chunk.insert(temp_chunk.end(), array[k].begin(), array[k].end());
                }
            MPI_Send(temp_chunk.data(), chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
         }
        else {
            local_array.reserve(local_size);
            for (int k = start_row; k < end_row; k++)
            {
                local_array.insert(local_array.end(), array[k].begin(), array[k].end());
            }

        }

        }
    }
    else {
        local_array.resize(local_size);
        MPI_Recv(local_array.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    long long local_sum = std::accumulate(local_array.begin(), local_array.end(), 0LL);

    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Total sum: " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
```

In this code, process 0 generates the array, calculates each process’s assigned rows, and then dispatches the relevant data to all other processes using point-to-point communication. The root process also retains its own section of the array. Each process computes its partial sum. The `MPI_Reduce` call then gathers and adds the partial sums across all processes, with the final result stored only at process 0. I prefer to avoid broadcasting the array to each process individually since that can quickly become a bottleneck.

**Example 2: Using MPI_Scatterv/MPI_Gatherv**

A more scalable method uses `MPI_Scatterv` and `MPI_Gatherv` for data distribution and collection. This can be more performant, particularly with larger processor counts, as it leverages optimized, collective communication routines. In my experience, these routines tend to be better tuned for performance within the MPI implementation than point-to-point operations.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int m = 1000;
    const int n = 500;
    std::vector<std::vector<int>> array;


    std::vector<int> send_counts(world_size);
    std::vector<int> send_displs(world_size);

    int rows_per_process = m / world_size;
    int remainder = m % world_size;
    int displ = 0;

    for (int i = 0; i < world_size; ++i) {
        int local_rows = rows_per_process;
        if (i < remainder) {
            local_rows++;
        }
       send_counts[i] = local_rows * n;
       send_displs[i] = displ;
       displ += send_counts[i];
    }

    std::vector<int> flattened_array;
    if (world_rank == 0)
    {
        array.resize(m, std::vector<int>(n));
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
             array[i][j] = i + j;
          }
        }

        flattened_array.reserve(m * n);
        for (const auto &row: array){
            flattened_array.insert(flattened_array.end(), row.begin(), row.end());
        }
    }

    int local_size = send_counts[world_rank];
    std::vector<int> local_array(local_size);


    MPI_Scatterv(flattened_array.data(), send_counts.data(), send_displs.data(), MPI_INT, local_array.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);


    long long local_sum = std::accumulate(local_array.begin(), local_array.end(), 0LL);

    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "Total sum: " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
```

In this second example, rather than sending chunks iteratively, I use `MPI_Scatterv`. Note how I flattened the 2D array for this to work effectively with MPI functions. The `send_counts` array details the number of integers sent to each process, and `send_displs` specifies offsets within the flattened array. I find the use of `MPI_Scatterv` and similar routines cleaner and more robust for parallel data distribution compared to manual send-receive loops. The remainder of the code follows the same principle: local sum calculation and global reduction.

**Example 3: Using MPI_Allreduce and Avoiding Root Process Gather**

Often, it is more valuable for each process to know the global sum, rather than just process 0. `MPI_Allreduce` achieves this, providing a more distributed result and often improves performance in subsequent computations. It eliminates the data dependency of relying only on the root process.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   const int m = 1000;
    const int n = 500;
    std::vector<std::vector<int>> array;


    std::vector<int> send_counts(world_size);
    std::vector<int> send_displs(world_size);

    int rows_per_process = m / world_size;
    int remainder = m % world_size;
    int displ = 0;

    for (int i = 0; i < world_size; ++i) {
        int local_rows = rows_per_process;
        if (i < remainder) {
            local_rows++;
        }
       send_counts[i] = local_rows * n;
       send_displs[i] = displ;
       displ += send_counts[i];
    }

    std::vector<int> flattened_array;
    if (world_rank == 0)
    {
         array.resize(m, std::vector<int>(n));
         for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                array[i][j] = i + j;
             }
         }
        flattened_array.reserve(m * n);
        for (const auto &row: array){
             flattened_array.insert(flattened_array.end(), row.begin(), row.end());
        }
    }

    int local_size = send_counts[world_rank];
    std::vector<int> local_array(local_size);
    MPI_Scatterv(flattened_array.data(), send_counts.data(), send_displs.data(), MPI_INT, local_array.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);


    long long local_sum = std::accumulate(local_array.begin(), local_array.end(), 0LL);

    long long global_sum = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    std::cout << "Process " << world_rank << ", total sum: " << global_sum << std::endl;

    MPI_Finalize();
    return 0;
}
```

This final example illustrates the use of `MPI_Allreduce`. With this function, the `global_sum` variable is updated on *every* process. This pattern proves very useful in scenarios requiring distributed calculation across all nodes.

For further learning about distributed array calculations and MPI, I would suggest consulting the following resources.  The book "Using MPI" provides a comprehensive introduction to the standard and its best practices.  "Parallel Programming in C with MPI and OpenMP" offers another helpful text and covers a broader range of parallel programming techniques. Furthermore, the online documentation for your chosen MPI implementation, usually provided with your operating system or HPC cluster, provides the most specific and up-to-date information on function signatures and optimization advice. Examining example codes from numerical libraries using MPI, like PETSc or Trilinos, can also provide valuable practical knowledge.
