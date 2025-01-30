---
title: "Why does MPI send/recv latency increase at 32 MiB message size?"
date: "2025-01-30"
id: "why-does-mpi-sendrecv-latency-increase-at-32"
---
The observed increase in MPI `send`/`recv` latency around the 32 MiB message size threshold is a common phenomenon I've encountered throughout my years working on high-performance computing clusters.  It's rarely a single, easily identifiable cause, but rather a confluence of factors related to underlying hardware and software interactions.  The critical point is the transition from utilizing highly optimized, cache-coherent memory access strategies to a reliance on significantly slower, non-coherent data transfer mechanisms.

My experience with this issue across various clusters points to a few key contributing factors.  Firstly, for messages smaller than the L3 cache size (often around 32 MiB on many contemporary architectures), the data involved is likely to reside entirely within the cache hierarchy of the sending and receiving processes. This allows for incredibly fast communication leveraging the processor's internal, high-bandwidth pathways. The latency is primarily determined by the overhead of the MPI communication primitives themselves.

However, once the message size surpasses the L3 cache capacity, data spilling over into main memory becomes inevitable.  This triggers a cascade of performance penalties. Data must be transferred from the cache to main memory, potentially requiring multiple cache line evictions, creating contention on the memory bus, and ultimately leading to slower access times.  Furthermore, the increased data volume necessitates the utilization of different data transfer mechanisms within the MPI implementation, typically relying on system-level DMA (Direct Memory Access) operations.  While DMA is designed for high throughput, its inherent latency is generally higher compared to cache-based transfers.  Finally, the network interface card (NIC) and the network itself become significant bottlenecks as larger chunks of data need to be transferred across the network fabric, introducing network latency, bandwidth limitations, and potentially network congestion as a factor.

Let's illustrate this with code examples focusing on different aspects of the problem.  These examples assume a basic familiarity with MPI programming and a standard MPI implementation like OpenMPI or MPICH.

**Example 1:  Measuring Latency with Varying Message Sizes**

This example directly measures the latency of MPI `send`/`recv` operations for various message sizes, including the critical 32 MiB region.  We measure the time taken for a single send-receive operation to highlight the latency changes.

```c++
#include <mpi.h>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    std::cerr << "This example requires exactly two processes." << std::endl;
    MPI_Finalize();
    return 1;
  }

  for (size_t msg_size = 1024; msg_size <= 67108864; msg_size *= 2) { // Sizes from 1KB to 64MB
    char* buffer = new char[msg_size];
    auto start = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
      MPI_Send(buffer, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    } else {
      MPI_Recv(buffer, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (rank == 1) {
      std::cout << "Message size: " << msg_size << " bytes, Latency: " << duration.count() << " microseconds" << std::endl;
    }
    delete[] buffer;
  }

  MPI_Finalize();
  return 0;
}
```

This code demonstrates a straightforward latency measurement. Note the exponential increase in message size, strategically chosen to observe the behavior around the 32 MiB threshold.  The output clearly shows the increase in latency as the message size grows beyond the cache size.


**Example 2:  Impact of Data Alignment and Memory Access Patterns**

This example demonstrates how memory access patterns can influence the performance, particularly for larger messages exceeding the cache size.

```c++
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    // ... (MPI Initialization as in Example 1) ...

    size_t msg_size = 33554432; // 32MB + 1MB for illustration

    std::vector<char> buffer(msg_size);
    auto start = std::chrono::high_resolution_clock::now();

    // ... (MPI Send/Recv as in Example 1) ...

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (rank == 1) {
        std::cout << "Message size: " << msg_size << " bytes, Latency: " << duration.count() << " microseconds" << std::endl;
    }

    //Simulate non-contiguous access for comparison (comment out for standard case)
    //for(size_t i = 0; i < msg_size; i+=1024){
    //  char temp = buffer[i];
    //}

    MPI_Finalize();
    return 0;
}
```

Uncommenting the `for` loop simulates non-contiguous memory access.  Running this with and without the loop will highlight the impact of cache misses and memory access patterns on the overall latency.  This emphasizes the importance of data alignment and efficient memory access strategies.  Poorly aligned data can exacerbate cache misses, significantly impacting larger message transfers.


**Example 3:  Investigating the Role of Network Bandwidth**

This example explores the impact of network bandwidth on larger message transfers.

```c++
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <cstring> //for memset

int main(int argc, char** argv) {
    // ... (MPI Initialization as in Example 1) ...

    size_t msg_size = 33554432; // 32MB

    char* buffer = new char[msg_size];
    memset(buffer, 'A', msg_size); // Fill buffer to ensure data transfer

    auto start = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        MPI_Send(buffer, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buffer, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (rank == 1) {
        std::cout << "Message size: " << msg_size << " bytes, Latency: " << duration.count() << " microseconds" << std::endl;
    }
    delete[] buffer;
    MPI_Finalize();
    return 0;
}
```

This example is similar to the first but includes `memset` to explicitly fill the buffer with data.  This helps ensure that the data is actually transferred across the network, highlighting the influence of network bandwidth constraints as a contributing factor to the observed latency increase.  On high-latency networks, this effect will be even more pronounced.


**Resource Recommendations:**

For a deeper understanding, I recommend consulting the MPI standard documentation, advanced MPI programming textbooks, and performance analysis tools specific to your MPI implementation.  Analyzing MPI performance using tools like `mpitrace` or similar profilers will offer valuable insights into the bottlenecks. Understanding the specifics of your hardware's cache hierarchy and memory architecture is also crucial for effective performance tuning.  Studying the underlying networking infrastructure and its limitations are equally important.  Finally, examining the documentation for your specific MPI implementation can reveal optimization strategies and potential limitations.
