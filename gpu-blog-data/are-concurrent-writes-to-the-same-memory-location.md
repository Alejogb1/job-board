---
title: "Are concurrent writes to the same memory location atomic?"
date: "2025-01-30"
id: "are-concurrent-writes-to-the-same-memory-location"
---
The assumption that concurrent writes to the same memory location are inherently atomic is a significant source of errors in multithreaded programming. Atomicity, in the context of memory operations, signifies an operation that executes indivisibly, without interruption from other threads. In most common hardware architectures and operating systems, standard write operations, particularly those involving data larger than a word size, are *not* atomic by default.

My experience debugging race conditions in high-throughput data processing pipelines has repeatedly underscored this point. When multiple threads attempt to modify the same shared variable concurrently, the resulting state can be unpredictable and often inconsistent, deviating substantially from expected behavior. This stems from the fundamental way modern CPUs and memory systems interact.

A standard write operation, for instance, incrementing an integer, often involves a read-modify-write sequence: the CPU first reads the current value from memory into a register, increments it, and then writes the updated value back to memory. These operations, while logically a single "increment," are not atomic. If two threads attempt this concurrently, the following can occur:

1. **Thread A** reads the value `X` from memory.
2. **Thread B** reads the value `X` from memory.
3. **Thread A** increments the value in its register to `X + 1`.
4. **Thread B** increments the value in its register to `X + 1`.
5. **Thread A** writes `X + 1` back to memory.
6. **Thread B** writes `X + 1` back to memory.

Instead of incrementing twice (resulting in `X + 2`), the value in memory is incremented only once. This is a fundamental race condition. The outcome is not deterministic and depends on the unpredictable timing of threads, making debugging exceptionally challenging. The root cause is the lack of atomicity in the write operation.

However, the word size of a machine often comes into play. For example, on a 64-bit architecture, writing a 64-bit integer *can* be atomic under certain circumstances, if the address is properly aligned. The underlying CPU architecture will guarantee atomicity at the hardware level for a write to memory whose size is no larger than its word size. However, these conditions aren't guaranteed across all platforms or when dealing with higher-level constructs (e.g., structures, objects, or strings). Crucially, this atomicity at the machine word level is often *not* sufficient for implementing complex data structures or algorithms. Therefore, relying on it is dangerous and often leads to fragile code.

To illustrate these points further and show ways to avoid these issues, consider the following examples written in C++ (chosen for its low-level control and widely applicable principles).

**Example 1: Non-Atomic Increment**

```c++
#include <iostream>
#include <thread>
#include <vector>

int shared_counter = 0;

void increment_counter() {
  for (int i = 0; i < 100000; ++i) {
    shared_counter++;
  }
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back(increment_counter);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::cout << "Final counter value: " << shared_counter << std::endl;
  return 0;
}
```

This code launches four threads, each of which increments the `shared_counter` 100,000 times. If writes were atomic, the final value should be 400,000. However, in practice, due to the race condition as explained, the result will almost certainly be less. This demonstrates the direct consequence of relying on non-atomic writes. The discrepancy is not simply a matter of lost increments; it's an example of data corruption. The precise value will vary between executions, a hallmark of race conditions.

**Example 2: Atomic Increment with Mutex**

```c++
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>

int shared_counter = 0;
std::mutex counter_mutex;

void increment_counter() {
  for (int i = 0; i < 100000; ++i) {
    std::lock_guard<std::mutex> lock(counter_mutex);
    shared_counter++;
  }
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back(increment_counter);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::cout << "Final counter value: " << shared_counter << std::endl;
  return 0;
}
```

This example introduces a mutex (`counter_mutex`). Each thread now acquires the mutex's lock before incrementing `shared_counter`, effectively serializing access to it. Only one thread at a time can hold the lock, guaranteeing atomicity for the entire increment operation – read, modify, and write, as a single unit, from all other threads’ perspectives. The final value will now reliably be 400,000. This is a fundamental way to protect shared resources, a core technique in concurrent programming. However, mutexes come with their own overhead and can introduce issues such as deadlocks if not handled carefully.

**Example 3: Atomic Operation (C++11 and later)**

```c++
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

std::atomic<int> shared_counter{0}; // Initialize to 0

void increment_counter() {
  for (int i = 0; i < 100000; ++i) {
    shared_counter++;
  }
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back(increment_counter);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::cout << "Final counter value: " << shared_counter << std::endl;
  return 0;
}
```

This version utilizes `std::atomic<int>`. `std::atomic` provides a type wrapper that guarantees that operations on it are atomic. Incrementing using the overloaded ++ operator on an atomic integer will happen as one indivisible action. This has better performance compared to the mutex approach and is often preferable when the required operation is a common one, such as incrementing, decrementing, or exchange. This approach leverages CPU instructions explicitly designed for atomic operations, which are usually faster than mutex locking and unlocking, and are simpler to implement correctly. It achieves the same goal as Example 2, but with a more efficient mechanism.

In summary, writing concurrently to the same memory location is inherently non-atomic for common datatypes across typical system architectures. To implement correct concurrent programs, one must use appropriate synchronization primitives to enforce atomicity of operations.

For further study, I recommend the following resources:
*   Textbooks focusing on operating system principles and concurrent programming.
*   Documentation for synchronization primitives specific to your programming language (e.g. mutexes, condition variables, semaphores, and atomic operations).
*   Materials on race conditions and data races specifically. Understanding the theory behind synchronization will lead to robust and correct code. Focus on understanding the different types of atomic and lock-free operations in the particular language you’re working with.
*   Practical exercises implementing concurrent algorithms using a range of synchronization primitives. The experience will be essential for practical programming.

Careful consideration and appropriate synchronization techniques are necessary to avoid data races and maintain the integrity of concurrent programs.
