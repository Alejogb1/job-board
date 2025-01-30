---
title: "Does Linux perf's call graph max-stack depth functionality function correctly?"
date: "2025-01-30"
id: "does-linux-perfs-call-graph-max-stack-depth-functionality"
---
The `perf record`'s call graph functionality, specifically its `-g` option with depth control via `--max-stack-depth`, occasionally exhibits unexpected behavior concerning the truncation of call chains.  My experience troubleshooting performance bottlenecks in complex, multi-threaded C++ applications has revealed inconsistencies in its reported call stacks, particularly when dealing with deeply nested function calls exceeding the specified depth. This isn't necessarily a bug in `perf` itself, but rather a consequence of the inherent limitations in accurately capturing and representing extremely deep call stacks within the sampling-based profiling methodology.


**1. Explanation of Perf's Call Graph and Depth Limitation**

`perf record -g` uses a statistical sampling approach to profile the application's execution.  It periodically interrupts the process and captures the call stack at that point in time.  The depth of the call stack is limited by both hardware and software factors.  Hardware limitations stem from the size of the registers used to store the return addresses in the stack frames.  Software limitations arise from the operating system's kernel and `perf`'s own implementation.

The `--max-stack-depth` option allows the user to control the maximum number of frames captured in each sampled stack trace.  Setting a lower depth reduces the overhead associated with stack unwinding, which is crucial for performance sensitive applications.  However, setting it too low will truncate potentially important information, leading to inaccurate call graph representation.  The critical point is that this truncation isn't necessarily a clean cut at the specified depth.  In my experience, I've observed instances where the call graph would be inconsistently truncated at depths *less* than the specified `--max-stack-depth`, particularly in scenarios with high stack frame contention or intricate thread interactions.  This often manifests as seemingly random omissions of crucial function calls near the top of the stack.

Furthermore, the accuracy of the call graph is intrinsically linked to the sampling rate.  A higher sampling rate improves the granularity of the profile but at the cost of increased overhead.  Conversely, a lower sampling rate reduces overhead but may lead to insufficient sampling to capture deeper call stacks reliably, even if the `--max-stack-depth` is set sufficiently high.  The interaction of sampling rate and `--max-stack-depth` contributes significantly to the observed inconsistencies.  In high-frequency, short-lived function calls, deeper stack frames may simply not be captured at all, regardless of the specified depth.

Finally, signal handling and asynchronous events can interrupt stack unwinding, leading to incomplete or corrupted call stacks.  This is especially relevant in applications that handle signals extensively or rely on asynchronous I/O operations.  The interaction between signal handling and stack unwinding can cause the `perf` sampling to miss intermediate frames, resulting in truncated stacks that appear to be shallower than expected.


**2. Code Examples and Commentary**

The following examples illustrate potential scenarios that highlight the limitations of `--max-stack-depth`. These examples assume a C++ environment, reflective of my primary experience, but the underlying principles apply to other languages as well.

**Example 1: Deeply Nested Recursion**

```c++
#include <iostream>

int recursive_function(int n) {
  if (n == 0) return 0;
  return recursive_function(n - 1) + 1;
}

int main() {
  recursive_function(1000); // Deeply nested recursion
  return 0;
}
```

Profiling this code with `perf record -g --max-stack-depth 500` might yield incomplete call stacks. Even though the specified depth should allow for the entire call stack to be captured, the recursive calls might exceed the effective depth due to stack unwinding overhead or signal interruptions.  The resulting flame graph might only show a partial view of the `recursive_function` calls.


**Example 2: Multi-threaded Application with High Stack Frame Contention**

```c++
#include <thread>
#include <vector>

void worker_function(int id) {
  // Intensive computations creating a potentially large stack frame
  std::vector<int> large_vector(100000);
  // ... more code ...
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; ++i) {
    threads.push_back(std::thread(worker_function, i));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  return 0;
}
```

In this multi-threaded example, high stack frame contention among threads might lead to inaccurate call stack sampling.  The `perf` sampler might capture only a subset of the complete stack frames, leading to a truncated view of the `worker_function` calls, even with an adequately high `--max-stack-depth`.


**Example 3:  Asynchronous Operations and Signal Handling**

```c++
#include <iostream>
#include <signal.h>
#include <unistd.h>

void signal_handler(int signal) {
  std::cout << "Signal received: " << signal << std::endl;
}

int main() {
  signal(SIGINT, signal_handler);
  // ... intensive computations with potential signals ...
  pause();  // Wait for signal
  return 0;
}
```

Here, the signal handler's execution might interrupt the stack unwinding process during `perf`'s sampling, producing an incomplete call stack.  The call stack related to the "intensive computations" might appear truncated even if the `--max-stack-depth` is set considerably higher.



**3. Resource Recommendations**

To further understand the complexities of profiling, consult the official `perf` documentation.   Explore the detailed descriptions of the sampling process and options related to call graph generation.  Examine materials on system-level programming and process management for a deeper understanding of how signals and threads interact with the operating system.  Study advanced profiling techniques and consider alternative profiling tools like VTune Amplifier or gprof for comparison and validation of your findings.  These resources should provide a comprehensive understanding of the limitations and potential inaccuracies involved in using `perf`'s call graph profiling functionality, especially concerning deeply nested call chains and the `--max-stack-depth` parameter.
