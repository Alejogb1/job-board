---
title: "What do Google's CPU performance tools measure in C++ profiling?"
date: "2025-01-26"
id: "what-do-googles-cpu-performance-tools-measure-in-c-profiling"
---

My experience optimizing high-throughput trading systems has taught me that understanding CPU performance at the micro level is critical, especially when working with latency-sensitive C++ applications. Google's profiling tools, like the ones integrated with the Google Performance Tools (gperftools) suite, measure a variety of CPU-centric metrics that allow developers to pinpoint performance bottlenecks. These measurements fall broadly into two categories: sampling-based and instrumentation-based profiling. Each method provides different insights into program execution.

Sampling-based profiling, often employed by tools like `pprof`, operates by periodically interrupting program execution and recording the current instruction pointer (IP) and the call stack. This creates a statistical representation of time spent in different parts of the code. The granularity of this information is tied to the sampling frequency, which is usually measured in Hertz (Hz). Higher sampling rates offer finer-grained data, but they also add overhead to program execution. A primary advantage of sampling is its low intrusion, meaning the profiling process itself does not drastically alter the performance characteristics of the code being profiled. The data collected typically represents the percentage of time the CPU spends executing specific functions or lines of code. This is highly useful for identifying "hot spots"—code regions consuming the most CPU cycles. The sampling method, however, may miss short-lived functions or events, particularly when using a lower sampling rate. Also, the sampled data provides only an approximation of the actual CPU time spent in different parts of the code. In essence, sampled profiling can identify areas of concern and guide further, more targeted investigation.

Instrumentation-based profiling, exemplified by tools that use function entry/exit hooks, collects much more precise data. This method injects code at the beginning and end of each function that the developer wants to monitor. Upon entering a profiled function, a timestamp is recorded, and another timestamp is recorded upon exit. The difference gives the actual time spent within the function. This allows for accurate measurements of individual function execution times, including very short-lived functions not generally captured by sampling. Additionally, call counts, average execution times, and total times spent within a function can be accurately computed from the collected data. The primary drawback of instrumentation profiling is its intrusive nature. Inserting these hooks adds overhead to the program execution, which can distort the very timing it aims to measure. This is particularly problematic in performance-critical applications, as the added instrumentation can significantly alter performance characteristics and obscure bottlenecks. Despite this, for functions whose performance is critical and needs accurate measurements, instrumentation offers better results. Google’s tools often use a combination of these methods. For example, `pprof` can operate in a sampling-based mode but can also utilize instrumentation for more precise data in certain situations.

The specific metrics Google's tools report include, but are not limited to: CPU time per function (both self-time, and cumulative-time including children function calls), call counts, number of samples in each function, cache misses (if hardware counters are available), branch mispredictions (again if hardware counters are available), and call graphs. Call graphs are diagrams depicting the call relationship between functions and can be highly effective in visually understanding program flow. This information is essential to detect inefficiencies, such as deeply nested call stacks, code that is unnecessarily executed frequently, and areas where hardware resources such as the CPU cache are underperforming.

Here are examples demonstrating how these profiling measurements are generated using a fictional profiling API:

**Example 1: Sampling Profiling**

```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "fake_profiler.h" // Assume this contains fake profiling APIs


void compute(std::vector<int>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = data[i] * data[i] + data[i]/2;
    }
}

int main() {
    std::vector<int> numbers(1000000, 5);
    FakeProfiler::startSampling(100);  // 100 Hz sampling rate

    for (int i = 0; i < 10; ++i) {
        compute(numbers);
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate work and delay
    }

    FakeProfiler::stopSampling();
    FakeProfiler::dumpSamplingData("sampling_profile.txt");
    return 0;
}
```

*Commentary:* This example simulates a profiling scenario using a fictional API. The `FakeProfiler::startSampling(100)` initiates the sampling process at 100 Hz.  The `compute` function, representing the workload, is invoked repeatedly. After computation, `FakeProfiler::stopSampling()` ends the profiler and `FakeProfiler::dumpSamplingData()` writes the collected data to a file, typically containing the call stack, instruction pointer, and counts of how many times each function was observed during a sample. The primary metrics reported in the `sampling_profile.txt` will be time spent in `compute` and possibly in `std::vector` related routines.

**Example 2: Instrumentation Profiling**

```cpp
#include <iostream>
#include <chrono>
#include "fake_profiler.h" // Assuming this provides profiling APIs

int slowFunction() {
    FakeProfiler::startInstrumentation("slowFunction"); // Start timing
    int sum = 0;
    for (int i = 0; i < 10000000; ++i) {
      sum += i;
    }
    FakeProfiler::stopInstrumentation("slowFunction"); // Stop timing
    return sum;
}


int fastFunction(){
   FakeProfiler::startInstrumentation("fastFunction"); // Start timing
   int sum = 0;
   for (int i = 0; i < 100; ++i){
      sum+=i;
   }
   FakeProfiler::stopInstrumentation("fastFunction"); // Stop timing
   return sum;
}

int main() {
    for(int i = 0; i < 10; i++){
      slowFunction();
    }
    for(int i = 0; i < 100; ++i){
        fastFunction();
    }

    FakeProfiler::dumpInstrumentationData("instrumentation_profile.txt"); // Write out profile data
    return 0;
}
```

*Commentary:* This example showcases instrumentation. `FakeProfiler::startInstrumentation` and `FakeProfiler::stopInstrumentation` record the start and end times of the `slowFunction` and `fastFunction`.  The data stored in `instrumentation_profile.txt` will include the exact execution time for each function call, call counts, average times, minimum times, maximum times, and potentially other relevant statistics. Notice `slowFunction`'s for loop is much larger then `fastFunction`'s. The file will make clear that `slowFunction` executes for significantly longer and should be analyzed to see if this run time can be improved.

**Example 3: Call Graph Profiling**

```cpp
#include <iostream>
#include "fake_profiler.h" // Fictional Profiler

void innermost() {
  FakeProfiler::startInstrumentation("innermost");
  int sum = 0;
  for(int i = 0; i < 10; ++i){
    sum+=i;
  }
  FakeProfiler::stopInstrumentation("innermost");
}
void middle() {
  FakeProfiler::startInstrumentation("middle");
  for(int i = 0; i < 10; ++i){
    innermost();
  }
    FakeProfiler::stopInstrumentation("middle");
}

void outer() {
  FakeProfiler::startInstrumentation("outer");
  for(int i = 0; i < 5; ++i){
      middle();
  }
    FakeProfiler::stopInstrumentation("outer");
}


int main() {
    outer();
    FakeProfiler::dumpCallGraph("call_graph.txt"); // Generate call graph data
    return 0;
}

```

*Commentary:*  Here, `outer` calls `middle`, which then calls `innermost`. The call graph file, `call_graph.txt`, generated by `FakeProfiler::dumpCallGraph` will capture this function calling hierarchy and the time spent within each function. This representation allows for quick identification of functions that are called recursively or whose performance impacts the overall execution time of functions higher up in the call stack. Call graphs are useful for visualizing the program’s control flow and finding opportunities to reduce work or avoid deep calls.

For a deeper understanding of C++ performance and optimization techniques, I recommend referring to Scott Meyers' "Effective C++" series for best practices. For understanding CPU architecture and low level performance characteristics, "Computer Organization and Design" by Patterson and Hennessy can be useful. Google’s open source documentation for gperftools provides valuable resources for understanding and using their profiling utilities.  These resources, along with careful experimentation with profiling tools, can significantly improve the performance of C++ applications.
