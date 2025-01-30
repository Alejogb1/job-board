---
title: "Why did Vitis HLS 2020.2 pre-synthesis fail without error messages?"
date: "2025-01-30"
id: "why-did-vitis-hls-20202-pre-synthesis-fail-without"
---
Vitis HLS 2020.2, under specific circumstances, can indeed fail pre-synthesis without providing explicit error messages. This frustrating behavior is typically rooted in subtle interactions between the design constraints, tool limitations, and the complex interplay of the various hardware resources the high-level synthesis tool attempts to manage. From my experience developing custom accelerators, I’ve seen this occur more frequently when design complexity pushes the boundaries of what the tool can readily interpret or when external libraries are introduced without proper directives.

The core issue stems from the pre-synthesis phase acting as an initial feasibility check. During this stage, Vitis HLS performs a series of rapid analyses to assess the viability of transforming the C/C++ or OpenCL code into a hardware description language like RTL. This analysis includes dependency analysis, resource estimation, and basic hardware scheduling heuristics. When these initial checks encounter fundamental contradictions or resource conflicts that the tool cannot resolve implicitly, it can abort the process without emitting a verbose error. Instead of producing a detailed report, the tool essentially concludes that the design as presented is not amenable to synthesis without modification. The reasons for these silent failures can be broadly categorized into resource over-utilization, ill-defined dependencies, or unsupported language constructs in particular contexts.

Resource over-utilization, particularly with aggressively optimized designs, is a common culprit. HLS relies heavily on target device resources like DSP slices, block RAM, and logic cells. If the HLS design, either through poorly optimized coding or unrealistic directives, demands more resources than available on the specified target platform before any detailed optimization is carried out, pre-synthesis can silently fail. This can happen even before memory allocation is addressed, which further complicates the issue.

Another source of this behavior arises from the intricacies of data dependencies. For example, poorly defined or highly irregular data access patterns can lead to complex memory interleaving and banking issues that Vitis HLS may not readily handle in the pre-synthesis stage. The tool evaluates data flow patterns, trying to identify parallelism, and if the data stream is too complex, it may fail before emitting an error. Similarly, when complex control logic or intricate loops create situations that are difficult to pipeline or schedule, the preliminary stages might fail due to perceived resource conflicts or infeasibility for direct hardware mapping without specific directives.

Furthermore, using external libraries or specialized function calls not explicitly recognized by HLS can also trigger silent pre-synthesis failures. Vitis HLS needs guidance to interpret the behavior of these functions in a hardware context, especially when they interact with memory or have side effects. Failure to provide appropriate directives can cause the pre-synthesis check to determine the design is too complex to process without any indication.

Let me illustrate with code examples and commentary:

**Example 1: Unbounded Resource Requirements**

Consider a scenario where an array is declared with a dynamically allocated size that is extremely large:

```cpp
#include <stdlib.h>
#include <stdint.h>
#define SIZE_FACTOR 1000000
void large_array_operation(int size_input) {
  int size = size_input * SIZE_FACTOR;
  uint32_t *large_array = (uint32_t *)malloc(size * sizeof(uint32_t));
  // ... some operations on large_array, for example
  for(int i = 0; i < size; i++) {
    large_array[i] = i;
  }
  free(large_array);
}
```

**Commentary:** While the `malloc` is present, Vitis HLS pre-synthesis does not fully account for its impact on hardware, but when it attempts an initial resource estimate, particularly for the size calculation, it might not find a readily-available solution within the tool's limitations. This doesn't generate a specific error, but simply halts pre-synthesis silently as the potential resource usage is deemed outside the bounds of feasible synthesis before any optimization is performed. Even though `malloc` is a standard C function, HLS often needs an explicit mapping of dynamic memory allocation onto hardware resources which are typically fixed in size. Even when the data size is reduced, if the data access within the loop is too complex with multiple address calculations, it can still silently halt.

**Example 2: Complex Data Dependencies and No Pipelining**

Here’s a code snippet with complex data access patterns:

```cpp
#include <stdint.h>
void complex_dependency(uint32_t *in_array, uint32_t *out_array, int size) {
  for (int i = 0; i < size; i++) {
    out_array[i] = in_array[(i * i) % size];
  }
}
```

**Commentary:** This code uses a complex address calculation, where every access to `in_array` is dependent on `i*i mod size`. This can prevent efficient loop pipelining during hardware implementation. While it is functionally correct in software, the tool, in its pre-synthesis phase, might determine that the data dependencies and memory access are too complex to create a viable hardware implementation without specific memory directives or loop unrolling instructions. The implicit assumption is that loop access must have predictable and straightforward dependencies for hardware pipelining. Without those, the synthesis can silently stop.

**Example 3: External Library Use without HLS Directives**

Consider a situation using an external mathematical library:

```cpp
#include <cmath>
#include <stdint.h>
void external_function_call(double input, double *output) {
  *output = std::sin(input);
}
```

**Commentary:** While `std::sin` is commonly used, Vitis HLS might lack a direct hardware mapping for this function, especially in its pre-synthesis phase. This is because HLS needs to know the exact resource use (e.g. lookup tables, DSP blocks). The function isn’t simply synthesizable directly to hardware in the same way as basic arithmetic. Without explicit instruction through pragmas to tell the tool how to handle this standard function, the pre-synthesis process can halt because it cannot determine how to synthesize the `std::sin` function and the tool gives no error, instead it just fails silently.

To mitigate these silent pre-synthesis failures, a systematic approach involving incremental code changes and judicious use of HLS directives is crucial. I recommend focusing on the following strategies:

1.  **Resource Constraints:** First, start by carefully assessing the resource consumption of the design. Refrain from overly aggressive optimizations during the initial stages. Use resource directives sparingly and iteratively. Try simplifying complex data structures, or re-implementing the complex memory access patterns that cause non-deterministic access with more straightforward algorithms. Begin by removing dynamic memory allocations using `malloc` and instead use static arrays.

2.  **Pipeline Directives:** Explicitly specify loop pipelining pragmas, starting with modest initiation intervals. For complex loops, it is crucial to analyze data dependencies carefully and possibly refactor the code to remove loop carried dependencies or introduce loop unrolling.

3.  **Library Abstraction:** If using external libraries, explore methods to either implement the functionality with HLS-compatible code or create a wrapper with HLS-recognizable interfaces with explicit resource and latency information. Use `interface` pragmas to specify how the function arguments will map to hardware ports.

4. **Incremental Synthesis:** Reduce code complexity by performing synthesis in an incremental approach by synthesizing small sections of the design first and then reintroducing complexity one step at a time.

5.  **Detailed Profiling:**  Employ profiling tools (both software and hardware where possible) to identify bottlenecks, pinpoint memory access issues, and gain a deeper understanding of data flow, this can be particularly useful to help simplify the system and allow pre-synthesis to complete successfully.

For comprehensive learning material, I suggest reviewing the Vitis HLS documentation provided by AMD (formerly Xilinx) and consider working through design examples included with the tool suite. Also, study the best practice guides from AMD on high-level synthesis methods for complex hardware implementations. Academic literature on high-level synthesis techniques and hardware architecture can provide deeper insight into the fundamental principles behind the HLS transformation process.
