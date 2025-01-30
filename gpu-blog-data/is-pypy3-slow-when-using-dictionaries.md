---
title: "Is Pypy3 slow when using dictionaries?"
date: "2025-01-30"
id: "is-pypy3-slow-when-using-dictionaries"
---
The perceived performance discrepancy surrounding PyPy3's dictionary handling stems primarily from the implementation details of its Just-In-Time (JIT) compiler and its interaction with dictionary operations, particularly those involving frequent key lookups and modifications within computationally intensive loops.  My experience optimizing computationally demanding Python applications for scientific computing has frequently highlighted this nuance. While PyPy3 generally offers significant speed improvements over CPython, certain microbenchmarks and real-world scenarios may reveal scenarios where the gains are less pronounced, or even where CPython exhibits superior performance for dictionary-heavy tasks. This is not indicative of inherent slowness in PyPy3's dictionary implementation, but rather a consequence of the trade-offs inherent in JIT compilation and its limitations in optimizing certain dictionary-based patterns.

**1. Clear Explanation:**

PyPy3 employs a tracing JIT compiler. This means it observes the execution path of your code during runtime, identifies frequently executed code blocks (often loops), and generates optimized machine code for these specific blocks.  Dictionaries in CPython are implemented using a hash table.  While PyPy3 also uses a hash table, the JIT compiler's optimization strategies may not always perfectly translate the performance characteristics of CPython's dictionary operations.  Specifically, the JIT compiler's effectiveness is highly dependent on the predictability of the program's control flow.  If the code exhibits highly dynamic behavior, such as unpredictable key accesses in a dictionary, the JIT compiler may struggle to generate highly efficient machine code.  This results in less dramatic performance improvements compared to scenarios with more predictable code patterns. Furthermore, the overhead of the JIT compilation itself can impact initial execution speeds, especially for smaller programs where the overhead outweighs the benefits of optimized code.  In such cases, the JIT's overhead in dynamically optimizing dictionary accesses might negate some performance gains initially observed in other areas of the code.

Another critical factor is the nature of the keys used.  If keys are of complex types requiring extensive hashing operations, the time spent in the hashing algorithm itself can dominate the overall dictionary access time, minimizing the benefits of the JIT compiler.  Finally, the size of the dictionary is crucial; small dictionaries will not show significant speed difference between CPython and PyPy3 because the overhead associated with JIT compilation outweighs benefits.


**2. Code Examples with Commentary:**

**Example 1:  Favorable Scenario for PyPy3**

This example showcases a scenario where PyPy3's JIT compiler excels.  The loop is predictable, and the key accesses are consistently within a pre-defined range.

```python
import time

data = {i: i * 2 for i in range(1000000)}

start_time = time.time()
for i in range(1000000):
    val = data[i]  # Predictable key access
    # Perform some operation with val
    pass
end_time = time.time()

print(f"PyPy3 Execution Time: {end_time - start_time} seconds")
```

In this scenario, the JIT compiler effectively optimizes the loop, leading to significant speedup compared to CPython. The consistent pattern of key access allows for efficient code generation.


**Example 2: Less Favorable Scenario for PyPy3**

This example demonstrates a situation where the less-predictable nature of key access hampers PyPy3's performance.

```python
import random
import time

data = {i: i * 2 for i in range(1000000)}
keys = list(data.keys())

start_time = time.time()
for _ in range(100000):
    random_key = random.choice(keys)
    val = data.get(random_key, 0) # Less predictable key access
    # Perform some operation with val
    pass
end_time = time.time()

print(f"PyPy3 Execution Time: {end_time - start_time} seconds")

```

Here, the random key selection makes it difficult for the JIT compiler to predict the memory accesses, resulting in less efficient code generation. The `get()` method, while safer, adds additional overhead.  This scenario might show a smaller performance advantage, or potentially even a slight disadvantage compared to CPython.


**Example 3:  Complex Key Types**

This example highlights the impact of complex key types on dictionary performance in both CPython and PyPy3.

```python
import time
import hashlib

data = {hashlib.sha256(str(i).encode()).hexdigest(): i * 2 for i in range(100000)}

start_time = time.time()
for i in range(100000):
    key = hashlib.sha256(str(i).encode()).hexdigest()
    val = data[key]
    pass
end_time = time.time()

print(f"PyPy3 Execution Time: {end_time - start_time} seconds")
```

The use of SHA256 hashes as keys introduces significant overhead in the hashing computation itself.  Both CPython and PyPy3 will experience considerable slowdown due to the computationally expensive key generation and comparison, minimizing the potential gains from PyPy3's JIT compilation.


**3. Resource Recommendations:**

For a deeper understanding of PyPy's internals and JIT compilation techniques, I would recommend consulting the official PyPy documentation and the research papers on tracing JIT compilers.  Furthermore, exploring the source code itself can be highly beneficial, though this requires substantial familiarity with C and compiler optimization techniques.  Finally, studying performance profiling tools and techniques can assist in identifying bottlenecks and optimizing specific code sections.  This includes understanding the difference between wall-clock time and CPU time. Mastering these resources is crucial for effectively leveraging PyPy's capabilities and mitigating potential performance issues.
