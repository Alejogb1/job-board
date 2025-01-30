---
title: "Why is my memory usage not decreasing after calling `gc.collect()`?"
date: "2025-01-30"
id: "why-is-my-memory-usage-not-decreasing-after"
---
Garbage collection in Python, specifically invoking `gc.collect()`, does not guarantee an immediate and substantial reduction in reported memory usage, particularly as observed by tools like `psutil` or system monitors. The primary reason for this lies in the complex interaction of Python's memory management, operating system memory allocation, and the nuances of garbage collection itself. I've seen this manifest frequently in long-running services I've managed, leading to a misconception that `gc.collect()` is a panacea for memory issues.

Python's memory management is a layered process. It starts with the CPython interpreter, which requests memory from the operating system (OS). This allocation is typically done in large chunks, not on a per-object basis. When Python objects are no longer referenced and become eligible for garbage collection, the garbage collector identifies these objects and marks their allocated memory for reuse. Crucially, this recycled memory is often retained within Python’s own heap, not immediately returned to the OS. This is done for performance: re-requesting memory from the OS is more expensive than re-using it within Python. Consequently, tools monitoring system-level memory usage might not register a decrease because the OS is still holding the previously allocated memory, even if Python’s internal structures have been cleaned.

Furthermore, `gc.collect()` primarily targets objects involved in reference cycles, situations where two or more objects refer to each other, preventing their reference count from reaching zero, even when no external references exist. The standard garbage collector is optimized to handle this cyclical dependency issue efficiently, not necessarily to aggressively shrink the Python heap. This is a critical distinction. If the memory pressure originates from a simple proliferation of non-cyclical objects, which are deallocated through reference counting rather than garbage collection, then `gc.collect()` will have a minimal impact. It addresses only a subset of memory-related concerns. Memory allocated through native C extensions, which do not use Python's memory manager, will not be released by the Python garbage collector; this also explains an apparent lack of change after a forced garbage collection cycle.

The timing of garbage collection is also a significant factor. Python's garbage collector operates periodically; the frequency is determined by internal heuristics, which can be influenced by the rate of allocations. Calling `gc.collect()` forces an immediate collection, but if memory usage primarily stems from allocated but currently in-use objects, or from memory that is managed elsewhere, then running `gc.collect()` will not result in any meaningful reduction in memory usage displayed by an external monitoring application. The `gc` module provides mechanisms to adjust thresholds and trigger garbage collection under specific conditions; however, it doesn't circumvent the underlying mechanics that can result in memory held by the OS even after deallocation.

Let’s examine a few concrete scenarios. Consider this code:

```python
import gc
import psutil
import time

def memory_usage_mb():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    return f"{mem:.2f} MB"

def create_large_list(size):
  large_list = list(range(size))
  return large_list

print(f"Memory before allocation: {memory_usage_mb()}")
large_data = create_large_list(1000000)
print(f"Memory after allocation: {memory_usage_mb()}")
large_data = None #remove the reference
print(f"Memory after dereferencing: {memory_usage_mb()}")
gc.collect()
print(f"Memory after gc.collect(): {memory_usage_mb()}")
```

Here, I'm creating a large list, then immediately dereferencing it. The output shows that even after `gc.collect()`, the apparent memory usage as reported by `psutil` doesn’t substantially decrease. The list is deallocated internally, but the OS-level memory is likely retained by the Python process for reuse. Python's memory allocation is not intended to be volatile, reallocating and releasing memory on each change. Instead, it aims for a steady state, and a forced garbage collection on an already deallocated list has minimal visible impact.

Now let's see an example of cyclical references:

```python
import gc
import psutil

def memory_usage_mb():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    return f"{mem:.2f} MB"

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

print(f"Memory before allocation: {memory_usage_mb()}")
node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1
print(f"Memory after allocation (with cycle): {memory_usage_mb()}")
node1 = None
node2 = None
print(f"Memory after dereferencing: {memory_usage_mb()}")
gc.collect()
print(f"Memory after gc.collect(): {memory_usage_mb()}")
```

In this case, we create a reference cycle between two `Node` objects. Dereferencing `node1` and `node2` does not immediately deallocate them because the reference cycle prevents their reference counts from reaching zero. `gc.collect()` is crucial here to break the cycle and allow the memory occupied by these nodes to be reclaimed by Python; but again, it may not be released from the OS.

Finally, an illustration of how non-Python memory can affect the observations:

```python
import gc
import psutil
import numpy as np

def memory_usage_mb():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    return f"{mem:.2f} MB"

print(f"Memory before allocation: {memory_usage_mb()}")
large_array = np.random.rand(1000000) # Numpy uses its own memory manager
print(f"Memory after Numpy allocation: {memory_usage_mb()}")
large_array = None
print(f"Memory after dereferencing: {memory_usage_mb()}")
gc.collect()
print(f"Memory after gc.collect(): {memory_usage_mb()}")
```
Here, the `numpy` library uses its own memory management, allocating a large array outside of Python's managed heap. Dereferencing the array will release the Python-level wrapper, but the underlying memory is not freed by the Python garbage collector. The apparent memory usage change after `gc.collect()` is negligible, as numpy's memory manager is separate. This shows that memory held outside of Python is not directly impacted.

The key takeaway is that `gc.collect()` is not a universally applicable solution for reducing apparent memory usage. It's designed to address reference cycles and internal Python memory management, not to directly manipulate the memory Python has claimed from the OS. Relying on it for immediate memory reduction when memory is held by the Python process (but not allocated by Python) or by another library is misguided. When addressing memory concerns, the focus should be on understanding the source of allocations, avoiding leaks or unintended object references, and, where necessary, using memory-mapped files or other low-level techniques when standard Python memory management is insufficient. Analyzing the memory footprint using tools like `tracemalloc` or memory profilers can offer insights into which parts of the program are holding memory, providing a more targeted approach. Consulting documentation on Python garbage collection is valuable to deepen one's understanding of the mechanics involved, and using memory profiling tools will aid in finding the precise problem causing the high memory footprint. For an in-depth analysis of system-level memory use, consider reviewing the documentation provided by your operating system.
