---
title: "How can Python memory usage be profiled?"
date: "2025-01-30"
id: "how-can-python-memory-usage-be-profiled"
---
Python’s dynamic nature, while offering great flexibility, can present challenges when it comes to understanding and optimizing memory consumption. Unlike languages with explicit memory management, Python’s garbage collector (GC) automatically handles object deallocation, which can sometimes obscure the root causes of unexpected memory growth. Efficient profiling requires tools and techniques that delve deeper than simply observing overall system memory usage. Over my years working on large-scale data processing pipelines in Python, I’ve encountered various scenarios demanding precise memory analysis.

The first step in effective memory profiling involves understanding that Python memory usage can be broadly divided into two categories: memory used by Python objects themselves and memory used by native extensions and data structures residing outside the Python heap. Profiling should ideally target both aspects. For Python objects, the focus is on identifying large objects or collections that accumulate unnecessarily and objects held in memory longer than expected due to reference cycles or persistent data structures. Native allocations, often associated with libraries like NumPy or pandas, require different tools and often a more nuanced approach.

A straightforward method to gain initial insights is leveraging the built-in `sys` module. The `sys.getsizeof()` function provides the size of a specific object in bytes, including the object's header but not the size of referenced objects. While this offers a quick look at individual object sizes, it doesn't capture the collective memory footprint of a complex data structure or identify which parts of a program are contributing the most to the overall memory usage.

For more comprehensive memory profiling within the Python heap, the `memory_profiler` library is invaluable. It provides line-by-line memory usage analysis, pinpointing the exact source code locations where memory allocations occur. The `@profile` decorator is applied to functions we are interested in profiling, enabling the library to track the memory delta at each line's execution.

Consider the following example demonstrating a function designed to process textual data, which can often be a significant source of memory usage:

```python
from memory_profiler import profile
import random

@profile
def process_text_data(num_lines):
    all_lines = []
    for _ in range(num_lines):
        line = ''.join(random.choices('abcdefg', k=1000)) # Generate a string of 1000 characters
        all_lines.append(line)
    processed_lines = [line.upper() for line in all_lines] # Create new strings in all_lines
    return processed_lines

if __name__ == '__main__':
    process_text_data(5000)
```

When we run this script with the `mprof run <script_name.py>` command, `memory_profiler` generates a line-by-line memory profile which is stored in a separate file. We see that, as expected, appending the large strings to `all_lines` contributes significantly to memory usage, followed by creation of new strings in the `processed_lines` list. This reveals that creating two lists to hold large strings might not be the most memory-efficient approach.

Another strategy for memory profiling involves the `tracemalloc` module. Introduced in Python 3.4, it provides a way to track memory allocations made by the CPython interpreter at a detailed level, allowing you to examine where memory is allocated and how much. This is particularly useful for detecting memory leaks. Unlike `memory_profiler`, it profiles memory allocation directly at the interpreter level, offering a more fundamental understanding.

Here is an example where `tracemalloc` helps pinpoint a potential memory leak, which can occur with certain third-party library usage if references are not carefully managed:

```python
import tracemalloc
import gc
import time
import weakref
import random

class TestObject:
    def __init__(self, data):
        self.data = data
    
    def __del__(self):
        pass # Prevent immediate deallocation

def create_objects():
  objects = []
  for _ in range(10000):
    obj = TestObject(''.join(random.choices('abcdefg', k=100)))
    objects.append(obj)
  
  return objects

def do_work():
    trace = tracemalloc.start()
    objects_to_keep = create_objects()
    time.sleep(0.1)
    del objects_to_keep # Explicitly delete the list
    gc.collect()  # Force a garbage collection pass
    time.sleep(0.1)
    snapshot = trace.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    tracemalloc.stop()
    print("[ Top 10 allocations ]")
    for stat in top_stats[:10]:
        print(stat)

if __name__ == "__main__":
  do_work()

```

This example deliberately creates objects, keeps a reference in `objects_to_keep`, then deletes the reference and runs garbage collection. Despite explicit deletion, the objects may not be immediately deallocated due to potential internal library caching or object interdependencies. The use of `tracemalloc`, particularly with the `snapshot.statistics('lineno')`, allows identification of the lines in `create_objects` where significant memory is being allocated, which can help debug memory-related issues and indicate where memory is not being released as expected.

Finally, it’s often valuable to track overall memory usage at a system level using libraries like `psutil`. These provide insights beyond Python's internal memory landscape, revealing the impact of external processes or native libraries on total RAM consumption. While not granular enough to dissect individual Python object allocation, this approach can identify when the overall Python process is consuming unexpectedly high memory.

Consider a scenario where a background process spawned by the Python script starts consuming significant memory:

```python
import psutil
import os
import time
import subprocess

def memory_monitor():
    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss
    
    print(f"Initial Memory Usage: {initial_mem / (1024 * 1024):.2f} MB")
    
    p = subprocess.Popen(['python', '-c', 'import time; time.sleep(5); print("Process Finished")']) # Simulate external memory usage
    
    time.sleep(1)
    current_mem = process.memory_info().rss
    print(f"Memory Usage After External Process Start: {current_mem / (1024 * 1024):.2f} MB")
    
    p.wait() # Wait for the process to end
    
    current_mem = process.memory_info().rss
    print(f"Memory Usage After External Process End: {current_mem / (1024 * 1024):.2f} MB")
    
if __name__ == "__main__":
    memory_monitor()
```

This script first measures initial memory usage of the script process, then launches a second subprocess and reports the memory usage again. After the subprocess ends, it again reports memory usage. This reveals the impact of the external process on the parent process's memory usage and helps identify resource consumption beyond Python-allocated objects.

For further study, I suggest researching the official Python documentation for `sys`, `tracemalloc`, and how memory management is implemented. For practical application, delving into the `memory_profiler` and `psutil` project pages would be beneficial. Additionally, resources discussing the intricacies of Python’s garbage collection mechanisms and how it can be affected by circular references will prove useful. Furthermore, examining examples from libraries like NumPy and pandas that demonstrate their memory-efficient usage patterns could enhance an understanding of memory optimization techniques.
