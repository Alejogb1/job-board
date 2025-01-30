---
title: "How do I clear the Python line profiler's results?"
date: "2025-01-30"
id: "how-do-i-clear-the-python-line-profilers"
---
The Python line profiler, specifically the `line_profiler` package, accumulates results between profiling runs; these results are not automatically cleared unless explicitly instructed. This accumulation can lead to confusion when assessing performance changes across iterations. My experience, particularly with iterative numerical algorithms, highlighted the need for a method to reliably reset the profiler state. I've found that while there is no dedicated function within the `line_profiler` module for clearing results, the solution involves re-instantiating the `LineProfiler` object itself. The underlying implementation stores profile data within the object's attributes, meaning a new instance starts fresh.

The core issue arises from the nature of the profiler's design. After each call to `kernprof.py -l <script>.py`, or its equivalent programmatic usage via `profile()`, the accumulated execution time and hit count for each line are appended to the existing data within the `LineProfiler` object. This object is not directly exposed or controllable through any specific clearing method. Therefore, to effectively “clear” the results, a new `LineProfiler` object must be created. Subsequent profiling actions with this new object will provide isolated metrics. This approach leverages the lifecycle of the profiler instance itself rather than directly modifying existing data structures within it.

Here’s how to accomplish this “clearing” in practice, coupled with examples to illustrate the process:

**Example 1: Basic Profiling and Implicit Accumulation**

This initial example shows the typical problem; results persist across profiling runs. I'll profile a trivial function that increments a value within a loop, and run it twice. I'll use the decorator method within a single script.

```python
from line_profiler import LineProfiler

@profile
def increment_list(data, num_iterations):
    for _ in range(num_iterations):
        for i in range(len(data)):
          data[i] +=1
    return data

if __name__ == "__main__":
  data = [1, 2, 3, 4, 5]

  # First profiling run
  increment_list(data, 1000)
  
  # Second profiling run, without clearing
  increment_list(data, 500)
```

Executing this script, for example by `kernprof.py -l <script>.py`,  and viewing the output in the `.lprof` file will reveal that the second run’s metrics are appended to the first. Specifically, the time and hit counts will be larger during the second run. This occurs because the `profile` decorator is using a single `LineProfiler` instance across these calls. The problem becomes clearer with longer functions and complex data structures.  It’s difficult to understand the impact of different iterations when the results are stacked.

**Example 2: Explicit Clearing via Re-instantiation**

This example demonstrates the correct method of clearing profile data by creating a new `LineProfiler` instance. I'll modify the previous example to profile the increment function twice, but this time, I will use the `LineProfiler` object directly and re-instantiate it.

```python
from line_profiler import LineProfiler

def increment_list(data, num_iterations):
    for _ in range(num_iterations):
        for i in range(len(data)):
          data[i] +=1
    return data

if __name__ == "__main__":
  data = [1, 2, 3, 4, 5]

  # Instantiate profiler for first run
  profiler = LineProfiler()
  profiler.add_function(increment_list) # Explicitly adding function
  
  # First profiling run
  profiler.runctx("increment_list(data, 1000)", globals(), locals()) # Running with context
  profiler.print_stats()

  # Clear by re-instantiating the profiler
  profiler = LineProfiler()
  profiler.add_function(increment_list) # Adding function for second run

  # Second profiling run, profiler is cleared
  profiler.runctx("increment_list(data, 500)", globals(), locals())
  profiler.print_stats()
```

By reassigning `profiler` to a new `LineProfiler()` instance before the second run, the profile data associated with the first run is effectively discarded. The stats printed following each run reflect only the calls to the function under the newly instantiated profiler.  The first run will show results corresponding to 1000 iterations while the second run will report only 500, as expected. I typically use this pattern when optimizing code, especially to ensure that changes to function structures are correctly reflected in profiling metrics.

**Example 3: Function Wrapper for Profiler Re-instantiation**

For enhanced modularity and repeated use across different scripts, it’s advantageous to wrap the profiler instantiation and execution into a function. This hides the complexity of managing the profiler and promotes cleaner code.

```python
from line_profiler import LineProfiler

def increment_list(data, num_iterations):
    for _ in range(num_iterations):
        for i in range(len(data)):
          data[i] +=1
    return data

def run_profile(func, *args, **kwargs):
    """Profiles a function with line_profiler and clears previous results."""
    profiler = LineProfiler()
    profiler.add_function(func)
    profiler.runctx("func(*args, **kwargs)", globals(), locals())
    profiler.print_stats()

if __name__ == "__main__":
  data = [1, 2, 3, 4, 5]

  # First profiling run
  run_profile(increment_list, data, 1000)

  # Second profiling run, with clear via function
  run_profile(increment_list, data, 500)
```

This approach defines a reusable function that handles profiler instantiation, function registration, execution, and output. The `run_profile` function now facilitates easy profiling of any function and ensures that data from previous runs is cleared. This abstraction helps maintain code clarity when conducting repeated profiling runs. It allows consistent usage across different scripts or modules, enhancing code maintainability and reducing errors. I prefer this modular setup when working with large projects, because it simplifies management of the profiling process.

Regarding resource recommendations, while the `line_profiler` package does not offer extensive in-depth documentation beyond its main README and source code, there are several routes for gaining additional context. I would advise consulting Python performance books, specifically chapters on profiling. Also, searching through StackOverflow for past questions on code optimization and profiling can yield practical insights into using `line_profiler` within various contexts. Further, examining code examples in open-source projects that employ performance tuning tools would be useful to understand typical profiling workflows.  Lastly, the documentation for the `dis` module, regarding code disassembly, would also be useful for detailed analysis. Combining this material will build a complete picture of performance optimization in Python, with a particular emphasis on effective `line_profiler` use.
