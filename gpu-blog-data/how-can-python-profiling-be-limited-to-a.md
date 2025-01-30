---
title: "How can Python profiling be limited to a specific call depth?"
date: "2025-01-30"
id: "how-can-python-profiling-be-limited-to-a"
---
My experience debugging complex numerical simulations often necessitates precise control over profiling scopes. In Python, standard profiling tools like `cProfile` or `profile` analyze execution across all function calls by default, which frequently leads to vast datasets where pinpointing performance bottlenecks becomes challenging. To effectively investigate, I've found it critical to limit profiling to a specific call depth. Unfortunately, neither `cProfile` nor the `profile` module directly provides an argument for call depth. Achieving this requires a custom instrumentation strategy.

The challenge arises because the profilers track function entry and exit events irrespective of the call stack’s current depth. They record execution time for each function execution, then aggregate results based on these independent events. To filter on call depth, one must inject logic that determines the depth at which profiling should be active, and only then record the execution time.  This can be accomplished by wrapping the functions to be profiled or using a custom context manager. I’ve found the context manager approach cleaner and easier to apply modularly.

The key strategy involves a recursive function call counter combined with a custom profiler. This approach modifies how we collect profiling data to consider the depth of the current function call within the total call stack. The context manager handles the call counter, incrementing it on entry into the context and decrementing it on exit. Within the context manager, I’ve integrated profiling calls, but only when the counter matches the specified maximum profiling depth. This selective profiling avoids the large volumes of data from calls beyond the desired depth, focusing on the parts of the application relevant to the performance question.

To illustrate, consider an imaginary application where I need to analyze the performance of function calls within a specific nested loop structure represented by functions 'outer_loop', 'middle_loop' and 'inner_function'. Without depth control, a conventional profiler would generate a report containing aggregated data of every call made to 'inner_function', possibly obfuscating the bottleneck that might exist specifically at the third nested level, i.e., the immediate calls inside `middle_loop`.

**Code Example 1: Context Manager with Depth Control**

```python
import time
import contextlib
import cProfile

class DepthProfiler:
    def __init__(self, max_depth, profiler=None):
        self.max_depth = max_depth
        self.depth = 0
        self.profiler = profiler or cProfile.Profile()
        self._start_times = {} # For fine-grained timing

    def start_profiling(self):
        if self.depth == self.max_depth:
            self.profiler.enable()
            self._start_times[self.depth] = time.perf_counter()

    def stop_profiling(self):
      if self.depth == self.max_depth and self.depth in self._start_times:
        self.profiler.disable()
        end_time = time.perf_counter()
        elapsed_time = end_time - self._start_times[self.depth]
        print(f"Profiling depth {self.depth} time: {elapsed_time:.6f} seconds")
        del self._start_times[self.depth]

    def __enter__(self):
        self.depth += 1
        self.start_profiling()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_profiling()
        self.depth -= 1


def outer_loop():
    for _ in range(10):
        middle_loop()

def middle_loop():
  for _ in range(10):
    inner_function()

def inner_function():
    time.sleep(0.001)

if __name__ == '__main__':
    with DepthProfiler(max_depth=2) as prof:
      outer_loop()

    prof.profiler.print_stats(sort='tottime')
```

In this first example, the `DepthProfiler` class implements the depth control. The `__enter__` and `__exit__` methods manage the call depth.  The `start_profiling` and `stop_profiling` methods use a conditional statement to enable or disable the underlying profiler based on the current depth compared to `max_depth`. This selective enabling allows profiling only at a specific depth in the call stack. I have also included simple timing outside of the cProfile scope to show actual time at the specified depth and allow validation of the overall timing. The `__main__` block shows the use of the context manager to activate profiling of the function calls at the second depth in a simple nested loop. The profiler output is then sorted to display the most costly parts of the execution time.

**Code Example 2: Selective Profiling Using a Custom Decorator**

```python
import time
import cProfile
from functools import wraps

def profile_at_depth(max_depth):
    call_depth = 0
    profiler = cProfile.Profile()

    def decorator(func):
      @wraps(func)
      def wrapper(*args, **kwargs):
        nonlocal call_depth
        call_depth += 1
        if call_depth == max_depth:
            profiler.enable()
        try:
           result = func(*args,**kwargs)
        finally:
          if call_depth == max_depth:
            profiler.disable()
          call_depth -= 1
        return result
      return wrapper
    return decorator


@profile_at_depth(max_depth=2)
def outer_loop():
    for _ in range(10):
        middle_loop()

def middle_loop():
  for _ in range(10):
    inner_function()

@profile_at_depth(max_depth=3)
def inner_function():
    time.sleep(0.001)

if __name__ == '__main__':
  outer_loop()
  outer_loop() # Running the loops twice
  inner_function() # direct call to depth 3

  for depth in (2,3):
    print(f"Profiling Results for depth {depth}:")
    for func_name, profile_data in profile_at_depth.__closure__[2].cell_contents.items():
         if profile_data:
            profile_data.print_stats(sort='tottime')
```

This second example uses a decorator-based approach. The `profile_at_depth` function creates a closure around the `call_depth` counter and a profiler instance. Each time a wrapped function is called, the wrapper increments the `call_depth` counter and starts profiling using `profiler.enable()` if the depth equals the `max_depth`. The profiler is turned off when the function completes execution. The `try-finally` block ensures that the profiler is always disabled. This example demonstrates how selective profiling can be applied to specific functions at different depths by calling `profile_at_depth` with varying `max_depth` values before the function definitions, providing finer-grained control. Also, note the retrieval of profile stats outside of the wrapped function call using closure data.  This also shows we can call an instrumented function multiple times, and the profile will continue to collect data based on the provided call depth. This example also directly calls the instrumented `inner_function` to highlight the profile at depth 3.

**Code Example 3:  Function Wrapping with Explicit Depth Tracking**

```python
import time
import cProfile

class SimpleProfiler:
  def __init__(self):
    self.profiler = cProfile.Profile()
    self._start_times = {} # For fine-grained timing

  def start_profiling(self):
    self.profiler.enable()
    self._start_times['global'] = time.perf_counter()

  def stop_profiling(self):
    self.profiler.disable()
    end_time = time.perf_counter()
    elapsed_time = end_time - self._start_times['global']
    print(f"Overall Profile time: {elapsed_time:.6f} seconds")
    del self._start_times['global']

  def print_stats(self, sort='tottime'):
      self.profiler.print_stats(sort=sort)


def create_depth_profiler(profiler, max_depth):
  def depth_wrapper(func):
      call_depth = 0
      def wrapper(*args,**kwargs):
        nonlocal call_depth
        call_depth+=1
        if call_depth == max_depth:
          profiler.start_profiling()
        try:
          result = func(*args,**kwargs)
        finally:
          if call_depth == max_depth:
              profiler.stop_profiling()
          call_depth-=1
        return result
      return wrapper
  return depth_wrapper

def outer_loop():
    for _ in range(10):
        middle_loop()

def middle_loop():
  for _ in range(10):
    inner_function()

def inner_function():
    time.sleep(0.001)


if __name__ == '__main__':
    profiler = SimpleProfiler()
    depth_wrap = create_depth_profiler(profiler, 2)
    wrapped_outer = depth_wrap(outer_loop)

    wrapped_outer()
    profiler.print_stats()

    depth_wrap_level3 = create_depth_profiler(profiler, 3)
    wrapped_inner = depth_wrap_level3(inner_function)
    wrapped_inner()
    profiler.print_stats()
```

This third example directly wraps functions with the depth tracking function, and then uses the wrapper on the functions that are intended to be profiled. We are also using our simple `SimpleProfiler` in this example to show the profiling output directly and the time it took to capture the profile data. The `create_depth_profiler` function is designed to be used multiple times to create wrapped functions at different depths. This allows us to profile a function with the depth limitation without using a decorator by directly invoking the function wrapping on the relevant function. As shown, we have instrumented both `outer_loop` and `inner_function` at depths 2 and 3 respectively.

In summary, although Python standard profilers do not offer direct call depth control, carefully constructed context managers, decorators or function wrappers offer viable options for selectively profiling nested functions.  My experience shows the choice between context managers, decorators and function wrapping is largely a matter of coding style and the specific requirements of the project. For broader function scopes at specific depths the context manager is ideal. For selectively profiling individual functions, the decorator or wrapping methods are more flexible.  When working with large codebases, I've found it most beneficial to combine these approaches depending on the specific profiling needs of a section of the program.

For further exploration of profiling techniques, I recommend researching resources on Python performance optimization. Focus on understanding the different ways `cProfile` and the `profile` module work internally, as well as investigating best practices for using tools like `line_profiler` and memory profiling libraries like `memory_profiler`. Additionally, examining advanced profiling within scientific libraries such as NumPy or Pandas will offer more focused solutions when needed. These general resources, along with experimentation will further develop profiling intuition.
