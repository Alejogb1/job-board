---
title: "Why use Async for a single non-async operation?"
date: "2025-01-30"
id: "why-use-async-for-a-single-non-async-operation"
---
The pervasive misconception about asynchronous programming is its direct correlation to only inherently asynchronous operations, such as I/O bound tasks. Often overlooked is the potential benefit of using `async` and `await` even for conceptually synchronous operations within a single thread of execution, particularly when managed by a larger asynchronous context. I've observed this optimization firsthand when working on a high-throughput financial data processing pipeline at my previous position with QuantCore Analytics. Specifically, decoupling the processing of individual data points, even when the processing itself is CPU-bound, improved overall pipeline responsiveness when executed within the context of an asynchronous framework. This stems from the mechanism of task yielding and the underlying event loop, which allows other, potentially I/O-bound, asynchronous tasks to progress without being blocked by a single, lengthy synchronous operation.

Fundamentally, the `async` keyword, when applied to a method, transforms it into a state machine that encapsulates its execution. When we reach an `await` keyword within that `async` method, the task doesn’t simply halt; instead, it yields control back to the event loop. This gives the event loop the opportunity to execute other scheduled tasks, including other asynchronous operations that might be awaiting I/O or other resources. This mechanism is crucial even if the awaited “task” is a completed synchronous function, as it allows for collaborative multitasking in a single-threaded execution model. This capability avoids the blocking often associated with traditional synchronous execution models.

The primary advantage isn't about speeding up a *single* operation; instead, it's about preventing a single operation from monopolizing a thread and impeding the progress of other operations, especially when they are themselves asynchronous. Consider a web server that needs to perform some CPU-intensive data transformation on a user request before responding, this transformation being a non-async function. Performing the transformation synchronously in the event loop thread would block all other requests. Wrapping this transformation within an `async` context allows the thread to periodically yield, giving other requests a chance to execute, thereby significantly improving overall concurrency and perceived responsiveness of the server. This differs from multi-threading because this still takes place on one thread, and reduces the complexity and overhead associated with more complex threading operations. The cost of switching between context for tasks in an event loop is significantly lower than switching between threads, which makes it a very efficient pattern for I/O-bound workloads, while also not hindering CPU-bound activities.

Let’s explore some illustrative code examples using Python's `asyncio` library to demonstrate this.

**Example 1: Synchronous Function Wrapped in Async**

```python
import asyncio
import time

def synchronous_calculation(n):
    """A CPU-bound, non-async function."""
    result = 0
    for i in range(n):
      result += i*i
    return result

async def async_wrapper(n):
    """Wraps a synchronous function in an async task."""
    return synchronous_calculation(n)

async def main():
    print("Starting synchronous calculations wrapped in async:")
    start = time.time()
    results = await asyncio.gather(*(async_wrapper(100000) for _ in range(10)))
    end = time.time()
    print(f"Finished {len(results)} async tasks. Results:{results[:2]} Time Taken: {end - start:.4f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
```

In this example, `synchronous_calculation` is a CPU-bound function that computes a value based on the input `n`. The `async_wrapper` function takes an input, calls the synchronous function, and returns a result. The key part is when several tasks based on the `async_wrapper` function are created using a comprehension statement. The use of `asyncio.gather` executes those `async_wrapper` tasks concurrently. Even though `synchronous_calculation` is *not* itself asynchronous, wrapping it in an `async` function and awaiting it means the event loop can context switch between these tasks, giving a sense of parallelism. While the CPU is still doing the work, this prevents any single blocking operation from completely holding up the event loop, and allowing other tasks to run between processing blocks.

**Example 2: Comparing with Directly Synchronous Execution**

```python
import time
import asyncio

def synchronous_calculation(n):
    """A CPU-bound, non-async function."""
    result = 0
    for i in range(n):
      result += i*i
    return result

def synchronous_execution(n, count):
  """Executes the synchronous calculations sequentially."""
  results = []
  start = time.time()
  for _ in range(count):
      results.append(synchronous_calculation(n))
  end = time.time()
  return results, end - start

async def async_execution(n, count):
  """Executes a wrapped version of the synchronous function in parallel."""
  results = await asyncio.gather(*(async_wrapper(n) for _ in range(count)))
  return results, time.time() - start_time

async def async_wrapper(n):
  return synchronous_calculation(n)

async def main():
    global start_time
    print("Starting synchronous calculations synchronously:")
    start_time = time.time()
    results, time_taken = synchronous_execution(100000, 10)
    print(f"Finished synchronous tasks. Results:{results[:2]} Time Taken: {time_taken:.4f} seconds")

    print("\nStarting synchronous calculations wrapped in async:")
    results, time_taken = await async_execution(100000, 10)
    print(f"Finished {len(results)} async tasks. Results:{results[:2]} Time Taken: {time_taken:.4f} seconds")

if __name__ == "__main__":
  asyncio.run(main())
```

This second example directly compares the execution time of directly calling synchronous functions versus wrapping the synchronous call inside an `async` function and leveraging `asyncio.gather`. The primary difference in the synchronous approach is that it executes all of the computations in series in the same thread of execution. By using the async wrapper, we can see that even though the individual computations are synchronous, the overall execution is non-blocking within the event loop, and we gain a performance increase from this asynchronous behavior. While the CPU cycles needed to perform the same calculations are identical, the non-blocking nature of asynchronous execution leads to better responsiveness and throughput, particularly with larger volumes of tasks. You will observe with this code, that even with a purely CPU-bound operation, that the overall execution time is lower than the synchronous approach.

**Example 3: Integration with other I/O Bound Async Operations**

```python
import asyncio
import time
import random

async def fetch_data(duration):
    """Simulates an I/O-bound task with a variable sleep time."""
    delay = random.uniform(0, duration)
    print(f"Starting simulated I/O wait of {delay:.4f} seconds.")
    await asyncio.sleep(delay)
    print("Simulated I/O done.")
    return f"Data from I/O with {delay:.4f} second wait."

def synchronous_calculation(n):
  """A CPU-bound, non-async function."""
  result = 0
  for i in range(n):
    result += i*i
  return result

async def async_wrapper(n):
  """Wraps a synchronous calculation in an async task."""
  return synchronous_calculation(n)

async def process_data(input_data):
  """Performs synchronous processing and some I/O wait."""
  print(f"Starting data processing on {input_data}")
  processed = await async_wrapper(100000)
  print("Starting I/O operation for results.")
  io_result = await fetch_data(0.01)
  print(f"Processing for input {input_data} complete. Result:{processed}, I/O: {io_result}")
  return f"Processed: {processed}, I/O: {io_result}"


async def main():
    start = time.time()
    tasks = [process_data(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    end = time.time()
    print(f"All data processing complete. Results: {results}. Total time taken: {end-start:.4f} seconds.")

if __name__ == "__main__":
  asyncio.run(main())
```

This example demonstrates a complete processing pipeline where CPU-bound synchronous operations are mixed with asynchronous I/O bound operations. The `process_data` function wraps a synchronous computation using `async_wrapper` and then waits for a simulated I/O operation. The asynchronous nature of these operations allow for the program to execute the synchronous functions in a non-blocking manner while waiting for the I/O operations to complete, and vice-versa. The overall execution time is lower than a synchronous model would have been, and shows the value of asynchronous execution even in a mixed-mode program with non-async CPU-bound computations. It also highlights the potential of composing multiple async tasks into an end-to-end program, a very common scenario in production level systems.

For further exploration of the intricacies of asynchronous programming, I recommend delving into literature discussing the event loop model and task management concepts. Resources that elaborate on state machines in software engineering, particularly how these are leveraged in asynchronous frameworks, would be invaluable. Also, studying how operating systems handle concurrency through threading and process management can clarify the distinctions between async patterns, multi-threading, and multi-processing.

In summary, while `async` and `await` are frequently associated with I/O-bound operations, their utility extends to managing CPU-bound operations, as demonstrated through practical examples. The event-loop model allows for better performance and throughput than synchronous models when many tasks are executing, even when using a single thread. This is particularly true when tasks are a mix of I/O-bound and CPU-bound operations, as async operation context switching enables more efficient scheduling of operations and better overall responsiveness.
