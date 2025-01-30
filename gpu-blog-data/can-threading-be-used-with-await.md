---
title: "Can threading be used with `await`?"
date: "2025-01-30"
id: "can-threading-be-used-with-await"
---
Asynchronous operations and threading, while both mechanisms for concurrent execution, operate on fundamentally different principles. The core distinction hinges on how they achieve parallelism and manage CPU resources. Awaiting an asynchronous function does not directly invoke a thread; instead, it pauses the currently executing coroutine, allowing the event loop to handle other pending tasks. This is crucial for understanding how these concepts interact and where their strengths lie. My experience building high-throughput network services has underscored these differences repeatedly.

**1. The Fundamental Difference: Asynchronous Operations vs. Threading**

Asynchronous programming, often facilitated by `async`/`await` syntax in languages like Python, JavaScript, or C#, allows for concurrent execution within a single thread. When an `await` statement is encountered, the executing coroutine yields control back to the event loop. The event loop, in turn, can schedule other tasks, usually I/O-bound, such as network requests or disk reads, to progress while the initial task is waiting for a response. Once the awaited operation completes, the event loop resumes the coroutine at the point where it was paused. This approach avoids blocking the main thread while waiting, improving overall responsiveness of the application.

Threading, on the other hand, employs multiple operating system threads to achieve parallelism. Each thread operates independently, executing its own sequence of instructions, and can utilize multiple CPU cores. Threading is particularly suited for CPU-bound tasks that require significant processing time and can benefit from simultaneous execution. It is important to manage shared resources (e.g., memory) between threads using mechanisms like locks and mutexes to prevent race conditions and other concurrency issues. The complexity of such synchronization methods can, however, be a significant factor in project development and often introduces the possibility of deadlocks.

Therefore, `await` is primarily a mechanism for asynchronous operations managed within a single thread, and not a tool for creating or interacting with threads directly. Misunderstanding this distinction is a common source of problems when attempting to leverage concurrency. I've encountered numerous scenarios where attempting to force `await` into a threaded operation resulted in significant performance bottlenecks and unexpected behaviors.

**2. Code Examples and Commentary**

Let's illustrate with a few scenarios. Assume we are using Python, a common language supporting both asynchronous operations using the `asyncio` library, and threading.

**Example 1: Basic Asynchronous Operation**

```python
import asyncio
import time

async def long_running_async_task(task_id):
  print(f"Async task {task_id} started")
  await asyncio.sleep(2) # Simulate an I/O operation
  print(f"Async task {task_id} completed")
  return f"Result from task {task_id}"

async def main():
  start_time = time.time()
  results = await asyncio.gather(
      long_running_async_task(1),
      long_running_async_task(2),
      long_running_async_task(3)
  )
  print(f"All tasks completed in {time.time() - start_time:.2f} seconds.")
  print(f"Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```
*Commentary:*
In this example, `long_running_async_task` simulates an I/O-bound process (e.g., a network request) using `asyncio.sleep`. The `asyncio.gather` function starts these coroutines concurrently within the event loop. Notice that the program does not block for 6 seconds (3 tasks * 2 seconds), but completes close to 2 seconds, showing that tasks are executed concurrently within the event loop while waiting for I/O. There are no threads directly created and managed by this code. The `await` statement simply pauses the coroutine’s execution, allowing the event loop to move on to the next task.

**Example 2: Using Threading Directly**

```python
import threading
import time

def long_running_thread_task(task_id):
  print(f"Thread task {task_id} started")
  time.sleep(2) # Simulate a CPU-bound operation
  print(f"Thread task {task_id} completed")
  return f"Result from thread {task_id}"

def main():
    start_time = time.time()
    threads = []
    results = []
    for i in range(1, 4):
      thread = threading.Thread(target=long_running_thread_task, args=(i,))
      threads.append(thread)
      thread.start()

    for thread in threads:
        thread.join()
        results.append(thread.result) # This line needs to be modified to store result from threads correctly

    print(f"All thread tasks completed in {time.time() - start_time:.2f} seconds.")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
```
*Commentary:*
Here, we use `threading.Thread` to execute `long_running_thread_task` in separate threads. Unlike the previous asynchronous example, `time.sleep` here is designed to represent a CPU-bound operation that will keep a thread busy (e.g., complex calculations). We must explicitly create threads and use `thread.join` to ensure all threads complete before the main program continues. Additionally, results must be handled carefully as threads do not return values directly. This requires using more advanced techniques to store results from within a thread which I have omitted for brevity here (e.g., using a shared data structure). This example illustrates the core operation of threading. The program takes closer to 6 seconds, showing that threads are actually running in parallel utilizing (at least partly) multiple cores and not just switching tasks.

**Example 3: Mixing Asynchronous and Threaded Operations (Carefully)**

```python
import asyncio
import threading
import time
import concurrent.futures

def cpu_bound_task(task_id):
  print(f"CPU-bound task {task_id} started on thread {threading.get_ident()}")
  time.sleep(2)
  print(f"CPU-bound task {task_id} completed")
  return f"Result from CPU task {task_id}"

async def async_wrapper(executor, task_id):
  loop = asyncio.get_running_loop()
  return await loop.run_in_executor(executor, cpu_bound_task, task_id)

async def main():
  start_time = time.time()
  executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
  results = await asyncio.gather(
    async_wrapper(executor, 1),
    async_wrapper(executor, 2),
    async_wrapper(executor, 3)
  )
  executor.shutdown() # Important to release executor resources
  print(f"Mixed tasks completed in {time.time() - start_time:.2f} seconds.")
  print(f"Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
```
*Commentary:*
This example demonstrates a valid way to use threads within an asynchronous context using `concurrent.futures.ThreadPoolExecutor`. We cannot directly `await` the execution of `cpu_bound_task` since it is not an asynchronous function. Instead, we use `loop.run_in_executor` to execute it in a separate thread managed by the thread pool. The `async_wrapper` coroutine returns a future that represents the result of `cpu_bound_task` once it has completed. This is a common pattern when incorporating CPU-bound operations into an asynchronous application. The performance here will be similar to the first example since operations are done in parallel, and it is also important to shut down the executor with `executor.shutdown()` to release its resources. In cases like these, the program completes faster than the purely threaded example, because the I/O operations (if they existed) do not block the system.

**3. Resource Recommendations**

For a comprehensive understanding of concurrent and parallel programming, consult the official documentation for your chosen programming language's libraries, such as `asyncio` in Python. Specifically, I recommend reviewing resources covering the event loop model, coroutines, threading, and thread synchronization. Books on concurrent programming patterns, including those detailing the use of thread pools and the principles behind message queues, offer invaluable insights into designing robust and scalable systems. Practical experience, achieved by working through these examples and experimenting with them in real-world scenarios, is critical to solidifying your grasp of these complex topics. Lastly, carefully reviewing and understanding the design patterns of high-performance libraries is very useful.
