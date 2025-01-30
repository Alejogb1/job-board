---
title: "How can I achieve parallelism with asyncio for a non-async external library function?"
date: "2025-01-30"
id: "how-can-i-achieve-parallelism-with-asyncio-for"
---
Python's `asyncio` excels at concurrency, but its cooperative multitasking nature demands that functions explicitly yield control to the event loop using `await`. This poses a challenge when integrating with synchronous, blocking external libraries. The core issue lies in the fact that synchronous functions monopolize the thread, preventing the `asyncio` event loop from progressing until they complete. Effectively achieving parallelism requires bridging this synchronous-asynchronous gap.

The primary approach involves offloading the synchronous function to a separate executor, typically a thread pool, and then `await`ing the result of this operation within the asynchronous context. I’ve encountered this scenario frequently, especially while building network applications reliant on legacy libraries not designed for cooperative concurrency.

The following sections provide a detailed explanation of the methodology, accompanied by practical examples.

**Explanation**

The solution revolves around the `asyncio.to_thread()` function (or the lower-level `loop.run_in_executor()`), introduced in Python 3.9. These functions provide a mechanism to delegate a blocking function call to a separate thread within a thread pool. The event loop remains responsive because the call is not directly executed in the main thread. The asynchronous code pauses execution at the `await` point until the worker thread completes its task and returns a result. This is paramount, as without `await` the synchronous function may execute, but the results are not propagated back into the asynchronous context, leading to unexpected behavior.

Under the hood, these functions create a `concurrent.futures.Future` object. This future represents the eventual result of the asynchronous operation. The event loop monitors this future; when the operation completes within the worker thread, the loop wakes up the waiting coroutine and continues its execution. This is what enables non-blocking, seemingly parallel operations.

The critical concept to grasp here is that the parallelism is realized via multi-threading, not true parallel execution across processor cores (which is limited by Python's Global Interpreter Lock, or GIL). We are concurrently running multiple asynchronous tasks via `asyncio` and using threads to handle synchronous operations off the main thread, which ensures responsive user interfaces.

**Code Examples**

Let’s examine some practical examples to solidify the idea. Assume we have a synchronous function from an external library, `external_library_call()`, which performs some computationally intensive process, blocking execution until complete. For demonstration, it will simulate work with `time.sleep()`.

**Example 1: Basic Offloading**

```python
import asyncio
import time

def external_library_call(duration):
    print(f"Starting external function call for {duration} seconds")
    time.sleep(duration)
    print(f"Finished external function call for {duration} seconds")
    return f"Processed in {duration} seconds"

async def async_wrapper(duration):
    print(f"Starting async wrapper for {duration} seconds")
    result = await asyncio.to_thread(external_library_call, duration)
    print(f"Finished async wrapper for {duration} seconds with result {result}")
    return result

async def main():
    results = await asyncio.gather(
        async_wrapper(2),
        async_wrapper(3),
        async_wrapper(1)
    )
    print(f"Final results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```
*Explanation:*
This example demonstrates the core mechanism. The `external_library_call()` simulates a blocking external function. The `async_wrapper()` function uses `asyncio.to_thread()` to execute `external_library_call()` in a separate thread. The `main()` function uses `asyncio.gather()` to concurrently run multiple instances of `async_wrapper()`. We’ll see from the output that the blocking calls run concurrently (though not truly in parallel), and the `asyncio` event loop proceeds even while these synchronous operations are pending. The `await` keyword ensures the main event loop waits for completion of all calls.

**Example 2: Data Passing and Error Handling**

```python
import asyncio
import time

def external_library_call_with_data(data, duration):
   print(f"Starting external call with data: {data} for {duration} seconds")
   time.sleep(duration)
   if data == "error":
        raise ValueError("Simulated error in external function")
   print(f"Finished external call with data: {data}")
   return f"Processed {data}"

async def async_wrapper_with_error(data, duration):
   print(f"Starting async wrapper with data: {data}")
   try:
       result = await asyncio.to_thread(external_library_call_with_data, data, duration)
       print(f"Finished async wrapper with data: {data} and result {result}")
       return result
   except ValueError as e:
       print(f"Error in async wrapper with data: {data}: {e}")
       return None

async def main_with_error():
    results = await asyncio.gather(
         async_wrapper_with_error("data1", 1),
         async_wrapper_with_error("error", 2),
         async_wrapper_with_error("data2", 3),
         return_exceptions=True #Allows graceful handling of exceptions in gather()
    )
    print(f"Final results: {results}")


if __name__ == "__main__":
    asyncio.run(main_with_error())
```

*Explanation:*
This example extends the previous one by demonstrating how to pass data to the synchronous function and how to handle exceptions that might occur within it. We pass data to the external function via `asyncio.to_thread`, and the return value of the external library call is passed back as the result of `await asyncio.to_thread(...)`. Error handling is done using a regular try/except block around the `await` call. This example also demonstrates the use of `return_exceptions=True` argument to `asyncio.gather()`, which prevents `gather` from raising an exception when an underlying task errors out and instead places the exception object into the result. This allows for more graceful error handling of concurrent async calls.

**Example 3:  Utilizing Thread Pools Directly**
```python
import asyncio
import time
import concurrent.futures

def external_library_call_pool(duration):
    print(f"Starting external function call via pool for {duration} seconds")
    time.sleep(duration)
    print(f"Finished external function call via pool for {duration} seconds")
    return f"Processed via pool in {duration} seconds"

async def async_wrapper_pool(duration, loop, executor):
    print(f"Starting async wrapper via pool for {duration} seconds")
    result = await loop.run_in_executor(executor, external_library_call_pool, duration)
    print(f"Finished async wrapper via pool for {duration} seconds with result {result}")
    return result

async def main_with_pool():
   loop = asyncio.get_running_loop()
   with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = await asyncio.gather(
            async_wrapper_pool(2, loop, executor),
            async_wrapper_pool(3, loop, executor),
            async_wrapper_pool(1, loop, executor)
         )
        print(f"Final results from pool: {results}")

if __name__ == "__main__":
    asyncio.run(main_with_pool())

```
*Explanation:*
This example illustrates how you could explicitly manage a thread pool using `concurrent.futures.ThreadPoolExecutor` and pass it to `loop.run_in_executor`. While `asyncio.to_thread()` is often sufficient, managing the executor directly allows you to configure pool parameters (like the `max_workers` parameter) for fine-tuning resource utilization if you need greater control over threading resources. The loop is obtained via `asyncio.get_running_loop()`.  The `with` statement ensures proper resource management.

**Resource Recommendations**

For deeper understanding, consult the official Python documentation for `asyncio`, paying close attention to `asyncio.to_thread()`, `loop.run_in_executor()`, and `concurrent.futures`. Additionally, resources detailing thread pool usage within `concurrent.futures` are beneficial. Exploring articles or presentations on the Global Interpreter Lock (GIL) and its influence on multithreading in Python is also advisable. Finally, studying examples of real-world asynchronous applications, especially those involving interactions with external synchronous systems, will provide practical context. I would look for resources centered around event-driven architectures to help understand design patterns when using asyncio.
