---
title: "Are `pathlib.Path`'s `unlink`, `mkdir`, and `rmdir` operations inherently asynchronous?"
date: "2025-01-26"
id: "are-pathlibpaths-unlink-mkdir-and-rmdir-operations-inherently-asynchronous"
---

`pathlib.Path`'s `unlink`, `mkdir`, and `rmdir` methods are inherently *synchronous*, operating within the same thread as the calling code. My experience building file management tools confirms this; these operations block the calling thread until the filesystem action completes. As a developer, I've frequently had to integrate these actions with asynchronous workflows and have learned firsthand the implications. Understanding their synchronous nature is critical for avoiding performance bottlenecks in applications requiring concurrent file system manipulations.

Let's dissect why these methods are synchronous and then consider how to handle them in asynchronous environments.

The `pathlib` module provides an object-oriented interface to interact with file paths. While convenient and highly readable, it ultimately relies on lower-level operating system calls for file system interactions. These operating system calls, such as the POSIX `unlink()`, `mkdir()`, and `rmdir()`, are inherently blocking operations. When a program invokes one of these system calls, the execution thread pauses until the system completes the file system operation, which typically involves accessing the storage device's hardware. Consequently, `pathlib`'s wrapper methods inherit this synchronous behavior.

Essentially, there's no asynchronous variant directly implemented within the `pathlib` library itself. The library designers prioritized straightforward, synchronous interaction with the underlying file system. This design choice simplifies basic file operations but necessitates explicit handling when asynchronous behaviors are required, like in network servers or UI applications that need to maintain responsiveness.

My projects frequently involve data ingestion processes where a large number of files must be moved, created, or deleted. During one project, attempting these operations directly with `pathlib.Path` in a concurrent environment caused significant slowdowns. The UI, which depended on other operations, would freeze until the file operations completed. This prompted me to explore appropriate asynchronous alternatives.

To further clarify the synchronous behavior, consider a simple code snippet demonstrating the direct use of these methods:

```python
import time
from pathlib import Path

def create_and_delete(path):
    start_time = time.time()
    path.mkdir(exist_ok=True)
    print(f"Created directory: {path} in {time.time() - start_time:.4f} seconds")
    time.sleep(1) # Simulate other operations
    start_time = time.time()
    path.rmdir()
    print(f"Removed directory: {path} in {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    test_path = Path("test_directory")
    create_and_delete(test_path)

```

Here, `create_and_delete` creates and removes the specified directory using `mkdir` and `rmdir`. Notice that the execution of the code is strictly sequential. The `mkdir` operation blocks the thread until it returns. Similarly, the following `rmdir` blocks until the removal completes. The `time.sleep(1)` simulates a delay while this single thread is busy and shows the impact of the synchronous execution. The print statements display the time taken, demonstrating that these operations are not instantaneous. This illustrates how blocking operations can impact performance.

For applications needing concurrency, using thread pools or asynchronous libraries can mitigate the blocking behavior of `pathlib`. Here’s an example using Python's `concurrent.futures.ThreadPoolExecutor` to execute the file operations in a separate thread:

```python
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def create_and_delete_async(path):
    start_time = time.time()
    path.mkdir(exist_ok=True)
    print(f"Created directory: {path} in {time.time() - start_time:.4f} seconds (Async Thread)")
    time.sleep(1) # Simulate other operations
    start_time = time.time()
    path.rmdir()
    print(f"Removed directory: {path} in {time.time() - start_time:.4f} seconds (Async Thread)")


if __name__ == "__main__":
    test_path = Path("test_directory_async")
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(create_and_delete_async, test_path)
        # simulate other work.
        time.sleep(0.5)
        print("Doing other work in main thread!")
    print("Main thread completed.")

```
This code utilizes a `ThreadPoolExecutor` to offload the file operations to a separate thread. Note the main thread continues without waiting for the executor to complete which is then displayed in the output. The `create_and_delete_async` method remains synchronous within that thread, but the concurrent execution keeps the main thread available to process other tasks. This demonstrates how to achieve concurrent file system interactions using threading.

Asynchronous programming with `asyncio` provides another powerful technique. However, since `pathlib`’s methods are not inherently asynchronous, it is necessary to use a wrapper to run the calls in a separate thread using `asyncio.to_thread`. Below, an implementation is shown using the `create_and_delete` function from the earlier example.

```python
import asyncio
import time
from pathlib import Path

async def async_create_and_delete(path):
    start_time = time.time()
    await asyncio.to_thread(path.mkdir, exist_ok=True)
    print(f"Created directory: {path} in {time.time() - start_time:.4f} seconds (Asyncio)")
    await asyncio.sleep(1) # Simulate other operations
    start_time = time.time()
    await asyncio.to_thread(path.rmdir)
    print(f"Removed directory: {path} in {time.time() - start_time:.4f} seconds (Asyncio)")

async def main():
    test_path = Path("test_directory_asyncio")
    await asyncio.gather(async_create_and_delete(test_path), asyncio.sleep(0.5), asyncio.sleep(0.1))
    print("Main Coroutine completed")

if __name__ == "__main__":
    asyncio.run(main())
```

In this final example, I leverage `asyncio.to_thread` to execute the `pathlib` operations within an asyncio event loop. This avoids blocking the main coroutine, effectively making the file operations behave as if they were asynchronous from the perspective of the main coroutine. The `asyncio.gather` function makes it clear that the execution of `async_create_and_delete` happens in parallel with the other asyncio tasks in the gather list. Using this approach, the main coroutine doesn't block and instead other coroutines are able to execute during the wait. The print statements clearly illustrate this asynchronous behavior.

In summary, while `pathlib.Path` is convenient, it does not inherently support asynchronous operations for methods such as `unlink`, `mkdir`, and `rmdir`. To use these methods effectively within asynchronous workflows, one should consider leveraging concurrency techniques, such as `ThreadPoolExecutor` for threading or `asyncio` combined with `asyncio.to_thread`. These techniques ensure that I/O-bound operations do not block program execution. For further information, the official Python documentation related to `pathlib`, `concurrent.futures`, and `asyncio` is extremely valuable. Also, books dedicated to concurrency and asynchronous programming in Python can provide further insight.
