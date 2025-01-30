---
title: "Can two independent tasks run in parallel without async/await?"
date: "2025-01-30"
id: "can-two-independent-tasks-run-in-parallel-without"
---
The fundamental constraint on true parallel execution within a single-threaded environment is the limitation of the execution engine itself. While seemingly counterintuitive, tasks can appear to run in parallel without explicit asynchronous programming, but this is achieved through mechanisms distinct from genuine, concurrent threading. This response will explore these mechanisms, specifically focusing on the operating system’s ability to manage multiple processes and, within a single process, the illusion of parallelism using event loops and callbacks. I’ll illustrate this with examples based on my experience building a data pipeline where efficient data processing without unnecessary context switching was critical.

The crucial distinction is between concurrency and parallelism. True parallelism requires multiple processing units capable of performing computations simultaneously. This is commonly achieved through multithreading or multiprocessing. Concurrency, on the other hand, involves managing multiple tasks that might not be executing at the exact same moment but appear to be happening in parallel.  Event loops, common in JavaScript and other single-threaded environments, are a core mechanism for achieving this concurrency.

While explicit `async/await` syntax often implies asynchronous behavior, it’s primarily a syntactic sugar for managing promises or other asynchronous operations within an event loop. It doesn’t inherently unlock true parallel execution. Instead, it allows the single thread to handle other tasks while waiting for an asynchronous operation, like a network request or a file I/O operation to complete. The underlying mechanism is that when a task reaches a point where it needs to wait, it registers a callback function and releases the thread back to the event loop. The loop then executes other pending tasks until the external wait is completed, and the callback is then enqueued for execution.

Now, how can tasks appear to run in parallel without this explicit `async/await`? The answer lies in leveraging different *processes* rather than threads within a single process. The operating system is capable of running multiple processes concurrently using scheduling algorithms to manage CPU time. Even on a single core processor, this rapid switching between processes can give the illusion of parallelism. Each process has its memory space and its own resources. This is how programs such as multiple web browsers or a word processor and a media player run concurrently. In this scenario, our “independent tasks” are implemented as separate processes rather than as distinct asynchronous operations in the same process.

My practical experience involved a data ingestion pipeline. I initially conceived this pipeline to involve several tasks: a database poll for new data, an API call to enrich data, and local storage. Trying to handle this in a single process with standard loop and blocking I/O resulted in poor performance. Leveraging the operating system’s ability to handle multiple processes was necessary.

Here’s the first example of a naive approach in Python, which does not achieve parallelism:

```python
import time

def task1():
    print("Task 1 started")
    time.sleep(2)
    print("Task 1 finished")

def task2():
    print("Task 2 started")
    time.sleep(1)
    print("Task 2 finished")

if __name__ == "__main__":
    task1()
    task2()
```
In this case, `task1` executes completely before `task2` starts. The execution is entirely sequential. `time.sleep()` is a blocking operation, causing the main thread to wait, and no other operation is performed during this sleep. There is no apparent or actual parallelism.

The next example demonstrates how to initiate these tasks as separate processes in Python using the `multiprocessing` module:

```python
import time
import multiprocessing

def task1():
    print("Task 1 started")
    time.sleep(2)
    print("Task 1 finished")

def task2():
    print("Task 2 started")
    time.sleep(1)
    print("Task 2 finished")

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=task1)
    p2 = multiprocessing.Process(target=task2)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
```

Here, the `multiprocessing.Process` creates separate operating system processes, each with its own Python interpreter. When `p1.start()` and `p2.start()` are called, the operating system scheduler begins executing the two processes concurrently. Although each individual process will execute its function sequentially, the two processes will run in parallel, and their outputs will appear interwoven in terms of when they are printed to the console. The program then waits for both processes to complete using `.join()`. This demonstrates actual parallel processing, not just concurrency. This approach increased the speed of my data pipeline immensely by splitting distinct processing steps into independent processes.

Finally, to clarify the limitations of only using event loops without multithreading, consider a Javascript example illustrating the single-threaded nature of concurrency. This example would need to be executed within a Javascript runtime like Node.js.

```javascript
function task1() {
  console.log("Task 1 started");
  let start = new Date().getTime();
  while (new Date().getTime() - start < 2000); // Simulate a blocking task
  console.log("Task 1 finished");
}

function task2() {
  console.log("Task 2 started");
  let start = new Date().getTime();
   while (new Date().getTime() - start < 1000); // Simulate a blocking task
  console.log("Task 2 finished");
}

console.log("Starting");
task1();
task2();
console.log("Finished");
```

Here, `task1` will complete its blocking operation (simulated here with a busy loop) before `task2` is called. Javascript operates single-threaded in this context; it does not create a new thread to run `task1` while also running `task2`. Neither does it execute `task1` and `task2` at the same time. Both tasks are synchronous, and therefore the second function `task2` will only be executed when the first function `task1` is complete. There is no benefit to non-blocking asynchronous operations. The JavaScript runtime can only execute one block of JavaScript code at a time using a single main thread. Even if these tasks were to make use of network operations using APIs like the `fetch` API, which are designed to be asynchronous, the event loop still only executes one operation or callback function at a time.

In summary, apparent parallelism can occur without explicit `async/await` through operating system-level process management.  `async/await` facilitates non-blocking behavior within a single thread. True parallelism requires multiple execution units, which are achieved through multiple processes or threads. Within a single-threaded environment like Javascript in a browser or Node.js, concurrency is achieved via an event loop and asynchronous operations, and true parallel execution within such a runtime is fundamentally impossible without multithreading techniques, or relying on the operating system level scheduler.

For further study, I would recommend books focusing on operating system concepts, specifically task scheduling and process management. Texts on concurrency and parallelism will provide a deeper theoretical understanding. To further understand how modern languages handle asynchronicity, resources covering event loops, callbacks and promises will be beneficial. Examining documentation for threading libraries and multiprocessing frameworks for specific languages will provide practical guidance, and research into different scheduler implementations used by operating systems is also recommended.
