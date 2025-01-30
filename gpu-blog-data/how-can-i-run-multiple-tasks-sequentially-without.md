---
title: "How can I run multiple tasks sequentially without waiting for each to complete?"
date: "2025-01-30"
id: "how-can-i-run-multiple-tasks-sequentially-without"
---
My primary experience in distributed systems has revealed that managing task concurrency without inadvertently creating bottlenecks is a common challenge. Running tasks sequentially *without* waiting for each to complete, while seemingly contradictory, refers to asynchronous execution. Instead of the classic synchronous model where each task blocks the thread until completion, asynchronous programming allows tasks to initiate and then yield execution control, permitting subsequent tasks to start immediately. The key here is understanding that "sequential" in this context describes the *order of initiation*, not the order of completion. This requires careful orchestration using concurrency constructs.

Specifically, achieving this usually involves a mechanism that enables non-blocking execution. In many programming languages, this translates to using threads or coroutines/async functions. The fundamental difference is that threads operate at the OS level, managing pre-emptive concurrency, whereas coroutines are lightweight, user-level abstractions that usually run on a single thread, managed via a cooperative multitasking model. The choice often depends on the nature of the tasks and the operating environment. For I/O-bound operations, where tasks frequently wait for external resources like database responses or network calls, asynchronous coroutines often perform more efficiently due to their lower overhead. For CPU-bound tasks, threads or dedicated process pools might be a better fit. Regardless, the core principle is to avoid blocking the execution thread, allowing the initiation of the next task without having to wait for the current task to finish. The orchestration mechanism ensures that each task is eventually completed. This means we need something, at minimum, to signal task completion and potentially return its result, if needed, and trigger the subsequent task.

The underlying implementation typically employs event loops or callback queues. An event loop constantly monitors the system for events, such as a task's readiness, and then dispatches those ready tasks to execute. A task initiates, and if it requires blocking (e.g., waiting on an I/O operation), it signals that to the event loop. The event loop then moves on and can schedule other ready tasks until that task signals completion. The important distinction is that the primary thread doesn’t block; it continues to execute other work. Callbacks provide a mechanism for a task to signal its completion and, possibly, to supply result values for other dependent tasks. The next task can then be initiated through a completion callback. Therefore, by scheduling tasks to be triggered in a sequential order through such callbacks or using the event loop’s schedule mechanism, we can achieve the effect of sequential task initiation, even if their executions overlap. This approach is often referred to as "non-blocking" or "asynchronous programming."

Let's consider a few code examples using Python and its `asyncio` library. This library provides a framework to manage asynchronous operations based on coroutines.

**Example 1: Basic Asynchronous Task Sequencing**

```python
import asyncio
import time

async def task_one():
  print("Starting task_one")
  await asyncio.sleep(2) # Simulate some work
  print("Finished task_one")
  return "Result from task_one"

async def task_two(input_data):
  print(f"Starting task_two with input: {input_data}")
  await asyncio.sleep(1) # Simulate some work
  print("Finished task_two")
  return "Result from task_two"


async def main():
  result_one = await task_one() #Initiates and waits for the result from task one.
  result_two = await task_two(result_one) #Initiates task two after task one has completed.
  print(f"Final Results: {result_two}")


if __name__ == "__main__":
  start = time.time()
  asyncio.run(main())
  end = time.time()
  print(f"Execution Time: {end - start:.2f} seconds")
```

In this first example, we have two asynchronous functions (`task_one` and `task_two`). `task_one` executes first, and then `task_two` executes using the result of `task_one`. While `task_one` and `task_two` use `await asyncio.sleep()`, which simulates I/O-bound operations, and returns control to the event loop, the key aspect here is that the `main` function waits for the result of `task_one` before it initiates `task_two`. In this scenario, the tasks are still performed in a sequential manner but in a non-blocking way, which is the objective. The total execution time reflects the sum of sleep periods of both tasks due to the sequential wait. Note that the use of `async/await` syntax is instrumental in ensuring that a task yield control and doesn't block the event loop.

**Example 2: Concurrent Task Initiation With `asyncio.gather`**

```python
import asyncio
import time


async def task_three(task_id, delay):
  print(f"Starting task_{task_id}")
  await asyncio.sleep(delay)
  print(f"Finished task_{task_id}")
  return f"Result from task_{task_id}"


async def main_concurrent():
  tasks = [task_three(1, 2), task_three(2, 1), task_three(3, 0.5)]
  results = await asyncio.gather(*tasks)  # Initiates all tasks concurrently
  print(f"Results: {results}")


if __name__ == "__main__":
  start = time.time()
  asyncio.run(main_concurrent())
  end = time.time()
  print(f"Execution Time: {end - start:.2f} seconds")
```

Here, the goal is to initiate tasks sequentially, but not to wait for the previous task to complete. `asyncio.gather` takes a list of coroutine objects and schedules them concurrently. Despite initiating them concurrently, the key here is that *all* the tasks are initiated without waiting, and control is immediately returned to the event loop. This allows task execution to overlap. The `asyncio.gather` function, upon completion of *all* tasks, produces a list of results. Notice that the overall time for this execution will be driven by the longest task instead of the sum of all delays.

**Example 3: Utilizing Task Dependencies With `asyncio.create_task`**

```python
import asyncio
import time


async def task_four(task_id, dependency=None):
  if dependency:
    await dependency
  print(f"Starting task_{task_id}")
  await asyncio.sleep(1)
  print(f"Finished task_{task_id}")
  return f"Result from task_{task_id}"


async def main_dependent():
    task1 = asyncio.create_task(task_four(1))
    task2 = asyncio.create_task(task_four(2, task1))
    task3 = asyncio.create_task(task_four(3, task2))
    results = await asyncio.gather(task1, task2, task3)
    print(f"Results: {results}")


if __name__ == "__main__":
  start = time.time()
  asyncio.run(main_dependent())
  end = time.time()
  print(f"Execution Time: {end - start:.2f} seconds")
```

In this last example, we use `asyncio.create_task` to create asynchronous tasks but manage task dependencies via task objects instead of waiting. `task2` depends on `task1`, and `task3` depends on `task2`, forming a chain. This is accomplished by passing the task object of the dependent task. Though created sequentially, these tasks run asynchronously. However,  `task2` will wait for `task1` to finish before executing and so on, creating an implicit order of execution. The tasks are still initiated sequentially, but because we do not use `await` on each function call, the control is returned to the event loop, which allows the tasks to run concurrently.

In summary,  achieving sequential initiation of tasks without waiting for completion requires understanding asynchronous constructs, coroutines, and event loops. Techniques like callback queues, event loops, and explicit task management provide a way to schedule tasks sequentially without blocking the primary execution thread. This approach improves overall system efficiency, especially for I/O-bound operations.

For further understanding of concurrency and asynchronous programming, several resources are available. Books dedicated to concurrency and distributed systems often cover theoretical and practical aspects. Documentation from specific languages, such as Python’s `asyncio` documentation, is also valuable. Studying specific concurrency patterns, such as the reactor pattern, is also beneficial. Additionally, research papers from academia or industry can provide deeper insights into the complexities and best practices of concurrency management. These resources, used in combination, greatly assist in implementing scalable and robust asynchronous applications.
