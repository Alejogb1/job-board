---
title: "Does asyncio in Python only facilitate asynchronous operations between functions?"
date: "2024-12-23"
id: "does-asyncio-in-python-only-facilitate-asynchronous-operations-between-functions"
---

Alright, let's unpack this. The question about `asyncio` in Python and whether it solely manages asynchronous behavior *between* functions is a common misconception, and it's a good one to clarify. In my experience, particularly during that large-scale data pipeline project a few years back, I initially approached it with the same assumption. However, the reality is more nuanced. `asyncio` isn't just about coordinating asynchronous function calls; it’s about orchestrating asynchronous operations, where functions represent just one facet of that.

At its core, `asyncio` provides a framework for writing single-threaded concurrent code using coroutines. Think of it as a sophisticated event loop. The event loop monitors various asynchronous events, which can be anything from network sockets ready for reading or writing, to timers, to even operations on files, provided that those operations are designed to be non-blocking. When a coroutine hits a point where it needs to wait (for example, waiting on a network request to complete), it yields control back to the event loop. Crucially, while that coroutine is paused, the event loop can then switch to another coroutine that is ready to proceed. This mechanism is how `asyncio` achieves concurrency *without* threads.

Now, where the misconception typically arises is that people think the "waiting" only occurs when one `async` function calls another. That's not strictly true. The waiting can happen anywhere within an `async` function where control is passed back to the event loop. This is frequently accomplished through `await` statements, but these await statements don't necessarily need to be on other `async` functions. They can be on objects that encapsulate a potentially long-running operation, providing their interface supports an awaitable pattern (usually by returning an awaitable object).

To illustrate this, let’s first examine a simple case where the asynchronous behavior is achieved by awaiting the result of other async functions.

```python
import asyncio
import time

async def fetch_data(id):
    print(f"Fetching data for id: {id}")
    await asyncio.sleep(1) # Simulate I/O
    print(f"Data fetched for id: {id}")
    return f"Data_{id}"

async def process_data(id, data):
    print(f"Processing data: {data} for id: {id}")
    await asyncio.sleep(0.5) # Simulate processing time
    print(f"Data processed for id: {id}")
    return f"Processed_{data}"

async def main():
    tasks = [
        asyncio.create_task(fetch_data(1)),
        asyncio.create_task(fetch_data(2)),
        asyncio.create_task(fetch_data(3)),
    ]
    results = await asyncio.gather(*tasks)
    print("Fetched data:", results)

    process_tasks = [asyncio.create_task(process_data(i+1,result)) for i, result in enumerate(results)]
    processed_results = await asyncio.gather(*process_tasks)
    print("Processed data:", processed_results)


if __name__ == "__main__":
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()
    print(f"Total time: {end-start:0.4f} seconds")
```

In this first snippet, the `await` keywords are indeed used primarily to pause the execution of a coroutine while another asynchronous function completes. We use `asyncio.sleep` to mimic time-consuming io. We create tasks for `fetch_data` and `process_data` then gather their results. The point to note here is that the execution order is not strictly sequential. This illustrates a common use case where the asynchronous nature of operations between functions is used to allow for concurrency.

But, `asyncio` can manage asynchronous operations beyond just function calls. The power of `asyncio` expands when working with custom objects, including those representing external interactions, like network sockets and file streams (that can be operated on asynchronously). To demonstrate this, let’s create an example using an async-compatible custom object:

```python
import asyncio

class AsyncTimer:
    def __init__(self, duration):
        self.duration = duration
        self.is_done = False

    async def start(self):
        print(f"Timer starting for {self.duration} seconds.")
        await asyncio.sleep(self.duration)
        self.is_done = True
        print(f"Timer of {self.duration} seconds is done.")
        return self

    def __await__(self):
        return self.start().__await__()

async def main():
    timer1 = AsyncTimer(3)
    timer2 = AsyncTimer(1)
    timer3 = AsyncTimer(2)

    await asyncio.gather(timer1, timer2, timer3)
    print("All timers completed.")

if __name__ == "__main__":
    asyncio.run(main())
```

In this snippet, the `AsyncTimer` class is an object that can be used with `await`. Specifically, it has the `__await__` method and that is where the async operation is done, it doesn’t necessarily await on another function. In this case we use `asyncio.sleep`, but the idea is that you can control the asynchronous behaviour through custom classes. The important factor is that the object returns an awaitable, which, in turn, allows the event loop to switch contexts efficiently.

Finally, to illustrate this further and push the bounds, let’s show an example with an object that simulates a non-blocking IO operation by using an internal queue:

```python
import asyncio
from collections import deque

class AsyncIOQueue:
    def __init__(self):
        self._queue = deque()
        self._waiting_tasks = deque()

    def put(self, item):
        self._queue.append(item)
        if self._waiting_tasks:
            task = self._waiting_tasks.popleft()
            task.set_result(item)

    async def get(self):
        if self._queue:
            return self._queue.popleft()
        else:
            future = asyncio.get_running_loop().create_future()
            self._waiting_tasks.append(future)
            return await future


async def producer(queue):
    for i in range(5):
        print(f"Producing item {i}")
        await asyncio.sleep(0.5)
        queue.put(f"item-{i}")
    print("Producer finished")

async def consumer(queue, id):
    while True:
        item = await queue.get()
        print(f"Consumer {id}: Received item {item}")
        if item == "item-4":
           break
    print(f"Consumer {id} finished")

async def main():
    queue = AsyncIOQueue()
    await asyncio.gather(
        producer(queue),
        consumer(queue,1),
        consumer(queue,2)
    )


if __name__ == "__main__":
    asyncio.run(main())
```

Here, `AsyncIOQueue` allows us to simulate a queue that doesn't block the event loop. When a consumer tries to get from an empty queue, the consumer is paused and then resumes only after the producer has put an item into the queue. Again, this showcases an asynchronous operation, yet the `await` call isn't directly waiting on a function.

In summary, `asyncio` is not limited to asynchronous function calls. It is a framework for concurrency based around an event loop that handles arbitrary awaitable objects, allowing you to manage asynchronous interactions between *operations*—functions being a type of operation. This flexibility is crucial when dealing with complex I/O-bound applications or any scenario that benefits from non-blocking operations.

For deeper exploration, I'd highly recommend the official Python documentation for `asyncio`, specifically focusing on coroutines and awaitables, as that is the core of the system. In addition, understanding the theory behind event loops is valuable; for this, “Operating Systems Concepts” by Silberschatz, Galvin, and Gagne offers a robust treatment of the subject. For a more practical, application-focused view, "Fluent Python" by Luciano Ramalho has a good explanation of `asyncio` as well as the underlying principles that make it work.
