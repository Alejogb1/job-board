---
title: "Are Python's async/await mechanisms based on yield?"
date: "2024-12-23"
id: "are-pythons-asyncawait-mechanisms-based-on-yield"
---

Let's tackle this one. It’s a common point of confusion, and I can see why. Over the years, particularly back when async programming was gaining wider adoption in the Python ecosystem, I recall a lot of misconceptions floating around, particularly concerning how `async`/`await` related to generators and the `yield` keyword. The short answer, and we’ll delve deeper shortly, is that while there are historical and conceptual connections, `async`/`await` are *not* built on top of generators in current versions of Python (3.5 onwards). They utilize an entirely different, more structured, and efficient mechanism.

Now, before getting into the core of the matter, let's briefly revisit the landscape *before* `async`/`await` became the standard. Generators, created with functions containing `yield`, could indeed be used to implement a form of cooperative multitasking. You’d use them to create stateful routines, effectively pausing and resuming execution to simulate concurrency within a single thread. There were frameworks that cleverly leveraged this feature for asynchronous I/O. The problem? The logic became quite complex and difficult to follow as projects grew. It was verbose, relying on manually managing the execution state of these generators.

The introduction of `async` and `await` in Python 3.5 was a game changer, directly addressing the shortcomings of generator-based concurrency. It introduced a new paradigm for writing asynchronous code, built around a dedicated `async` function definition and the `await` keyword. Instead of `yield` pausing a generator, `await` pauses the execution of an `async` function until an awaitable object (like a coroutine or a future) is ready. This mechanism is implemented using the interpreter's internal machinery and the concept of "coroutines" that are distinct from generators.

The key distinction is how these pauses occur and are handled. With generators, the program explicitly "yields" control; with `async`/`await`, control is given up implicitly by awaiting a "future" that the event loop can monitor. This is not mere semantics, the underlying architecture is quite different.

Let's look at a simple example to highlight the conceptual difference. First, let’s examine a generator-based approach to a hypothetical task that could involve some kind of simulated delay:

```python
import time

def delayed_task_gen(delay):
    print(f"Starting task with delay {delay}")
    yield
    time.sleep(delay)
    print(f"Task completed after {delay} seconds")

def run_tasks_gen():
    tasks = [delayed_task_gen(1), delayed_task_gen(0.5), delayed_task_gen(2)]
    for task in tasks:
        next(task)
    while True:
        completed_count = 0
        for task in tasks:
           try:
              next(task)
           except StopIteration:
                completed_count += 1
        if completed_count == len(tasks):
            break
        time.sleep(0.1)

run_tasks_gen()

```

In this example, `delayed_task_gen` uses `yield` to simulate a pause. The `run_tasks_gen` function steps through the generators and manually checks to see when they are done. It's a basic illustration but shows the kind of manual control needed. This is less readable than `async/await` and requires explicit iteration and exception handling.

Now, let's rewrite the same task using `async`/`await`:

```python
import asyncio

async def delayed_task_async(delay):
    print(f"Starting task with delay {delay}")
    await asyncio.sleep(delay)
    print(f"Task completed after {delay} seconds")


async def run_tasks_async():
    await asyncio.gather(delayed_task_async(1), delayed_task_async(0.5), delayed_task_async(2))

asyncio.run(run_tasks_async())
```
Here, `delayed_task_async` is defined using `async`, and we use `await` to wait on `asyncio.sleep()`. The `asyncio.gather()` function handles the concurrency implicitly through the event loop, which we launch using `asyncio.run()`. The overall code is clearer and more concise. The event loop manages the scheduling for us, rather than relying on manual steps.

To further highlight that async/await is not generator-based, consider this third illustrative example of how they interact with each other.

```python
import asyncio

async def async_gen():
    for i in range(3):
        print(f"yielding {i}")
        yield i
        await asyncio.sleep(0.1)

async def run_async_gen():
    async for x in async_gen():
        print(f"received {x}")


asyncio.run(run_async_gen())
```
Here we have a generator inside an async context, and note that for this to work an `async for` had to be used, which in turn makes the generator async. You can see that generators can interact with async mechanisms, but they are not the same mechanism.

In essence, Python's `async`/`await` leverages asynchronous coroutines, which are handled by an event loop that is implemented at a much lower level than simple generators. The core of the mechanism depends on underlying system calls (like `epoll` or `kqueue` in Linux and macOS, or `IOCP` in Windows) for efficient non-blocking I/O operations. When an `await` is encountered, the event loop is informed that the coroutine is waiting, and control is then passed to other eligible coroutines. Once the awaited operation completes, the event loop resumes the coroutine where it left off.

If you're looking for a deeper understanding, I strongly recommend reading the original PEP 492 that introduced async/await, as well as the "Concurrency with asyncio" section in the Python documentation itself, particularly the explanation of "Event Loops, Coroutines, and Tasks". For a solid theoretical background on event loops, I'd point you towards "Operating Systems Concepts" by Silberschatz, Galvin, and Gagne, which covers event-driven programming and its benefits in detail. Furthermore, for advanced practical insights, consider "Fluent Python" by Luciano Ramalho, particularly the chapters on concurrency and asyncio. These resources will offer a more formal and comprehensive understanding of the topic than what you will generally find on a blog or forum.

While generators played a historical role in paving the way for async programming in Python, the `async`/`await` constructs are a significant advancement in terms of both readability, performance, and their implementation mechanism. The underlying techniques are radically different from the generator-based approaches used previously. So, the critical takeaway is: don't confuse the historical lineage with the current implementation. They are separate beasts.
