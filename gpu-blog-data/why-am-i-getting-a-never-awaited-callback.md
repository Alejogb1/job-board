---
title: "Why am I getting a 'Never awaited callback' error when fetching websocket price data from a crypto exchange using async/await and websockets?"
date: "2025-01-30"
id: "why-am-i-getting-a-never-awaited-callback"
---
The "Never awaited callback" error when working with asynchronous operations and WebSockets, particularly in the context of fetching real-time cryptocurrency price data, often arises from a misunderstanding of how asynchronous generators, event-driven architectures, and the Python `asyncio` event loop interact. It indicates that an asynchronous task, typically a callback function associated with a WebSocket event, was scheduled but never properly awaited, preventing the event loop from effectively managing and completing the operation. In essence, the code initiated a task, then moved on without ensuring the task's eventual execution and cleanup, causing the underlying promise or coroutine to be left unresolved.

My experience building a low-latency market data aggregator confirms that these errors are frequently associated with the core architecture of asynchronous libraries such as `websockets` and `asyncio`. These libraries utilize an event-driven pattern. When a message arrives over the WebSocket connection, a user-defined callback is triggered. However, unlike synchronous programming, where each function runs to completion in a predictable sequence, asynchronous tasks need explicit instruction to pause, yield control back to the event loop, and resume when the necessary data or signal is available. For the error to occur, a failure within your callback likely means this pausing and yielding process hasn’t been setup correctly.

The crux of the issue lies in the asynchronous nature of the WebSocket’s message handling. Most WebSocket libraries, including the commonly used `websockets` package, deliver incoming messages through callbacks associated with event listeners. When using async/await, it becomes necessary to explicitly make these callbacks awaitable tasks, otherwise the event loop may not register them properly. Specifically, if a callback contains asynchronous operation, such as processing the JSON response or updating some internal state, and that operation is not itself awaited, the event loop is not aware that it should continue working on that specific task. As the code moves on to a subsequent process, the callback is left incomplete, resulting in the "Never awaited callback" error being thrown, possibly upon garbage collection of the underlying asynchronous task.

To clarify this common pitfall, let’s explore some concrete examples.

**Example 1: The Incorrect Approach**

The first example demonstrates a basic but erroneous approach. It establishes a WebSocket connection, registers a callback to handle incoming messages, but fails to properly await it.

```python
import asyncio
import websockets
import json

async def process_message(message):
    """Incorrectly processes an incoming message."""
    data = json.loads(message)
    print(f"Received price: {data['price']}")
    #No await is used here

async def subscribe_to_prices():
    uri = "wss://example.com/ws/prices"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            process_message(message) # Incorrectly, NOT awaited

async def main():
    await subscribe_to_prices()

if __name__ == "__main__":
    asyncio.run(main())
```

In this flawed example, the `process_message` function is called within the `async for` loop, but its asynchronous nature is effectively ignored. Because the `process_message` is not awaited, the event loop does not treat it as an unfinished task and thus it moves on. Subsequently, the connection may be closed without the message having been fully processed and handled, thereby triggering the “never awaited” error.

**Example 2: Correct Approach Using `asyncio.create_task`**

The following code provides the most common, reliable, and robust solution to this problem. It leverages `asyncio.create_task` to ensure that `process_message` is treated as an independent task within the event loop.

```python
import asyncio
import websockets
import json

async def process_message(message):
    """Processes an incoming message, properly awaited."""
    data = json.loads(message)
    print(f"Received price: {data['price']}")
    await asyncio.sleep(0.01)  # Example of an awaitable operation

async def subscribe_to_prices():
    uri = "wss://example.com/ws/prices"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            asyncio.create_task(process_message(message)) # Correct, task created

async def main():
    await subscribe_to_prices()

if __name__ == "__main__":
    asyncio.run(main())
```

The key change in this example is the use of `asyncio.create_task`. By wrapping the call to `process_message` in `asyncio.create_task`, we tell the event loop to treat this call as a separate, concurrent task. Critically, we don't need to await the result of this newly created task directly inside the loop itself. It can run independently without blocking the WebSocket reading loop, while still being managed and run to completion by the event loop. The asynchronous call to `await asyncio.sleep(0.01)` is included as a basic example, but it is here to simulate real work being done inside this function that needs to be awaited.

**Example 3: Correct Approach with Task Group**

This approach introduces `asyncio.TaskGroup`, a powerful method for managing and awaiting multiple concurrent tasks, especially useful for ensuring all created tasks are completed before exiting scope.

```python
import asyncio
import websockets
import json

async def process_message(message):
    """Processes an incoming message within a task group."""
    data = json.loads(message)
    print(f"Received price: {data['price']}")
    await asyncio.sleep(0.01) # Example of an awaitable operation

async def subscribe_to_prices():
    uri = "wss://example.com/ws/prices"
    async with websockets.connect(uri) as websocket:
      async with asyncio.TaskGroup() as tg:
         async for message in websocket:
             tg.create_task(process_message(message)) # Correct, task created

async def main():
    await subscribe_to_prices()

if __name__ == "__main__":
    asyncio.run(main())
```

In this version, `asyncio.TaskGroup` creates a context that ensures all the created tasks within it are awaited, providing a structured way to handle multiple concurrently running asynchronous operations related to the WebSocket messages. It guarantees that the tasks within the group are finished before the group exits, thus avoiding the error.

To summarize, the "Never awaited callback" error signals a critical oversight in managing asynchronous operations within an event-driven context. The core problem lies in failing to await the result of asynchronous tasks, leaving them in a pending state, unknown by the event loop. To rectify this, use `asyncio.create_task` to allow the event loop to manage each asynchronous operation efficiently, or even better, `asyncio.TaskGroup` to more carefully structure the execution and awaiting of those tasks.

Further, I recommend focusing on resources that specifically deal with the following topics, as they are key to a deeper understanding and will help you avoid similar pitfalls in the future:

1.  **Asynchronous Programming:** Study the fundamental concepts of asynchronous programming, focusing on coroutines, event loops, and the `async`/`await` keywords. Specifically look at how different frameworks handle the execution of coroutines, and their respective differences.

2.  **`asyncio` Library:** Deep-dive into Python's `asyncio` module. Understand its event loop, task creation, cancellation, and task groups for robust concurrent programming. In particular, note how functions like `create_task` and the `TaskGroup` operate, and when each one should be selected for use.

3.  **WebSocket Libraries:**  Carefully review the documentation of the WebSocket library you are using (e.g. `websockets`). Specifically, study its event handling model, message delivery mechanisms, and how these interact with `asyncio`. These libraries often provide examples that will help you identify the proper techniques for handling the asynchronous nature of message streams.

By mastering these resources, one will be well equipped to build robust and error-free asynchronous applications that effectively handle real-time data streams, such as those coming from a cryptocurrency exchange's WebSocket API. Through personal experience, I’ve found a strong understanding of these principles is paramount for building reliable and performant asynchronous applications.
