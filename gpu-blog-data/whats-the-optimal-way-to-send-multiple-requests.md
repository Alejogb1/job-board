---
title: "What's the optimal way to send multiple requests from a FastAPI API?"
date: "2025-01-30"
id: "whats-the-optimal-way-to-send-multiple-requests"
---
Batching requests efficiently within a FastAPI application requires a careful consideration of underlying asynchronous capabilities and potential bottlenecks. Overly simplistic approaches can lead to suboptimal performance, especially when dealing with high-throughput scenarios. My experience building microservices for a large-scale e-commerce platform highlighted the critical need for a robust, asynchronous solution rather than synchronous, sequential operations.

A naive implementation might involve a loop executing individual HTTP requests one after another. This approach inherently suffers from latency; each request must complete before the next is initiated, leading to a serial execution path. The server spends significant time waiting for I/O bound operations to finish, rather than maximizing concurrency. While relatively straightforward to implement, this strategy becomes a severe limitation as the number of requests increases. To mitigate this, FastAPI and its underlying framework, Starlette, provide mechanisms for efficient asynchronous request handling.

The optimal approach revolves around leveraging Python's `asyncio` library and the `async` and `await` keywords to achieve concurrent execution of requests. Essentially, we create a series of asynchronous tasks, initiate them, and then `await` their completion, allowing the application to handle other processes while waiting for the individual tasks to finish. This paradigm allows FastAPI to non-blockingly handle multiple requests concurrently, dramatically reducing overall processing time. For this, one would usually use a client library offering async capabilities, such as `httpx`. If the target endpoint is also an internal FastAPI instance, it will also benefit from the concurrency model and not waste time on a blocking event loop.

The following example illustrates the contrast between a synchronous and an asynchronous request pattern. The synchronous implementation iterates through a list of IDs, creating and dispatching requests sequentially. The asynchronous implementation, on the other hand, uses `asyncio.gather` to concurrently execute multiple requests.

```python
from fastapi import FastAPI
import httpx
import time

app = FastAPI()

# Synchronous Implementation
@app.get("/sync/{num_requests}")
def sync_requests(num_requests: int):
    start_time = time.time()
    for i in range(num_requests):
       response = httpx.get("https://httpbin.org/get") # Example URL. Replace with target API.
    end_time = time.time()
    return {"message": f"Processed {num_requests} requests synchronously in {end_time - start_time:.2f} seconds"}
```
This synchronous approach directly blocks the server on each `httpx.get()` call, preventing other requests from being processed until this specific request completes. This pattern is acceptable for a very limited number of operations, but performs poorly under load. `httpbin.org/get` is chosen as an external API to simulate a standard network request.

```python
import asyncio
from fastapi import FastAPI
import httpx
import time

app = FastAPI()

# Asynchronous Implementation
async def fetch_data(client, id):
  response = await client.get("https://httpbin.org/get") # Example URL. Replace with target API.
  return response.status_code

@app.get("/async/{num_requests}")
async def async_requests(num_requests: int):
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        tasks = [fetch_data(client, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    end_time = time.time()
    return {"message": f"Processed {num_requests} requests asynchronously in {end_time - start_time:.2f} seconds", "status_codes": results}
```
This asynchronous version utilizes `httpx.AsyncClient()` to conduct non-blocking requests. Each call to `fetch_data` is now an asynchronous task, allowing other tasks to execute while waiting for the network operations to complete. The `asyncio.gather` function ensures that all tasks are launched concurrently, and their results are collected upon completion. This approach provides significant performance improvements for multiple requests by enabling the server to remain responsive and non-blocking. The response now also includes the status code of each individual request, for further inspection.

An even further optimization can be introduced by limiting the number of concurrent requests dispatched at any given time. This is particularly useful if the target API has rate limits, or the target network has limited bandwidth. An unlimited number of concurrent requests can be detrimental to either the sender or receiver. The following example shows how to use `asyncio.Semaphore` to achieve this.

```python
import asyncio
from fastapi import FastAPI
import httpx
import time

app = FastAPI()

async def fetch_data_with_semaphore(client, semaphore, id):
    async with semaphore:
        response = await client.get("https://httpbin.org/get") # Example URL. Replace with target API.
        return response.status_code

@app.get("/async_sem/{num_requests}/{max_concurrent}")
async def async_requests_semaphore(num_requests: int, max_concurrent: int):
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient() as client:
        tasks = [fetch_data_with_semaphore(client, semaphore, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    end_time = time.time()
    return {"message": f"Processed {num_requests} requests asynchronously (with semaphore) in {end_time - start_time:.2f} seconds", "status_codes": results}
```

Here, we initialize `asyncio.Semaphore` with the `max_concurrent` parameter. Each task will attempt to acquire a permit from the semaphore before sending its request, blocking only if all permits are already acquired. This introduces backpressure and avoids overwhelming the target API. This example allows for easy control over the concurrency level. A value of 1 would equate to the synchronous request model, while a high value could push the system to its limits if the target server and network are incapable of handling it. The status codes are included in the result.

For resources, I would recommend consulting the official Python documentation for `asyncio`, focusing on `async` and `await` keywords, task creation, and synchronization primitives such as semaphores. The `httpx` documentation is also crucial for implementing non-blocking HTTP requests, along with the FastAPI documentation, particularly its chapters on asynchronous operations and background tasks. The concept of concurrency vs parallelism should be well understood. Finally, reading about rate limiting, both from a client and server perspective, will further aid the optimisation process. These resources offer extensive details and best practices that have guided my development of high-performance APIs.
