---
title: "Why is my FastAPI server unresponsive due to 'bvar is busy' logs?"
date: "2025-01-30"
id: "why-is-my-fastapi-server-unresponsive-due-to"
---
A "bvar is busy" log message within a FastAPI application typically signals a contention issue within the underlying asynchronous machinery, specifically with how background tasks (or similar asynchronous operations) are being handled. This message, though somewhat cryptic, points towards an event loop blockage arising from concurrent modifications to a shared mutable object, often within an asyncio task or a function that's part of a route's execution. My experience deploying and maintaining several high-throughput APIs using FastAPI has shown that these errors almost always stem from a misunderstanding of asyncio's cooperative multitasking model and how it interacts with mutable state.

The core issue isn't that FastAPI itself is failing; rather, it’s that an underlying asynchronous operation isn’t yielding control back to the event loop in a timely fashion. When this happens, other scheduled tasks – including processing subsequent client requests – are delayed. This delay manifests as the “bvar is busy” error, hinting that the object’s access is blocked because another task currently has control, and that further attempts to access the shared mutable object are queued. The situation is exacerbated by the fact that Python’s global interpreter lock (GIL) serializes thread execution in CPython, leading to single-core bottlenecks in CPU-bound tasks, and that the event loop does not preempt tasks, relying on yielding control. In short, the “bvar” likely represents a shared variable in memory that is being accessed and modified by multiple async operations, in ways that are not thread-safe. This is unlike standard threading where the operating system can interrupt a thread's execution for another; in async, the programmer is explicitly in charge of when an async process will pause execution to allow the event loop to check on other pending processes.

To better illustrate the root of this problem, consider a scenario where multiple endpoints of a FastAPI application need to update a shared dictionary. Without explicit coordination, this leads to race conditions, which are exactly what async's cooperative nature and the "bvar is busy" messages are trying to warn us against. Let's examine a simple FastAPI endpoint with this issue:

```python
from fastapi import FastAPI, BackgroundTasks
import asyncio
import time

app = FastAPI()

shared_data = {"counter": 0}

async def update_counter_unsafe():
    global shared_data
    for _ in range(10000):
        shared_data["counter"] += 1
        await asyncio.sleep(0.00001) #simulate I/O

@app.get("/unsafe-update")
async def unsafe_update(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_counter_unsafe)
    return {"status": "update initiated"}

```

This code demonstrates an unsafe usage of shared mutable state. The `update_counter_unsafe` coroutine is modifying the global `shared_data` dictionary without any synchronization mechanisms. While asyncio operations (like `await asyncio.sleep()`) allow the event loop to process other tasks, modifying shared mutable state concurrently can lead to data corruption. If multiple requests call the `/unsafe-update` endpoint simultaneously, this function will likely trigger "bvar is busy" warnings, and the counter value may also be incorrect due to the race conditions. The use of asyncio.sleep() does not guarantee thread safety, instead only yielding control of the event loop for a moment. In practice, a more realistic scenario might have the dictionary storing application configurations, results from expensive lookups, or user sessions; these can be a significant source of bugs when they’re mutated simultaneously by multiple requests.

The key takeaway from the previous example is that we need proper synchronization when accessing mutable shared resources in an asyncio context. The most practical way to accomplish this is through thread-safe primitives, provided by the `asyncio` library and other Python concurrency libraries. Here's how the previous example could be improved:

```python
from fastapi import FastAPI, BackgroundTasks
import asyncio
from asyncio import Lock
import time

app = FastAPI()

shared_data = {"counter": 0}
lock = Lock()

async def update_counter_safe():
    global shared_data
    for _ in range(10000):
        async with lock:
            shared_data["counter"] += 1
        await asyncio.sleep(0.00001)

@app.get("/safe-update")
async def safe_update(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_counter_safe)
    return {"status": "update initiated"}
```

In this version, an `asyncio.Lock` is used to control access to the shared dictionary. This ensures that only one coroutine can modify the `shared_data` dictionary at a time, preventing race conditions and the consequent "bvar is busy" messages. The `async with lock:` statement ensures exclusive access for as long as the block executes. When multiple background tasks are launched, they will take turns modifying the dictionary, preventing conflicts. While this doesn’t eliminate context switching, it does eliminate race conditions. The `Lock` also serves another subtle, but important, purpose: if another task was also attempting to access the lock, it will yield control to the event loop, therefore eliminating the cause of the “bvar is busy” error.

Another important pattern, common with FastAPI and other frameworks that use dependency injection, is to understand when a dependency is created, and how frequently it is recreated. Consider this example:

```python
from fastapi import FastAPI, Depends, BackgroundTasks
import asyncio
import aiohttp

app = FastAPI()

async def make_client():
    async with aiohttp.ClientSession() as session:
        return session

@app.get("/make_client")
async def make_client_endpoint(background_tasks: BackgroundTasks, client = Depends(make_client)):
    background_tasks.add_task(example_request, client)
    return {"status": "client created"}

async def example_request(client):
    # some logic that uses the client goes here
    await asyncio.sleep(0.01)
```

This code will also result in "bvar is busy" log statements because a new aiohttp.ClientSession is created for *every* request. Since there's a timeout associated with creating a ClientSession, and `make_client()` is created concurrently with multiple requests, contention will occur when trying to complete all the ClientSession setup. A proper solution is to create the ClientSession once, and then provide that reference for all requests. Since FastAPI allows dependency injection, this is relatively trivial to fix:

```python
from fastapi import FastAPI, Depends, BackgroundTasks
import asyncio
import aiohttp

app = FastAPI()
client = None

async def startup_event():
    global client
    client = aiohttp.ClientSession()

app.add_event_handler("startup", startup_event)

async def get_client():
    return client

@app.get("/make_client")
async def make_client_endpoint(background_tasks: BackgroundTasks, client = Depends(get_client)):
    background_tasks.add_task(example_request, client)
    return {"status": "client created"}

async def example_request(client):
    # some logic that uses the client goes here
    await asyncio.sleep(0.01)

async def shutdown_event():
    global client
    await client.close()

app.add_event_handler("shutdown", shutdown_event)
```

This version initializes the client as a global singleton, and uses an event handler to create a single aiohttp.ClientSession during the server’s startup, and closes it at shutdown. Then, the dependency `get_client()` will return the same client object for all requests, avoiding race conditions related to concurrent creation.

In summary, when encountering "bvar is busy" logs in a FastAPI application, the diagnostic process should involve careful analysis of the application's handling of shared mutable data, a deep understanding of event-loop cooperative multitasking, and a strong reliance on proper synchronization primitives. It's easy to fall into the trap of thinking that simply using async/await will magically eliminate race conditions, and that's not the case. Resources that offer a solid foundation on asynchronous Python programming and the intricacies of the `asyncio` library should be reviewed. For instance, I would recommend exploring official Python documentation on `asyncio`, and delving into specific sections on `Lock` and other synchronization primitives. Furthermore, exploring the details of dependency injection specific to your chosen framework is useful. Ultimately, a well-architected application will address these potential issues before they arise in production.
