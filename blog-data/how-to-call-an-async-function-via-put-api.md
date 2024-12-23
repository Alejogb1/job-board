---
title: "How to call an async function via PUT API?"
date: "2024-12-16"
id: "how-to-call-an-async-function-via-put-api"
---

, let's talk about calling an asynchronous function via a put api. This isn't a straightforward, cookie-cutter scenario, and I've certainly seen my fair share of teams trip over this in the past. The core issue boils down to how http and async operations interact—or rather, *don't* interact inherently. A put request, like other http methods, expects a response within a reasonable timeframe. Async functions, by their nature, don't always complete immediately; they might be waiting for i/o, network operations, or other tasks that take time. We need to bridge this gap.

The fundamental challenge is that you can't directly "return" the result of an asynchronous operation in a synchronous http handler. When a put request comes in, our http server needs to generate a response. Blocking that request while waiting for the async operation to complete will lock up the server, leading to performance issues and potentially dropped connections. Instead, we need to decouple the request handling from the actual async operation.

In a past life, I was working on a system that processed image uploads. Users would send a put request with an image, and we needed to run several potentially lengthy operations on that image: resizing, format conversion, metadata extraction, etc. These were perfect candidates for asynchronous processing. Our initial naive approach was to try to do it all within the put handler, resulting in awful response times and frequent timeouts. We quickly learned that the put endpoint should acknowledge receipt and immediately return, leaving the actual processing to the background.

So, how do we do it? We have a few common patterns to accomplish this, each with its pros and cons. The core idea involves a job queue or message broker. Here’s a general breakdown:

1.  **Receive the Put Request:** The put endpoint receives the request, validates the data (as much as possible quickly), and stores the details needed for the async operation, typically including the resource id and payload.

2.  **Enqueue the Job:** Instead of running the operation directly, we create a 'job' representation of the async task and place it on a message queue or similar system (think rabbitmq, redis lists, kafka etc.). This queue acts as an intermediary, allowing us to handle requests rapidly without blocking.

3.  **Acknowledge the Request:** The put handler responds to the client with a success code (e.g., 202 Accepted), perhaps with a resource location or a job id that allows the client to track the status.

4.  **Async Processing:** A separate worker process or processes listen to this job queue and grab jobs as they appear. These workers then perform the actual async operation.

5.  **Response and Status Updates:** Depending on the system design, after the async operation, we may send additional notifications (like an email) or write changes to a data store where the client can subsequently check for updates. We must avoid pushing complex payloads in the initial put response body, as this can go against the idea of an immediate ack.

Let's get into some code examples. These examples use python with aiohttp for the put api and redis for the message queue; keep in mind that this is just an illustration, and the core logic can translate to other languages and tools.

**Example 1: Basic Job Enqueue with Redis**

```python
import aiohttp
from aiohttp import web
import asyncio
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def process_async_task(resource_id, payload):
    # Simulating an asynchronous operation. Replace with your actual logic
    await asyncio.sleep(2)
    print(f"Processing task for resource_id {resource_id} with payload: {payload}")
    # Example: Store results in redis for later retrival
    redis_client.set(f"result:{resource_id}", "processed")

async def put_handler(request):
    try:
        data = await request.json()
        resource_id = data.get('resource_id')
        payload = data.get('payload')
        if not resource_id or not payload:
            return web.Response(status=400, text="Missing resource_id or payload")

        # Enqueue job to redis (message queue)
        redis_client.lpush("task_queue", f"{resource_id}:{payload}")

        return web.Response(status=202, text=f"Task enqueued for processing, Resource id {resource_id}", content_type="text/plain")

    except Exception as e:
        return web.Response(status=500, text=str(e))

async def worker():
    while True:
        try:
            _, task_data = redis_client.blpop("task_queue", timeout=1)
            if task_data:
                task_data = task_data.decode()
                resource_id, payload = task_data.split(":", 1)
                await process_async_task(resource_id, payload)
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {e}")
            await asyncio.sleep(5)

async def main():
    app = web.Application()
    app.add_routes([web.put('/api/resource', put_handler)])
    asyncio.create_task(worker()) #start the worker in the background

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    print("Server started...")
    await asyncio.Future()  # Keep the server running indefinitely

if __name__ == '__main__':
    asyncio.run(main())
```

In this example, we are using redis list to create a basic queue. When a put request comes in, we push the necessary data to this redis list. The worker process, running in the background, pulls data out of the queue and processes the asynchronous tasks.

**Example 2: Using a proper message queue system (RabbitMQ)**

```python
import aiohttp
from aiohttp import web
import asyncio
import aio_pika

RABBITMQ_URL = "amqp://guest:guest@localhost/"
QUEUE_NAME = "task_queue"

async def process_async_task(resource_id, payload):
    # Simulating asynchronous task
    await asyncio.sleep(2)
    print(f"Processing task for resource_id {resource_id} with payload: {payload}")

async def put_handler(request):
    try:
        data = await request.json()
        resource_id = data.get('resource_id')
        payload = data.get('payload')
        if not resource_id or not payload:
            return web.Response(status=400, text="Missing resource_id or payload")

        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        async with connection:
            channel = await connection.channel()
            await channel.declare_queue(QUEUE_NAME, durable=True)
            message_body = f"{resource_id}:{payload}"
            message = aio_pika.Message(body=message_body.encode(), delivery_mode=2)
            await channel.default_exchange.publish(message, routing_key=QUEUE_NAME)

        return web.Response(status=202, text=f"Task enqueued for processing with resource id {resource_id}", content_type="text/plain")
    except Exception as e:
        return web.Response(status=500, text=str(e))

async def worker():
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(QUEUE_NAME, durable=True)

        async def process_message(message):
            async with message.process():
                task_data = message.body.decode()
                resource_id, payload = task_data.split(":", 1)
                await process_async_task(resource_id, payload)

        await queue.consume(process_message)
        await asyncio.Future() #keep the consumer alive forever

async def main():
    app = web.Application()
    app.add_routes([web.put('/api/resource', put_handler)])
    asyncio.create_task(worker())

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    print("Server started...")
    await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())

```

This second example uses RabbitMQ, a robust message broker. The core logic remains the same: we enqueue the job onto rabbitmq, and a worker consumes and processes it.

**Example 3: Using Celery with Redis as a broker**

```python
import aiohttp
from aiohttp import web
import asyncio
from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@celery_app.task
def process_async_task(resource_id, payload):
    import asyncio
    # Simulating async task
    asyncio.run(asyncio.sleep(2))
    print(f"Processing task for resource_id {resource_id} with payload: {payload}")

async def put_handler(request):
    try:
        data = await request.json()
        resource_id = data.get('resource_id')
        payload = data.get('payload')
        if not resource_id or not payload:
           return web.Response(status=400, text="Missing resource_id or payload")

        # Enqueue task via celery
        process_async_task.delay(resource_id, payload)
        return web.Response(status=202, text=f"Task enqueued for processing, Resource id {resource_id}", content_type="text/plain")

    except Exception as e:
        return web.Response(status=500, text=str(e))

async def main():
    app = web.Application()
    app.add_routes([web.put('/api/resource', put_handler)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    print("Server started...")
    await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())
```

Finally, this last example uses Celery. Celery is a popular task queue library. We simply define a task via the decorator `@celery_app.task`, and we invoke this task with `.delay()` on the http endpoint when the put request is received.

As for learning more, I would strongly recommend looking into the following resources:

*   **"Enterprise Integration Patterns" by Gregor Hohpe and Bobby Woolf:** This book is a must-read if you are dealing with async processing at any scale; it describes a wide range of integration patterns, including those related to message queues.
*   **"Designing Data-Intensive Applications" by Martin Kleppmann:** Provides deep insight into the challenges of building reliable, scalable, and maintainable applications. Specifically, its chapters on messaging, distributed systems and data storage will be invaluable for this scenario.
*   **The official documentation for your specific message broker.** Whether you go with RabbitMQ, Kafka, Redis Streams or something else, understanding the specifics of that technology will be crucial.

In conclusion, calling an async function via a put API isn't directly possible due to the synchronous nature of http. The solution is to decouple request handling from the async operation using a job queue or message broker. The example code snippets above offer a few patterns.  Remember, building a solid system requires careful consideration of your specific needs, scaling requirements, and failure modes.
