---
title: "How does Aiogram utilize multiprocessing?"
date: "2024-12-23"
id: "how-does-aiogram-utilize-multiprocessing"
---

Let’s delve into how aiogram leverages multiprocessing—a topic I’ve encountered several times in my years working with asynchronous Python frameworks. It's not always immediately obvious how a framework primarily designed around concurrency with `asyncio` also manages to utilize multiprocessing, and this often becomes a key consideration when scaling bot applications to handle high volumes of requests.

The short answer is: aiogram itself doesn’t inherently use multiprocessing directly in the way you might use Python's `multiprocessing` module for CPU-bound tasks. Instead, aiogram relies primarily on asyncio's cooperative multitasking within a single process to handle concurrent Telegram API interactions. However, it *can* be indirectly connected to multiprocessing scenarios, typically when needing to run CPU-intensive operations that would block the main asyncio event loop, or when distributing workload across different machines. This indirection is usually facilitated by the application architecture, not by aiogram’s core mechanics.

The core of aiogram is built around asyncio. This means that all the telegram API requests, updates, and internal operations are non-blocking. When an update comes in, the main event loop schedules the appropriate handlers to be called. These handlers should ideally be quick and non-blocking; if a handler performs a long-running task like heavy computation or disk i/o, it will hold up the event loop, preventing other updates from being processed, and thus resulting in a bottleneck.

Here's where multiprocessing can become useful, but it’s crucial to understand that aiogram isn't inherently orchestrating this. Instead, you, as the application developer, are making use of multiprocessing facilities to move CPU-bound tasks away from the main asyncio loop where aiogram is processing updates. This usually involves creating separate processes which aiogram then interacts with through inter-process communication (ipc) mechanisms.

Let me illustrate this with a few examples, based on similar situations i have handled before. I once worked on a bot that needed to perform complex image processing after receiving certain types of image updates. Performing this directly within the aiogram handler severely degraded the bot's performance under load. Therefore, I had to implement multiprocessing to prevent the main event loop from being blocked.

Here's a simplified example using Python's `multiprocessing` module:

```python
import asyncio
import multiprocessing
import time
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import os

TOKEN = os.getenv('BOT_TOKEN') # Replace with your token

bot = Bot(TOKEN)
dp = Dispatcher()

def cpu_bound_task(input_data):
    time.sleep(2)  # Simulate a cpu-intensive task
    return f"Processed: {input_data}"

async def process_in_separate_process(input_data):
    with multiprocessing.Pool(processes=4) as pool: # Example usage with 4 processes
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            pool.apply,
            cpu_bound_task,
            (input_data,)
        )
        return result

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    result = await process_in_separate_process("initial data")
    await message.answer(f"Start command processed: {result}")

@dp.message()
async def any_message(message: types.Message):
    result = await process_in_separate_process(message.text)
    await message.answer(f"Processed your message: {result}")


async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())

```

In this example, the `process_in_separate_process` function uses `multiprocessing.Pool` to create a pool of worker processes. `asyncio.get_running_loop().run_in_executor` is used to run the blocking `pool.apply` call in a separate thread, which prevents the main asyncio event loop from blocking. The `cpu_bound_task` represents a function performing a computationally expensive operation. When the bot receives the `/start` command, it triggers `process_in_separate_process` which executes `cpu_bound_task` in separate process and returns the result via ipc.

Another common pattern is using a message queue like Redis or RabbitMQ to distribute tasks. This provides more flexibility and robustness, especially when the application needs to scale horizontally across multiple machines. Here's how one might integrate such a queue with aiogram. This approach becomes beneficial when you might want to have completely decoupled systems with the telegram bot handling requests while other back-end processes deal with processing data.

```python
import asyncio
import aioredis
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import os
import json

TOKEN = os.getenv('BOT_TOKEN')  # Replace with your token
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

bot = Bot(TOKEN)
dp = Dispatcher()


async def enqueue_task(redis_conn, task_data):
    await redis_conn.rpush("task_queue", json.dumps(task_data))

async def dequeue_task(redis_conn):
    _, task_json = await redis_conn.blpop("task_queue", timeout=0)
    return json.loads(task_json)

async def process_queued_task(task_data):
    await asyncio.sleep(2)  # Simulate cpu-intensive processing
    return f"Processed task data: {task_data}"

async def worker_process(redis_conn):
    while True:
        task_data = await dequeue_task(redis_conn)
        if task_data:
            result = await process_queued_task(task_data)
            print(f"Worker process finished: {result}")

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    redis = await aioredis.Redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}")
    task_data = {"user_id": message.from_user.id, "text": "some initial work"}
    await enqueue_task(redis, task_data)
    await message.answer("Task enqueued!")


@dp.message()
async def any_message(message: types.Message):
    redis = await aioredis.Redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}")
    task_data = {"user_id": message.from_user.id, "text": message.text}
    await enqueue_task(redis, task_data)
    await message.answer("Task enqueued!")

async def main():
    redis = await aioredis.Redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}")
    asyncio.create_task(worker_process(redis))  # Run worker process
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
```

In this code, messages received by the bot are enqueued into Redis using `enqueue_task`. A separate `worker_process` constantly monitors the queue using `dequeue_task` and processes messages as they come. The work is still async, but it runs in parallel with the main bot process since it is executed inside an independent task that operates with the help of a message broker. This enables better scaling and system decoupling.

Finally, I've also used a more advanced approach involving distributed task queues like Celery, especially when dealing with very large-scale systems. This might be overkill for smaller applications, but it is the right choice when the demands of your bot grow to a significant scale.

```python
from celery import Celery
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import os

TOKEN = os.getenv('BOT_TOKEN')  # Replace with your token
CELERY_BROKER = 'redis://localhost:6379/0'

celery_app = Celery('tasks', broker=CELERY_BROKER)

@celery_app.task
def process_celery_task(task_data):
    import time
    time.sleep(2)
    return f"Processed celery task: {task_data}"


bot = Bot(TOKEN)
dp = Dispatcher()


async def send_to_celery(task_data):
   result = process_celery_task.delay(task_data)
   return result.id # you would need to implement logic for result retrieval

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    task_id = await send_to_celery({"type": "start_command"})
    await message.answer(f"Celery task started with id: {task_id}")


@dp.message()
async def any_message(message: types.Message):
    task_id = await send_to_celery({"type": "message", "text": message.text})
    await message.answer(f"Celery task started with id: {task_id}")


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())

```

In this example, `process_celery_task` is decorated with `@celery_app.task`, enabling it to be run as a background task by Celery. When an aiogram handler calls `send_to_celery`, the task is handed to celery, and celery takes care of executing it independently.

In all of these examples, the crucial thing to remember is that aiogram itself doesn't initiate multiprocessing. Instead, we, as developers, architect our applications to employ these techniques alongside aiogram, particularly when the asynchronous nature of asyncio isn’t enough. Understanding these patterns will significantly enhance your ability to build robust and scalable bot applications using aiogram. For a deeper understanding of async programming, I recommend "Effective Python: 90 Specific Ways to Write Better Python" by Brett Slatkin and "Programming in Lua" by Roberto Ierusalimschy, which delves into the underlying concepts that often power such systems and help gain a holistic view of concurrent programming concepts. For a deeper understanding of multiprocessing in python, "Python Cookbook" by David Beazley and Brian K. Jones offers many practical examples.
