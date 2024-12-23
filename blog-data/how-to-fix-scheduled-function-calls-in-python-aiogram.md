---
title: "How to fix scheduled function calls in Python aiogram?"
date: "2024-12-23"
id: "how-to-fix-scheduled-function-calls-in-python-aiogram"
---

Alright,  I remember a particularly thorny incident a few years back involving a high-volume telegram bot I was managing, built with aiogram. Scheduled functions, or rather, their *failure* to execute reliably, became a significant source of headaches. The problem often isn't with the scheduler *itself* but with how we're integrating it into the asynchronous framework aiogram provides. Here's what I've learned, and how to approach these issues, focusing on pragmatic solutions over abstract theory.

The core problem, as I've frequently seen, lies in the interaction between aiogram's asynchronous event loop and traditional, blocking scheduling mechanisms. Simple timers like `time.sleep()` are a no-go because they freeze the event loop and prevent other handlers from processing incoming messages, rendering your bot unresponsive. Similarly, naive implementations using `schedule` or similar libraries often fail within the async context because they weren't designed for aiogram's environment. The consequence is missed scheduled calls, unexpected delays, or outright crashes if not handled properly.

The most robust solution, and what I ended up implementing, involves leveraging asyncio’s built-in capabilities in conjunction with aiogram’s context. Basically, instead of relying on external blocking timers, we schedule coroutines to execute at specific intervals or times. This ensures everything remains within the asynchronous flow. There are three primary approaches I’ve found consistently effective.

**1. Using `asyncio.sleep` for simple delays**

For simple, recurring tasks, `asyncio.sleep()` within a separate asynchronous task is often sufficient. This method involves launching a background task that sleeps for a defined duration and then executes a function. I used this extensively for sending periodic notifications, for example.

```python
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message
import logging

API_TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN' #Replace with your real bot token

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


async def periodic_task():
    while True:
        logging.info("Executing scheduled task...")
        # Your logic here - e.g., sending a message
        await bot.send_message(chat_id='YOUR_USER_CHAT_ID', text="Periodic message")
        await asyncio.sleep(60) # wait 60 seconds before the next execution

async def main():
    asyncio.create_task(periodic_task())
    # Start polling here to handle incoming messages
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
```
Here, `periodic_task()` runs in an infinite loop, executing our function, sleeping, and then repeating. The crucial part is `asyncio.sleep()` – it yields control back to the event loop, allowing other tasks (like message handling) to proceed. This approach works well for background jobs that can tolerate minor timing variations. Note I've added proper logging, a crucial habit in development, and a placeholder for your bot token and user chat id; these need to be filled with your actual data. You should also implement robust error handling around the message sending to avoid your task crashing if the bot encounters an error.

**2. Using `asyncio.create_task` with `datetime` for specific times**

For schedules that need to adhere to a precise time (e.g., a daily report), simply sleeping for a fixed amount isn't enough. You'll need to calculate the time difference until the next scheduled execution point. This is where `datetime` comes in.

```python
import asyncio
import datetime
from aiogram import Bot, Dispatcher, types
import logging

API_TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN' #Replace with your real bot token

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

async def scheduled_task():
    while True:
      now = datetime.datetime.now()
      target_time = now.replace(hour=10, minute=0, second=0, microsecond=0) # Set to 10 AM every day
      if now > target_time: # Handle case where it's already past the target time today
        target_time = target_time + datetime.timedelta(days=1)
      time_diff = (target_time - now).total_seconds()
      logging.info(f"Next task execution in {time_diff} seconds")
      await asyncio.sleep(time_diff)
      # Your scheduled logic here
      await bot.send_message(chat_id='YOUR_USER_CHAT_ID', text="Daily Report")

async def main():
    asyncio.create_task(scheduled_task())
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
```

This approach calculates the time difference between the current time and the next scheduled time, sleeps for that duration, and then executes the task. It's important to handle cases where the current time is already past the target time for the day, as shown in the code. Again, logging and proper error handling within the scheduled task is crucial. This approach is more flexible than the previous one, letting you set specific times for your scheduled actions.

**3. Using a dedicated scheduler with an asynchronous wrapper**

For more complex scheduling scenarios, like using cron-like expressions, you might benefit from integrating an asynchronous scheduler library such as `apscheduler`. This approach requires an intermediary wrapper that transforms the blocking library functions to async compatible versions by using `asyncio.to_thread()`. While technically this is utilizing threads, in the context of i/o bound operations, it doesn't break the event loop's concurrency in a meaningful way and provides a flexible solution. This approach is best when you need very complex scheduling patterns that can't be easily implemented with the previous techniques.

```python
import asyncio
from datetime import datetime
from aiogram import Bot, Dispatcher, types
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

API_TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN' #Replace with your real bot token

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

async def send_message_wrapper():
    await bot.send_message(chat_id='YOUR_USER_CHAT_ID', text="Cron Scheduled Message!")

async def main():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(send_message_wrapper, CronTrigger(hour="*", minute="0", second="0")) # Every hour at 0 minutes and 0 seconds
    scheduler.start()
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
```

Here, `apscheduler` manages the schedule, but I've wrapped the task itself in an asynchronous function. In situations where you need a large number of complex, dynamically-created scheduled tasks, a dedicated library like this is far superior in terms of manageability. In essence, it acts as a scheduling engine that can execute async compatible code, greatly expanding the scheduling possibilities. Always check if a library explicitly supports asyncio first, as there might be a more direct integration available.

In all these examples, it's vital to handle potential exceptions that may arise during task execution. You don't want your entire bot to crash because of a single failed scheduled call.

For further reading, I highly recommend delving into “Concurrent Programming in Python” by David Beazley. This book is a great resource for understanding how asyncio works under the hood. Also, review the official asyncio documentation for more clarity on the fundamentals. Additionally, researching documentation of `apscheduler` is crucial to understanding the scheduling possibilities it offers if you decide to implement that approach.

Remember, the key is understanding how asynchronous programming works, and the limitations of blocking operations within the aiogram context. By using `asyncio.sleep()`, combining it with `datetime` or using a proper asynchronous scheduling library, you can handle scheduled task reliably and keep your bot functional and performant.
