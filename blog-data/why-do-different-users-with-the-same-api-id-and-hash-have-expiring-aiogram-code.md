---
title: "Why do different users with the same API ID and hash have expiring aiogram code?"
date: "2024-12-23"
id: "why-do-different-users-with-the-same-api-id-and-hash-have-expiring-aiogram-code"
---

,  The issue of expiring aiogram code, even when users share the same API ID and hash, is actually more common than one might initially think. It points to a fundamental misunderstanding about how aiogram, and indeed, most asynchronous frameworks, operate in the context of multiple user sessions. I've personally run into this exact scenario during a project involving a Telegram bot meant for a small community; the frustration was real, and the fix wasn't immediately apparent.

The core problem isn't necessarily the api id and hash themselves; those are more analogous to a shared login credential. They're a gatekeeper, granting access to the telegram api. The real problem surfaces in how each user's *session* is handled, specifically with how the aiogram library structures its internal event loop and state management. Let's break down what’s happening:

Each user interaction with a bot (a message, a command, a callback) initiates a series of operations within aiogram. These operations aren't just processed directly; they are managed within an asynchronous event loop. When you start an aiogram bot, an event loop is initialized, often behind the scenes. This loop is like a central dispatcher, managing all incoming events and distributing them to the correct handlers. Now, consider this: you've configured your bot with an api id and hash, and multiple users are engaging with it simultaneously. Each user initiates their own separate *session*, which is often stored in memory or sometimes in a small local database. These sessions are what aiogram uses to associate a user with their specific state in the interaction (e.g., a user's position in a conversation, or the data they've provided in previous interactions).

The problem of expiring aiogram code, in the context of multiple users, is that these sessions are not designed to live indefinitely. They have a finite lifespan, mostly due to security protocols on the Telegram side and the management of resources on the aiogram bot's server. When a session expires, it can cause errors, force user re-authentication, or result in the code seemingly “not working” for that particular user. This isn't a bug in your aiogram code (usually), it’s a matter of understanding how sessions are handled and what triggers their expiration. Here are some typical culprits:

1. **Inactivity Timeout:** Telegram's api, and often aiogram itself, will expire a session if a user is inactive for a prolonged period. This mechanism prevents abuse and also frees up resources on the server.

2. **Server Restarts or Code Updates:** If your bot server restarts (e.g., due to a deployment), the in-memory session data is typically lost, forcing all users to re-establish their connections. Similarly, deploying a new version of the bot can lead to the invalidation of old sessions.

3. **Explicit Session Logout:** Though less common, some events can explicitly trigger a session logout on the Telegram side, though aiogram will usually try to reconnect in this case.

4. **Storage Limitations:** When session data is stored in memory, there's a risk of resource exhaustion, especially with a large number of users, which might cause sessions to be prematurely discarded.

So, how do we tackle this? We need to implement mechanisms to manage and persist sessions more reliably. Here are three code snippets that show how we might handle this:

**Example 1: Using `MemoryStorage` with a custom expiration:**

While memory storage isn’t the ideal solution for large-scale deployments, you can tweak the built-in `MemoryStorage` to demonstrate the concept of expiration. In this simplified form we'll manually simulate an invalidation:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import asyncio
import datetime

# assuming BOT_TOKEN, API_ID, and API_HASH are set.
storage = MemoryStorage()
bot = Bot(token="YOUR_BOT_TOKEN")
dp = Dispatcher(bot, storage=storage)

user_sessions = {}

async def check_session(user_id):
    session = user_sessions.get(user_id)
    if not session or (datetime.datetime.now() - session['last_activity']) > datetime.timedelta(minutes=10):
        return False
    return True

async def update_session_activity(user_id):
  user_sessions[user_id] = {'last_activity': datetime.datetime.now()}


@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
  user_id = message.from_user.id
  if await check_session(user_id):
     await message.reply("Welcome back!")
  else:
     await update_session_activity(user_id)
     await message.reply("Welcome, new user!")

@dp.message_handler()
async def echo_handler(message: types.Message):
  user_id = message.from_user.id
  if await check_session(user_id):
     await update_session_activity(user_id)
     await message.reply(message.text)
  else:
     await update_session_activity(user_id)
     await message.reply("Your session is new. Please restart using /start")

async def main():
    await dp.start_polling(allowed_updates=types.AllowedUpdates.all())

if __name__ == '__main__':
    asyncio.run(main())
```

This demonstrates a basic implementation to simulate session management, and shows how you might manually implement an expiration check. Note that `MemoryStorage` in aiogram does not offer direct expiration capabilities, hence the need to implement this manually for demonstration purposes.

**Example 2: Using a Database (PostgreSQL example)**

For a more robust solution, persisting session data in a database is essential. Here is an illustration using PostgreSQL with `aiogram.contrib.fsm_storage.redis`:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.redis import RedisStorage2
import asyncio
import redis

# Assuming BOT_TOKEN is set, along with redis settings
redis_host = 'localhost'  # Replace with your redis host if needed
redis_port = 6379 # Replace with your redis port if needed
redis_db = 0 # Replace with your redis database number if needed

storage = RedisStorage2(host=redis_host, port=redis_port, db=redis_db)
bot = Bot(token="YOUR_BOT_TOKEN")
dp = Dispatcher(bot, storage=storage)

@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
   await message.reply("Welcome!")


@dp.message_handler()
async def echo_handler(message: types.Message):
   await message.reply(message.text)

async def main():
    await dp.start_polling(allowed_updates=types.AllowedUpdates.all())

if __name__ == '__main__':
    asyncio.run(main())
```

This uses `RedisStorage2`, it persists session data using the redis database, allowing for session persistence across restarts or code updates. You can easily configure similar functionality with a PostgreSQL or MySQL database using their respective aiogram storage modules. Make sure that you have redis installed and accessible on your server.

**Example 3: Managing Session with explicit `bot.close()` and manual re-initialization**

While not ideal for general usage, explicitly closing and re-initializing the bot instance can force session refreshes on the user side. This is useful during development or when you want to force a fresh state for testing. While it doesn't *solve* session expiry, it demonstrates how session lifecycles are tied to the bot's running instance, and it allows you to understand the impact of restarting:

```python
from aiogram import Bot, Dispatcher, types
import asyncio
import time


bot = Bot(token="YOUR_BOT_TOKEN")
dp = Dispatcher(bot)

async def run_bot():
    await dp.start_polling(allowed_updates=types.AllowedUpdates.all())

async def main():
    while True:
        try:
            print("Starting bot...")
            await run_bot()
        except Exception as e:
            print(f"Exception caught: {e}")
            await bot.close() # explicit close to trigger session loss
            print("Bot closed, will restart after 10 seconds")
            await asyncio.sleep(10)  # optional: add a delay
        else:
            print("Bot finished. Restarting...")


@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    await message.reply("Hello, new session!")


@dp.message_handler()
async def echo_handler(message: types.Message):
   await message.reply(message.text)


if __name__ == '__main__':
    asyncio.run(main())
```

This example restarts the bot after any exception, effectively causing a 'session reset' for all users, showcasing how bot instance lifecycle affects the underlying session management.

For a more in-depth understanding of how these mechanisms work, I'd recommend delving into the source code of the `aiogram.contrib.fsm_storage` modules and thoroughly reading the official documentation of `aiogram`. Specifically, focusing on the sections that discuss state persistence and asynchronous programming models. Additionally, the book "Programming Telegram Bots" by Ahmed Youssef provides a strong overview of building robust and reliable Telegram bots. For async concepts, David Beazley’s talks and writings on Python concurrency are always gold. Remember, the key is understanding how the sessions are maintained and managed within the context of the framework to address session expiration and create a seamless experience for the users.
