---
title: "How can I delete old bot messages when sending new ones in aiogram?"
date: "2024-12-23"
id: "how-can-i-delete-old-bot-messages-when-sending-new-ones-in-aiogram"
---

Alright, let's talk about managing those pesky bot messages when using aiogram. It's a scenario I've bumped into countless times during my tenure developing Telegram bots, especially when dealing with rapidly changing information or interactive interfaces. You're essentially trying to avoid message clutter by removing the old context before presenting the new, and it's a very valid concern. Simply flooding a chat with updates leads to a poor user experience.

The core problem stems from Telegram's architecture. When you send a message via the Bot API, it becomes a permanent fixture, at least until manually removed by the bot or the user. There isn't an inherent 'replace' function, so you have to manage the removal process yourself. Thankfully, aiogram gives us the tools to do this effectively. The fundamental approach involves remembering the message ids of previously sent messages and using `bot.delete_message()` to clean them up before sending new ones. Let's break down how to achieve this cleanly and reliably.

The first step is to think about the scope. How long do you need to retain the old messages id? Is it just for the duration of the current user interaction or across multiple interactions? The way you manage and store these ids will depend on that. For a relatively simple case, where you only want to replace the most recent message during a single interaction, storing the id in memory as part of the user's state is sufficient. This is straightforward if you're utilizing aiogram's finite state machine.

Let me provide an example of this. Suppose we have a simple interaction where the user selects an option from an inline keyboard and we update the message according to that choice. Here's what the code would look like:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils import executor

API_TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN' # Replace with your token

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

class ChoiceState(StatesGroup):
    waiting_for_choice = State()

@dp.message_handler(commands=['start'])
async def start(message: types.Message, state: FSMContext):
    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton("Option 1", callback_data="option1"))
    keyboard.add(types.InlineKeyboardButton("Option 2", callback_data="option2"))
    msg = await message.answer("Choose an option:", reply_markup=keyboard)
    await state.update_data(last_message_id=msg.message_id)
    await ChoiceState.waiting_for_choice.set()

@dp.callback_query_handler(lambda c: c.data.startswith('option'), state=ChoiceState.waiting_for_choice)
async def handle_choice(callback_query: types.CallbackQuery, state: FSMContext):
  user_data = await state.get_data()
  last_message_id = user_data.get('last_message_id')

  if last_message_id:
      try:
          await bot.delete_message(callback_query.message.chat.id, last_message_id)
      except Exception as e: # Adding robust error handling
          print(f"Error deleting message: {e}")

  if callback_query.data == "option1":
      new_msg = await bot.send_message(callback_query.message.chat.id, "You chose option 1!")
  elif callback_query.data == "option2":
      new_msg = await bot.send_message(callback_query.message.chat.id, "You chose option 2!")
  
  await state.update_data(last_message_id=new_msg.message_id)
  await callback_query.answer() # Acknowledge callback

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

```
In this example, each time a user interacts with the inline keyboard, the previous message is deleted before the new one is shown. Critically, error handling is included around `bot.delete_message()` since there are instances when this deletion might fail (e.g., if the message has already been deleted). It's vital to remember that Telegram can delete messages through other means, so never assume the deletion call will succeed.

Now, if we consider a slightly more complex scenario, say you need to keep track of messages across different states and update them repeatedly, storing just the last message id within the FSM might not cut it. You’ll need something more persistent. This is where a simple dictionary or a database lookup could be useful to hold the different messages that have been sent.

Let's assume you've chosen a straightforward dictionary keyed by user id. Here's how that might look:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.utils import executor
import asyncio

API_TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN' # Replace with your token

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

user_messages = {} # dictionary to store message ids for each user

@dp.message_handler(Command("start"))
async def start_command(message: types.Message):
    user_id = message.from_user.id
    
    if user_id in user_messages:
        for msg_id in user_messages[user_id]:
           try:
              await bot.delete_message(message.chat.id, msg_id)
           except Exception as e:
                print(f"Error deleting message: {e}")
        user_messages[user_id] = []
    
    new_message = await message.answer("First message for this user.")
    if user_id not in user_messages:
      user_messages[user_id] = [new_message.message_id]
    else:
      user_messages[user_id].append(new_message.message_id)

@dp.message_handler(Command("update"))
async def update_command(message: types.Message):
    user_id = message.from_user.id

    if user_id in user_messages:
        for msg_id in user_messages[user_id]:
           try:
                await bot.delete_message(message.chat.id, msg_id)
           except Exception as e:
                print(f"Error deleting message: {e}")
        user_messages[user_id] = []
    
    new_message = await message.answer("Updated message.")
    if user_id not in user_messages:
        user_messages[user_id] = [new_message.message_id]
    else:
        user_messages[user_id].append(new_message.message_id)



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

```
Here, we maintain a `user_messages` dictionary. Whenever a user sends a command, all the old messages are deleted and new ones are added to the dictionary. As you can imagine, this would become a bottleneck if your bot was dealing with a large number of users and message. For more complex bots, something like Redis or a dedicated database would become a necessity.

Finally, let's illustrate a slightly more advanced technique by using a more complex approach, utilizing an async Redis client for storing message ids. This is a more scalable solution suitable for larger deployments. You will have to install `aioredis`.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.utils import executor
import asyncio
import aioredis

API_TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN' # Replace with your token
REDIS_HOST = 'localhost'  # Or your Redis host
REDIS_PORT = 6379 # or your redis port


bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

async def get_redis_connection():
  return await aioredis.create_redis_pool(f'redis://{REDIS_HOST}:{REDIS_PORT}')


async def cleanup_messages(user_id, chat_id, redis_pool):
    async with redis_pool.get() as redis:
        message_ids = await redis.lrange(f"user:{user_id}:messages", 0, -1)
        if message_ids:
            for msg_id in message_ids:
                try:
                    await bot.delete_message(chat_id, int(msg_id))
                except Exception as e:
                   print(f"Error deleting message: {e}")
            await redis.delete(f"user:{user_id}:messages")


async def store_message(user_id, message_id, redis_pool):
    async with redis_pool.get() as redis:
        await redis.rpush(f"user:{user_id}:messages", message_id)

@dp.message_handler(Command("start"))
async def start_command(message: types.Message, redis_pool=None):
  if redis_pool is None:
    redis_pool = await get_redis_connection()
  user_id = message.from_user.id
  await cleanup_messages(user_id, message.chat.id, redis_pool)
  new_message = await message.answer("First message for this user.")
  await store_message(user_id, new_message.message_id, redis_pool)


@dp.message_handler(Command("update"))
async def update_command(message: types.Message, redis_pool = None):
  if redis_pool is None:
    redis_pool = await get_redis_connection()
  user_id = message.from_user.id
  await cleanup_messages(user_id, message.chat.id, redis_pool)
  new_message = await message.answer("Updated message.")
  await store_message(user_id, new_message.message_id, redis_pool)

async def main():
  redis_pool = await get_redis_connection()
  try:
    await dp.start_polling(skip_updates=True, redis_pool = redis_pool)
  finally:
    redis_pool.close()
    await redis_pool.wait_closed()

if __name__ == '__main__':
  asyncio.run(main())

```

In this example, message ids are stored in a Redis list, keyed by user id. This approach enhances scalability and is much more robust than the previous solutions, particularly when dealing with a high volume of users or if you need to preserve state across different bot instances.

For further reading, I highly recommend the official aiogram documentation, which details the intricacies of state management and message manipulation. You might also find the "Programming Telegram Bots" book by Huseyin Tugrul Buyukisik insightful, as it delves into practical aspects of bot development. Lastly, explore research papers and articles on effective state management techniques in distributed systems if you want a deeper understanding of scaling solutions, particularly those dealing with concurrent message updates. Remember to always handle errors gracefully, especially with `bot.delete_message()`, as messages might have already been removed by other actions. I hope this information will be beneficial for you.
