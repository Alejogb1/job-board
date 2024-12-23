---
title: "How do I use aiogram to delete past bot messages when sending a new one?"
date: "2024-12-23"
id: "how-do-i-use-aiogram-to-delete-past-bot-messages-when-sending-a-new-one"
---

,  From the perspective of someone who’s spent a fair amount of time building Telegram bots, handling message cleanup with aiogram is a task I've certainly encountered numerous times. It’s a common requirement – keeping the chat tidy, especially when the bot is frequently updating information or guiding users through multi-step processes. The key, as you might suspect, isn't just about sending messages but also effectively managing the lifespan of those preceding them.

Fundamentally, the process involves obtaining the `message_id` of the messages you intend to remove and then using the `delete_message` method within the aiogram library. This operation isn't inherently complex, but it demands careful handling, especially in asynchronous environments where race conditions and unexpected behaviors can quickly become an issue. Let’s break down how this is typically managed in practice, remembering that message ids are specific to a chat instance. A message id from one chat will not work in another, and this is a common mistake when you’re building bots across multiple chats.

Let me share a scenario from a past project; I was building a dynamic menu bot for managing a small e-commerce platform within Telegram. It required updates to product listings, and to keep things clean, old menu messages needed to be replaced by new ones, rather than stacked up. This meant each time the menu was updated, I had to delete the previous menu messages before sending the new one. The architecture was crucial; I chose to keep track of message ids in a small in-memory store, but the method you choose depends on the complexity and scale of your bot. A larger bot might rely on a database for such persistent storage of past messages, or even a redis cache.

Here's a basic principle: each message you send via the bot yields a `Message` object, which contains a crucial attribute—`message_id`. You must capture this id after sending a message if you intend to subsequently delete it. The `delete_message` method expects a `chat_id` and this `message_id` as input. Here’s a simple example:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio

TOKEN = "YOUR_BOT_TOKEN"

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


async def send_and_delete(chat_id: int):
    message_to_replace = await bot.send_message(chat_id, "Initial message.")
    await asyncio.sleep(1)  # Simulate some processing

    await bot.delete_message(chat_id, message_to_replace.message_id)
    await bot.send_message(chat_id, "New message, replacing the old one.")

@dp.message_handler(commands=['start'])
async def on_start(message: types.Message):
   await send_and_delete(message.chat.id)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

This snippet demonstrates the essential flow: send a message, capture its id, perform an action, then delete the message using that captured id before sending a replacement. This is functional for a simple one-time deletion, but it lacks persistence and would only delete the initial message once the command has been issued.

Now, lets explore a more practical example. Let’s assume we want to keep the last n messages from a specific handler, and delete older ones:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio
from typing import Dict, List

TOKEN = "YOUR_BOT_TOKEN"
MESSAGE_HISTORY_LIMIT = 3

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

message_history: Dict[int, List[int]] = {}  # {chat_id: [message_id, message_id, ...]}


async def manage_message_history(chat_id: int, new_message_id: int):
    if chat_id not in message_history:
        message_history[chat_id] = []

    message_history[chat_id].append(new_message_id)

    if len(message_history[chat_id]) > MESSAGE_HISTORY_LIMIT:
      message_to_delete = message_history[chat_id].pop(0) # LIFO
      await bot.delete_message(chat_id, message_to_delete)

@dp.message_handler(commands=['test'])
async def test_handler(message: types.Message):
    new_message = await bot.send_message(message.chat.id, "A new message in the stream!")
    await manage_message_history(message.chat.id, new_message.message_id)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

Here, I've introduced a basic, in-memory history management system. We track `message_id`s per `chat_id`. Once the history exceeds the defined limit, the oldest message is deleted. This is far more adaptable, useful when you have multiple handler outputs and you want to have some limit to the chat history your bot is controlling.

For more complex workflows, where you might need persistent storage or more granular control over message deletion (e.g., deleting messages after a timeout), you would likely want to use a database or other external storage. In such cases, you would store the message ids alongside a timestamp or other relevant metadata. I've found that using a structured database is invaluable when implementing more intricate interaction patterns.

As a final example, consider a situation where you need to edit an existing message rather than deleting and replacing it. Telegram bots can do that as well! If you need to maintain context and a persistent message, it is often a better choice to edit. Here is a simple example, where each call of the handler will edit the same original message:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio

TOKEN = "YOUR_BOT_TOKEN"

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

tracked_message = None

@dp.message_handler(commands=['track'])
async def track_handler(message: types.Message):
  global tracked_message
  if tracked_message is None:
    tracked_message = await bot.send_message(message.chat.id, "Message being tracked! This message will be updated.")
  else:
    await bot.edit_message_text(text="Message tracked updated!", chat_id=tracked_message.chat.id, message_id=tracked_message.message_id)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

This example introduces a global variable to store the `message` object, and we use the edit functionality instead of deleting the message.

Some essential resources to deepen your understanding of aiogram and managing message lifecycles include the official aiogram documentation (which is excellent), and specifically the aiogram.api.methods. There are also some very useful examples in the aiogram examples github repository. Furthermore, I'd suggest studying asynchronous programming in Python more generally. Specifically, you may want to review 'Concurrency with asyncio' from the official python docs.

In practical use, managing message deletion is far more complex than the simple examples given here. Proper logging, error handling (what if you try to delete a message that no longer exists or has been deleted by someone else?), and careful management of message id storage become vital considerations for a production bot. This is especially true at scale, where multiple users can interact with a bot simultaneously. The goal, of course, is a clean, responsive, and robust Telegram bot that enhances user experience, and not one that generates a deluge of messages and confuses the users. In summary, understanding message ids, using `delete_message` (or `edit_message`) effectively, and managing message histories is paramount for a well-behaved bot. Remember to choose your strategy carefully, and choose one that scales well for your application.
