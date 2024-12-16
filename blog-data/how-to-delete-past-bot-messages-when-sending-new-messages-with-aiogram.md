---
title: "How to delete past bot messages when sending new messages with aiogram?"
date: "2024-12-16"
id: "how-to-delete-past-bot-messages-when-sending-new-messages-with-aiogram"
---

Okay, let's unpack this one. Handling bot message history, especially when you’re aiming for a cleaner user experience, is a common challenge. I've certainly been there, wrestling with aiogram's asynchronous nature and Telegram's API quirks when trying to keep bot chats streamlined. Back in my early days with a chatbot project for a local community group – before we even thought about sophisticated NLP – we ran into exactly this problem. The bot was just spitting out updates, one after another, and it quickly became a mess. We needed a way to replace, not just add to, the existing message thread.

The core issue here isn't a simple “delete all past messages” function; Telegram's API doesn't work that way. Instead, the most effective and practical approach is to identify *specific* past messages that you want to remove and then delete them, either individually or in controlled sequences. This hinges on storing message IDs, so you can refer back to them later. Let's break down how this typically works in aiogram and then I’ll share a few code examples.

First, the fundamental principle: When you send a message with `bot.send_message()`, aiogram returns a `Message` object. This object includes a `message_id` property, which is crucial for subsequent operations like editing or deletion. The strategy here is always to save this `message_id` when you send something that you might want to replace later. It could be in memory (like a dictionary, suitable for short-lived sessions), or ideally in a database if you need persistence across bot restarts.

Now, let's get into some practical examples, starting with a basic scenario.

**Example 1: Simple Message Replacement (In-Memory)**

This example uses a dictionary to keep track of the last message sent. It's suitable for simple use cases but remember this data will be lost when your script restarts.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio

API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN' # Replace with your actual token
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

last_message_id = {}  # Dictionary to store the last sent message ID

async def send_and_replace(chat_id, text):
    global last_message_id

    if chat_id in last_message_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=last_message_id[chat_id])
        except Exception as e:
            print(f"Error deleting message: {e}")
    
    sent_message = await bot.send_message(chat_id, text)
    last_message_id[chat_id] = sent_message.message_id

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await send_and_replace(message.chat.id, "Initial Message")


@dp.message_handler(commands=['update'])
async def update(message: types.Message):
    await send_and_replace(message.chat.id, "Updated Message")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

In this code, `last_message_id` stores message IDs per chat. The `send_and_replace` function checks if a previous message exists for the current `chat_id`, deletes it, and then sends the new one, storing the new ID. This demonstrates the core principle of deleting by message ID.

**Example 2: Multiple Messages Deletion with a Database**

For a more robust solution, especially in a persistent bot, you’ll want a database. Here’s an example using a simple text file as a stand-in (but ideally, use something like sqlite, postgresql, or another proper database):

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio
import json

API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN' # Replace with your actual token
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

MESSAGE_DB_FILE = 'message_db.json'

def load_message_db():
    try:
        with open(MESSAGE_DB_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_message_db(db):
    with open(MESSAGE_DB_FILE, 'w') as f:
        json.dump(db, f)

async def send_and_replace_multiple(chat_id, text):
    message_db = load_message_db()

    if str(chat_id) in message_db:
        for message_id in message_db[str(chat_id)]:
            try:
                await bot.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception as e:
                print(f"Error deleting message: {e}")

    sent_message = await bot.send_message(chat_id, text)
    message_db[str(chat_id)] = [sent_message.message_id]  # Store only the most recent message id

    save_message_db(message_db)


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await send_and_replace_multiple(message.chat.id, "Initial Message Series")


@dp.message_handler(commands=['update'])
async def update(message: types.Message):
     await send_and_replace_multiple(message.chat.id, "Updated series")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

Here, we store message IDs in a json file (again, use a database in production!). `load_message_db()` and `save_message_db()` handle the file i/o. The `send_and_replace_multiple` function fetches previous IDs, deletes them, sends the new message and stores *only* the most recent message. This approach ensures data persistence between bot restarts and is more suitable for real-world applications. Notice how the structure allows storing multiple message ids for more complex use cases by not overwriting but rather appending to the list

**Example 3: Deleting specific past message based on message content.**

Let's say you only want to delete messages that match a certain pattern.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio
import json

API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN' # Replace with your actual token
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

MESSAGE_DB_FILE = 'message_db.json'

def load_message_db():
    try:
        with open(MESSAGE_DB_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_message_db(db):
    with open(MESSAGE_DB_FILE, 'w') as f:
        json.dump(db, f)

async def send_and_replace_specific(chat_id, text, pattern_to_delete):
    message_db = load_message_db()
    if str(chat_id) in message_db:
        for message_id, message_text in message_db[str(chat_id)]:
          if pattern_to_delete in message_text:
            try:
               await bot.delete_message(chat_id=chat_id, message_id=message_id)
               message_db[str(chat_id)].remove((message_id, message_text))
            except Exception as e:
                print(f"Error deleting message: {e}")

    sent_message = await bot.send_message(chat_id, text)
    if str(chat_id) not in message_db:
       message_db[str(chat_id)] = []
    message_db[str(chat_id)].append((sent_message.message_id, text))
    save_message_db(message_db)


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
  await send_and_replace_specific(message.chat.id, "Initial Message with pattern old", "old")

@dp.message_handler(commands=['update'])
async def update(message: types.Message):
    await send_and_replace_specific(message.chat.id, "updated message without pattern", "old")
    await send_and_replace_specific(message.chat.id, "second message with pattern old", "old")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

This example is slightly more complex because we store tuples of the message id and the message content and only delete messages that match a provided pattern. This approach has a bit more overhead in terms of storage but gives more control about which message are deleted.

**Key Takeaways:**

*   **Message IDs are fundamental:** You must store the `message_id` returned by `bot.send_message()` to manage messages programmatically.
*   **Storage is crucial:** Decide whether in-memory is sufficient, or if a database (even a simple file-based one) is necessary for persistence.
*   **Error Handling is essential:** Always wrap deletion attempts in a try-except block to avoid abrupt bot terminations if a message has already been deleted or is not found.
*   **Telegram API Limitations:** Be mindful of the rate limits and other API constraints to avoid hitting issues when deleting messages.

For further reading, I highly recommend diving deep into:

*   **The official aiogram documentation:** This is the primary resource for anything aiogram-related. Pay close attention to the `types.Message` object and the `Bot` class's methods.
*   **The Telegram Bot API documentation:** Knowing the underlying API is critical. Understand how messages are represented and the limitations for modifications and deletion.
*   **Database fundamentals:** If you're unfamiliar, read up on basic database concepts. Start with something simple like sqlite if you are not familiar with SQL databases.
*   **"Designing Data-Intensive Applications" by Martin Kleppmann**: While not directly specific to aiogram or Telegram, this book offers very important concepts on how to store and retrieve data reliably in modern applications, relevant for choosing and implementing storage solutions that will scale well with a growing bot.

Dealing with message deletion in aiogram involves a few considerations, but with these principles in mind, you can effectively manage your bot's chat history. Remember to always think about user experience and strive for clean, informative interactions.
