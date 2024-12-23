---
title: "How can I delete past bot messages in aiogram when sending new ones?"
date: "2024-12-23"
id: "how-can-i-delete-past-bot-messages-in-aiogram-when-sending-new-ones"
---

Okay, let's tackle this. I’ve actually run into this exact scenario a few times back when I was setting up a Telegram bot for monitoring server metrics. It’s a pretty common challenge, and getting it smooth requires understanding how aiogram handles message updates. The core issue, as you're finding, isn't simply that aiogram lacks a `delete_previous_messages` function, but rather that the bot needs to actively manage message ids and their deletion, as Telegram doesn't inherently "replace" a message; it's a new message altogether.

The main strategy here revolves around storing the message id of the previous bot message and then using `bot.delete_message` with that id before sending the new one. This involves a few key steps: firstly, sending the initial message, retaining its id, secondly, storing this id somewhere persistent (like a dictionary in memory, a database, or even a simple file). Lastly, when a new message needs to be sent, retrieve the stored id, delete the message and send the new one, finally updating the stored id.

Let’s consider a basic in-memory approach initially, since this is great for smaller bots and quick experimentation. We can hold a simple dictionary mapping chat ids to message ids:

```python
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart

# Replace with your bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

previous_message_ids = {}  # In-memory storage for message ids

async def delete_previous_message(chat_id: int):
    if chat_id in previous_message_ids:
        try:
            await bot.delete_message(chat_id, previous_message_ids[chat_id])
        except Exception as e:
            print(f"Error deleting previous message: {e}")
        del previous_message_ids[chat_id]

async def send_message_and_update(chat_id: int, text: str):
    message = await bot.send_message(chat_id, text)
    previous_message_ids[chat_id] = message.message_id


@dp.message(CommandStart())
async def start_handler(message: types.Message):
    chat_id = message.chat.id
    await delete_previous_message(chat_id)
    await send_message_and_update(chat_id, "Initial message!")
    await asyncio.sleep(2) # Simulate some processing
    await delete_previous_message(chat_id)
    await send_message_and_update(chat_id, "Second message replaces the first!")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

```

In this first example, the `previous_message_ids` dictionary acts as a temporary store. Each chat id is mapped to its last sent message id. When a new message needs to be sent, `delete_previous_message` is called to attempt deletion if a previous id exists, then `send_message_and_update` sends the new message and updates the stored id. Notice the error handling inside `delete_previous_message`; it's crucial because message deletion can fail for several reasons, such as the message being too old to be deleted, or simply if the id is already gone.

However, relying solely on an in-memory dictionary isn't practical for any serious use case; it’s volatile and won't persist between bot restarts. We need something more persistent, such as a database or even simple file storage. Here’s an example using a basic json file to store the data:

```python
import asyncio
import json
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart

# Replace with your bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

MESSAGE_ID_FILE = "message_ids.json"

def load_previous_message_ids():
    try:
        with open(MESSAGE_ID_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_previous_message_ids(data):
     with open(MESSAGE_ID_FILE, 'w') as f:
        json.dump(data, f)

previous_message_ids = load_previous_message_ids()  # Load ids from file


async def delete_previous_message(chat_id: int):
    global previous_message_ids # access global state
    if str(chat_id) in previous_message_ids: # Convert chat_id to string for json dict
        try:
           await bot.delete_message(chat_id, previous_message_ids[str(chat_id)])
        except Exception as e:
            print(f"Error deleting previous message: {e}")
        del previous_message_ids[str(chat_id)]
        save_previous_message_ids(previous_message_ids) # save updated state


async def send_message_and_update(chat_id: int, text: str):
    global previous_message_ids  # access global state
    message = await bot.send_message(chat_id, text)
    previous_message_ids[str(chat_id)] = message.message_id # Convert chat_id to string for json dict
    save_previous_message_ids(previous_message_ids) # save updated state


@dp.message(CommandStart())
async def start_handler(message: types.Message):
    chat_id = message.chat.id
    await delete_previous_message(chat_id)
    await send_message_and_update(chat_id, "Initial message!")
    await asyncio.sleep(2) # Simulate some processing
    await delete_previous_message(chat_id)
    await send_message_and_update(chat_id, "Second message replaces the first!")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
```

Here, instead of storing the data in-memory we store it in a `message_ids.json` file. We load the data when the bot starts, save it after each update. This persists message ids across bot restarts. Note we use `str(chat_id)` as keys because json dictionaries require string keys. While basic, this example serves to illustrate that persistent storage is necessary for real-world usage.

Finally, for a more robust setup, consider using a database. Here’s an example using SQLite, which is simple enough to illustrate the point without needing a separate server:

```python
import asyncio
import sqlite3
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart

# Replace with your bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

DATABASE_FILE = "bot_data.db"

def create_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS message_ids (
        chat_id INTEGER PRIMARY KEY,
        message_id INTEGER
    )
    ''')
    conn.commit()
    return conn

def get_previous_message_id(conn, chat_id: int):
    cursor = conn.cursor()
    cursor.execute("SELECT message_id FROM message_ids WHERE chat_id = ?", (chat_id,))
    result = cursor.fetchone()
    return result[0] if result else None

def set_previous_message_id(conn, chat_id: int, message_id: int):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO message_ids (chat_id, message_id) VALUES (?, ?)", (chat_id, message_id))
    conn.commit()


async def delete_previous_message(chat_id: int, conn):
    message_id = get_previous_message_id(conn, chat_id)
    if message_id:
        try:
           await bot.delete_message(chat_id, message_id)
        except Exception as e:
            print(f"Error deleting previous message: {e}")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM message_ids WHERE chat_id = ?", (chat_id,))
        conn.commit()

async def send_message_and_update(chat_id: int, text: str, conn):
    message = await bot.send_message(chat_id, text)
    set_previous_message_id(conn, chat_id, message.message_id)


@dp.message(CommandStart())
async def start_handler(message: types.Message):
    chat_id = message.chat.id
    conn = create_connection()
    await delete_previous_message(chat_id, conn)
    await send_message_and_update(chat_id, "Initial message!", conn)
    await asyncio.sleep(2) # Simulate some processing
    await delete_previous_message(chat_id, conn)
    await send_message_and_update(chat_id, "Second message replaces the first!", conn)
    conn.close()

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
```

This final example introduces an SQLite database to store message id information. It’s much more durable and suitable for production bots.

For further reading to solidify your grasp on these concepts, I highly recommend exploring the aiogram documentation directly. Specifically, pay attention to the `aiogram.Bot.delete_message` and `aiogram.Bot.send_message` methods. Also, reviewing chapters on database interactions within the Python standard library documentation regarding `sqlite3`, and if you're using a different database, the associated library documentation will be extremely valuable. Finally, taking some time to investigate Telegram's bot API documentation regarding message deletion behaviour will round out your understanding. These resources will deepen your understanding of bot mechanics, making your future bot projects much more manageable. Remember that while managing message ids is crucial for this particular task, there are different message updating strategies within Telegram itself that might reduce your needs to delete and resend, should you start encountering API call limits.
