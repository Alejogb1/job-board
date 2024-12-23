---
title: "Can Aiogram delete Telegram messages with links?"
date: "2024-12-23"
id: "can-aiogram-delete-telegram-messages-with-links"
---

Alright, let's talk about deleting Telegram messages with links using Aiogram. It’s a topic I’ve definitely had my hands dirty with in past projects, specifically around moderation bots. The short answer is yes, you absolutely can delete messages containing links using Aiogram, but there's more to it than just a single command. It requires careful handling of message data and a clear understanding of how Telegram's API interacts with bots.

When we break it down, we’re essentially looking at a multi-stage process. First, the bot needs to receive messages. Second, it needs to inspect the content for links. Third, if a link is found, it needs to issue a delete command. Let's explore each of these stages, including potential pitfalls and practical code examples that I’ve found useful over the years.

The first step is straightforward with Aiogram. You'll usually be listening for `Message` events using a dispatcher. This is the core of any bot’s functionality when it comes to receiving user input. We don't need to dive into the basic setup since I'm assuming you're past that point. The key is how you'll be interpreting that incoming message. Here's a snippet showing you how to grab the text:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
import re
import asyncio


TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler()
async def process_message(message: types.Message):
    message_text = message.text
    if message_text:  # Ensuring it's not a media message without text
        if re.search(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', message_text):
            await bot.delete_message(message.chat.id, message.message_id)
            print(f"Deleted message {message.message_id} from {message.chat.id}")
    else:
        print("Received a message without text, nothing to check.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
```

This code focuses on the core functionality. We’re extracting the message's text and using a regular expression to check for any form of http or https links. The regex itself (`r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'`) may look intimidating, but it's a standard pattern for catching most URL structures. Note the `re.search` function – it efficiently determines if our desired pattern is present within the string. If found, we invoke `bot.delete_message()`, passing the chat ID and message ID to remove the offending message. Logging, such as the print statement, is crucial in debugging these systems.

Now, there are several improvements we can add to this. What if the link is embedded within a more substantial text message? What if a user sends a message with a link as part of a larger conversational element? For that, we need to expand our approach. This modified code block handles such cases, incorporating some practical experience from my past moderating tasks:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, MessageEntity
import re
import asyncio

TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler()
async def process_message(message: types.Message):
    if message.text:
        if message.entities:
           for entity in message.entities:
                if entity.type in ['url', 'text_link']:
                    await bot.delete_message(message.chat.id, message.message_id)
                    print(f"Deleted message {message.message_id} from {message.chat.id} due to URL entity.")
                    return # Exit after deleting a single message.
        else:
            if re.search(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', message.text):
                await bot.delete_message(message.chat.id, message.message_id)
                print(f"Deleted message {message.message_id} from {message.chat.id} due to regex match.")

    elif message.caption:
        if message.caption_entities:
           for entity in message.caption_entities:
                if entity.type in ['url', 'text_link']:
                    await bot.delete_message(message.chat.id, message.message_id)
                    print(f"Deleted message {message.message_id} from {message.chat.id} due to URL entity in caption.")
                    return
        else:
            if re.search(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', message.caption):
                 await bot.delete_message(message.chat.id, message.message_id)
                 print(f"Deleted message {message.message_id} from {message.chat.id} due to regex match in caption.")

    else:
        print("Received a message without text or caption, nothing to check.")


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
```

This version is more robust. It checks `message.entities` which is the preferred way of checking for urls and other formatted text as telegram recognizes these already, instead of purely relying on regex. Telegram's API will mark entities as urls or text links if they are in a message. The code also now accounts for the fact that messages can be media with captions, so the same check is applied to `message.caption` and `message.caption_entities` fields. If a single URL is found we delete the message and return; this is a basic precaution that prevents further processing of an already deleted message and simplifies the code structure. The 'return' is a small touch, but it avoids unnecessary cycles.

One more step, let's address the possibility of users abusing the bot by rapidly sending messages with links to trigger deletion events, which could impact performance. A rate limit can be implemented to mitigate this issue. Here's how you can adapt the code:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, MessageEntity
import re
import asyncio
from collections import defaultdict

TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

RATE_LIMIT_WINDOW = 5  # Seconds
RATE_LIMIT_MAX_MESSAGES = 2
user_message_counts = defaultdict(lambda: {'count': 0, 'last_time': 0})

@dp.message_handler()
async def process_message(message: types.Message):

    user_id = message.from_user.id
    current_time = asyncio.get_event_loop().time()

    if current_time - user_message_counts[user_id]['last_time'] > RATE_LIMIT_WINDOW:
          user_message_counts[user_id]['count'] = 0

    user_message_counts[user_id]['count'] +=1
    user_message_counts[user_id]['last_time'] = current_time

    if user_message_counts[user_id]['count'] > RATE_LIMIT_MAX_MESSAGES:
      print(f"User {user_id} has reached rate limit.")
      return # User has triggered the rate limit, don't process this message

    if message.text:
        if message.entities:
           for entity in message.entities:
                if entity.type in ['url', 'text_link']:
                    await bot.delete_message(message.chat.id, message.message_id)
                    print(f"Deleted message {message.message_id} from {message.chat.id} due to URL entity.")
                    return
        else:
            if re.search(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', message.text):
                await bot.delete_message(message.chat.id, message.message_id)
                print(f"Deleted message {message.message_id} from {message.chat.id} due to regex match.")
                return

    elif message.caption:
        if message.caption_entities:
           for entity in message.caption_entities:
                if entity.type in ['url', 'text_link']:
                    await bot.delete_message(message.chat.id, message.message_id)
                    print(f"Deleted message {message.message_id} from {message.chat.id} due to URL entity in caption.")
                    return
        else:
            if re.search(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', message.caption):
                await bot.delete_message(message.chat.id, message.message_id)
                print(f"Deleted message {message.message_id} from {message.chat.id} due to regex match in caption.")
                return
    else:
        print("Received a message without text or caption, nothing to check.")


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
```

Here, we track the number of messages each user sends within a `RATE_LIMIT_WINDOW`. If a user surpasses `RATE_LIMIT_MAX_MESSAGES` in that timeframe, we immediately cease further message processing for that user. This implementation is a straightforward in-memory system; in a production environment, you’d typically employ something like Redis for more robust rate limiting. It is imperative to implement proper rate limits to ensure the bot doesn't cause any issues due to being used maliciously.

For further study, I recommend starting with the official Telegram Bot API documentation, it’s essential for understanding the underlying mechanisms. For a deeper dive into Aiogram specifically, review their official documentation. Additionally, the book "Fluent Python" by Luciano Ramalho is an excellent resource to better your python skills, including the regex component used, if needed. These are the sources that I generally find useful, and would likely serve you well also. Remember that bot creation requires incremental development and constant refining.
