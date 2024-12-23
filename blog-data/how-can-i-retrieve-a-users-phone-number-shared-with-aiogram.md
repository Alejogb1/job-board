---
title: "How can I retrieve a user's phone number shared with AIOgram?"
date: "2024-12-23"
id: "how-can-i-retrieve-a-users-phone-number-shared-with-aiogram"
---

Alright, let's tackle this. The question of retrieving a user's phone number via AIOgram, while seemingly straightforward, can get a little nuanced depending on *how* that number was shared. It’s not simply about pulling it from some readily available attribute; privacy and user consent are paramount, and AIOgram respects this. Let me walk you through what I’ve seen and how I've handled it in past projects.

First, understand that AIOgram, unlike some older frameworks, doesn't automatically give you a user's phone number the moment they interact with your bot. You actually need the user to *explicitly* share it. This is done using Telegram’s contact request mechanism, generally through a custom keyboard. Let's break down the most common way this works, then we can look at some code examples.

Usually, you'll set up a `ReplyKeyboardMarkup` with a button requesting the user’s phone number. When the user presses that button, Telegram triggers a special `message` update that contains the user's `contact` data. This data structure is where you find the phone number. The trick is correctly detecting that special update and accessing the relevant fields. The `contact` field is part of the message itself. It’s *not* a user attribute, directly accessible via `user.phone_number` (for instance). This distinction is crucial; I’ve seen developers stumble on this more than once in code reviews.

Now, for some code. Imagine we’re building a simple bot that asks for a user’s contact details. Here's the initial setup, showing how to present the button to the user:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio

TOKEN = "YOUR_BOT_TOKEN" # Replace with your actual bot token

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    share_phone_button = types.KeyboardButton(text="Share phone number", request_contact=True)
    keyboard.add(share_phone_button)
    await message.answer("Please share your phone number to proceed.", reply_markup=keyboard)


@dp.message_handler(content_types=types.ContentTypes.CONTACT)
async def contact_handler(message: types.Message):
    if message.contact:
        phone_number = message.contact.phone_number
        user_id = message.from_user.id
        await message.answer(f"Thanks! Your phone number ({phone_number}) has been received and is associated with user id {user_id}.")
    else:
         await message.answer("Hmm, it seems there was an issue getting the contact.")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)


```

In this snippet, we first create a `ReplyKeyboardMarkup` with a button that uses the `request_contact=True` parameter. This tells Telegram to trigger a contact share. Notice the dedicated `message_handler` decorated with `@dp.message_handler(content_types=types.ContentTypes.CONTACT)`. This handler *only* activates when a message containing a contact is received. Inside, we check if `message.contact` is present before attempting to access the `phone_number` attribute. This small safety check prevents potential `NoneType` errors.

Now, let’s say you need to handle a scenario where the user might try to bypass the button and directly send you their number as a text. In that case, you should handle this gracefully by prompting the user to utilize the button again. The following example demonstrates this:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio

TOKEN = "YOUR_BOT_TOKEN" # Replace with your actual bot token

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    share_phone_button = types.KeyboardButton(text="Share phone number", request_contact=True)
    keyboard.add(share_phone_button)
    await message.answer("Please share your phone number using the button below.", reply_markup=keyboard)


@dp.message_handler(content_types=types.ContentTypes.CONTACT)
async def contact_handler(message: types.Message):
    if message.contact:
        phone_number = message.contact.phone_number
        user_id = message.from_user.id
        await message.answer(f"Thanks! Your phone number ({phone_number}) has been received and is associated with user id {user_id}.")
    else:
        await message.answer("Hmm, it seems there was an issue getting the contact.")


@dp.message_handler(lambda message: message.content_type == types.ContentType.TEXT)
async def handle_text(message: types.Message):
  keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
  share_phone_button = types.KeyboardButton(text="Share phone number", request_contact=True)
  keyboard.add(share_phone_button)
  await message.answer("Please use the button to share your phone number. Your text has not been processed.", reply_markup = keyboard)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

Here, the new handler using a lambda function intercepts any text-based messages that do not come through as a contact update. This handler re-prompts the user with the phone number share button, ensuring a consistent flow for the contact sharing functionality. This pattern is key for creating a user-friendly and reliable process.

Finally, sometimes you may need to store the phone number or perform some further actions with it. It is usually beneficial to organize these tasks into their own independent function that can be tested and maintained separately. Here's an updated version incorporating that:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import asyncio
# Dummy storage, replace with database
phone_storage = {}

TOKEN = "YOUR_BOT_TOKEN"  # Replace with your actual bot token

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


async def process_contact(user_id: int, phone_number: str):
    phone_storage[user_id] = phone_number
    print(f"User {user_id} stored with number: {phone_number}")
    # In a real app, store to a database, etc.

@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    share_phone_button = types.KeyboardButton(text="Share phone number", request_contact=True)
    keyboard.add(share_phone_button)
    await message.answer("Please share your phone number to proceed.", reply_markup=keyboard)


@dp.message_handler(content_types=types.ContentTypes.CONTACT)
async def contact_handler(message: types.Message):
    if message.contact:
        phone_number = message.contact.phone_number
        user_id = message.from_user.id
        await process_contact(user_id, phone_number)
        await message.answer(f"Thanks! Your phone number has been received and processed.")

    else:
        await message.answer("Hmm, it seems there was an issue getting the contact.")

@dp.message_handler(lambda message: message.content_type == types.ContentType.TEXT)
async def handle_text(message: types.Message):
  keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
  share_phone_button = types.KeyboardButton(text="Share phone number", request_contact=True)
  keyboard.add(share_phone_button)
  await message.answer("Please use the button to share your phone number. Your text has not been processed.", reply_markup = keyboard)



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

This code uses a simple dictionary as a placeholder for data storage, but in real application scenarios you would likely use a database, and make the storage of the phone number and user ID transactional.

A key takeaway here is the need to handle different types of input and the separation of concerns using a dedicated processing function. The `process_contact` function is where you would include logic for storing the contact details or perform further business tasks.

For further reading, I would strongly suggest getting a copy of *Programming Telegram Bots* by Syed Umar Anis, it's very thorough and covers bot interactions effectively. For more in-depth Telegram API specifics, reviewing the official Telegram Bot API documentation is essential. Also, check out the AIOgram documentation itself. Knowing both the core API and the specific library implementation is crucial for more complex scenarios.

Remember that these are fundamental examples, but the principles apply to more sophisticated workflows. Always prioritize user privacy and follow Telegram's guidelines for handling sensitive user data. This process should provide a solid basis for retrieving user phone numbers within AIOgram.
