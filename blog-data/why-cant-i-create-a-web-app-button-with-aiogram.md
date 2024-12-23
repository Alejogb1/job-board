---
title: "Why can't I create a web app button with aiogram?"
date: "2024-12-23"
id: "why-cant-i-create-a-web-app-button-with-aiogram"
---

, let's tackle this one. I've spent a fair bit of time elbow-deep in both web development and Telegram bot creation, so I’ve run into similar scenarios myself. The short answer, without getting too bogged down, is that aiogram primarily deals with interacting with the Telegram Bot API. This API offers methods for sending messages, images, and various types of keyboard interfaces *within* Telegram, but it doesn't directly provide functionality for rendering or interacting with standard web components like buttons that you'd find on a webpage. Let me explain in a little more depth and provide some concrete examples from past projects, which should help illustrate where things get tricky.

The core issue here stems from fundamentally different architectures. A web app, typically built with HTML, CSS, and Javascript, operates within a browser environment. It renders visually, responds to user input on the client-side, and then communicates with a server. aiogram, on the other hand, works on the server-side and interacts with Telegram's cloud. It sends and receives formatted messages that Telegram's clients (mobile app, desktop client) then display. Think of it like this: aiogram sends instructions to Telegram on what to display; it doesn't control the rendering engine of a browser.

Now, the Telegram Bot API does offer *inline keyboards* and *reply keyboards*. These are visual buttons, yes, but they are not the same as HTML buttons. They trigger actions within the Telegram ecosystem rather than manipulating a web page. An inline keyboard, for instance, might initiate a callback query that your bot can then process. A reply keyboard, similarly, suggests a set of text prompts which the user can send as a message. Let's look at a quick example using aiogram to create an inline keyboard:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor

# Replace with your actual bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    button1 = types.InlineKeyboardButton(text="Button 1", callback_data="button_1_pressed")
    button2 = types.InlineKeyboardButton(text="Button 2", callback_data="button_2_pressed")
    keyboard.add(button1, button2)
    await message.reply("Choose an option:", reply_markup=keyboard)


@dp.callback_query_handler(lambda c: c.data == 'button_1_pressed')
async def process_button1(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id, text="Button 1 was pressed!")

@dp.callback_query_handler(lambda c: c.data == 'button_2_pressed')
async def process_button2(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id, text="Button 2 was pressed!")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

```

In this code, I’m creating an inline keyboard with two buttons. When the user presses a button, the Telegram client sends a 'callback query' to my bot, identified by a unique `callback_data` value. The `@dp.callback_query_handler` decorator then picks this up and responds. Notice this isn't triggering something on a webpage, it's triggering an action *within* the bot.

Years ago, I recall attempting to embed a link in a similar bot interface, and I stumbled upon the same limitation: you can make an inline button that *opens* a web page, but it's not an embedded button *on* the bot. It's a link that takes the user out of Telegram. Let’s illustrate that.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor

# Replace with your actual bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['link'])
async def link_command(message: types.Message):
    keyboard = types.InlineKeyboardMarkup()
    web_button = types.InlineKeyboardButton(text="Open Website", url="https://www.example.com")
    keyboard.add(web_button)
    await message.reply("Visit our website:", reply_markup=keyboard)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

```

This example creates a button that, when clicked, opens a URL. This is very handy for pointing users to a web resource, but it doesn't place an interactive web button inside the Telegram client.

So, to truly create a web app button experience, you essentially need to leverage *Telegram Web Apps*. This approach uses a special kind of bot integration to host a full web application within Telegram’s client. Here, aiogram acts as the orchestrator, telling Telegram to open a web app hosted elsewhere. When a user interacts with the app (including its web buttons), the data from the user interaction then communicates back to the bot. The bot, running via aiogram, can then respond appropriately to the app’s events.

To illustrate this, a minimal (but still functional) example is more intricate. It involves having both an aiogram bot and a basic web server. Here's a simplified representation of the bot side, indicating that the bot sends a web app request:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import WebAppInfo

# Replace with your actual bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# Replace with the actual URL of your web app
WEB_APP_URL = "https://your-web-app-domain.com"


@dp.message_handler(commands=['webapp'])
async def webapp_command(message: types.Message):
    keyboard = types.InlineKeyboardMarkup()
    web_app_button = types.InlineKeyboardButton(text="Open Web App", web_app=WebAppInfo(url=WEB_APP_URL))
    keyboard.add(web_app_button)
    await message.reply("Open the web app:", reply_markup=keyboard)


@dp.message_handler(content_types=types.ContentTypes.WEB_APP_DATA)
async def process_web_app_data(message: types.Message):
    # Here we receive data from the web app after the user interacts
    web_app_data = message.web_app_data.data
    await message.answer(f"Data received from web app: {web_app_data}")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```
This code snippet does not show the web app itself, which is a completely separate application deployed on a server (e.g., your `https://your-web-app-domain.com`). This `WEB_APP_URL` points to the location of a regular web app that you host. Crucially, the web app is responsible for generating and sending the web data when the user interacts with it and triggers a bot message using javascript. The `process_web_app_data` handles the data sent back to the bot from the web app. The crucial difference with this example is that it bridges bot interface with web interface via Telegram’s web app functionality.

If you’re keen to dive deeper, I highly recommend consulting the official Telegram Bot API documentation, specifically the sections on inline and reply keyboards and, more importantly, Telegram Web Apps. Also, I found the book "Programming Telegram Bots" by Asadbek Nematov to be a very good resource when I was first exploring bot development. That offers real-world use case examples that go beyond the basic API documentation. In addition, research the implementation details of handling data flow in Telegram Web Apps.

In conclusion, while you cannot create a standard webpage button directly with aiogram, it is very possible to simulate the functionality through inline/reply keyboards or by implementing Telegram Web Apps, which allows for an embedded browser experience within the Telegram interface and a richer interaction between bots and web applications. Choosing the appropriate path depends largely on your needs and the extent of the UI you envision.
