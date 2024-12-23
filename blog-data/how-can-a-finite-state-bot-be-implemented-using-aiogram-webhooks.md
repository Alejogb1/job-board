---
title: "How can a finite state bot be implemented using aiogram webhooks?"
date: "2024-12-23"
id: "how-can-a-finite-state-bot-be-implemented-using-aiogram-webhooks"
---

Alright, let's unpack this. Implementing a finite state machine (fsm) for a Telegram bot using aiogram webhooks is a fairly common requirement, and over the years I’ve seen it implemented… well, let’s just say there’s a spectrum of approaches out there. I’ve certainly had my share of learning experiences – that one project involving a multi-level onboarding process and a botched state transition still makes me shudder a little. But enough reminiscing; let's focus on a clean and robust implementation.

The core idea is to manage the bot's conversational flow using defined states. Each state represents a particular point in the conversation, dictating which handlers are active and what kind of input the bot expects. Think of it like a flowchart, where each node is a state, and the arrows are the transitions triggered by user messages or other events. Aiogram, with its built-in support for state management, makes this relatively straightforward, especially when coupled with webhooks.

First, let's consider the fundamental structure. We'll leverage `aiogram.dispatcher.FSMContext` to manage state. This is crucial because webhooks, unlike long polling, aren’t constantly running and therefore can't rely on simple in-memory variables to remember the context of a conversation. `FSMContext` provides a persistent storage mechanism, typically backed by a database or an in-memory storage suitable for smaller deployments. For larger projects, databases like postgresql or redis are generally recommended for scalability and reliability.

Let’s move directly to the examples.

**Example 1: A simple name-gathering bot using a memory storage**

This example demonstrates the core principle. We'll use in-memory storage for simplicity, but keep in mind that it’s generally not suitable for production environments. Here, our bot will transition between two states: `GETTING_NAME` and `GETTING_AGE`.

```python
from aiogram import Bot, Dispatcher, types, executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

# Replace with your bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"

# Configure the bot and dispatcher
storage = MemoryStorage()  # In-memory storage
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)


class UserData(StatesGroup):
    GETTING_NAME = State()
    GETTING_AGE = State()


@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message, state: FSMContext):
    await message.answer("Hello, what's your name?")
    await UserData.GETTING_NAME.set()

@dp.message_handler(state=UserData.GETTING_NAME)
async def get_name_handler(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['name'] = message.text
    await message.answer("Nice to meet you, " + message.text + ". How old are you?")
    await UserData.GETTING_AGE.set()

@dp.message_handler(state=UserData.GETTING_AGE)
async def get_age_handler(message: types.Message, state: FSMContext):
   async with state.proxy() as data:
       data['age'] = message.text
       name = data.get('name')
       age = data.get('age')
   await message.answer(f" {name}, you are {age} years old! Thank you")
   await state.finish()

async def on_startup(dispatcher):
    await bot.set_webhook("YOUR_WEBHOOK_URL")  # Use your actual webhook URL here

async def on_shutdown(dispatcher):
    await bot.delete_webhook()

if __name__ == '__main__':
     executor.start_webhook(
        dispatcher=dp,
        webhook_path='/',
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host='0.0.0.0',
        port=8080,
    )
```
In this code, `StatesGroup` defines our states. The `@dp.message_handler(state=UserData.GETTING_NAME)` decorator ensures that the `get_name_handler` is only called when the bot is in the `GETTING_NAME` state. The `state` parameter in handler functions gives access to the `FSMContext`. Using `state.proxy()` lets us store user information like name and age in the state. The `state.finish()` at the end of a conversation cleans up the storage and ends the state. The `start_webhook` method initiates the bot with webhook updates.

**Example 2: Incorporating a custom dispatcher condition**

Now, let’s introduce a bit more complexity. Imagine we have a state where the bot expects either a number (to proceed) or a message with the text "cancel" (to break out of the process). We'll use a custom filter to handle this:

```python
from aiogram import Bot, Dispatcher, types, executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Filter
from aiogram.dispatcher.filters.state import State, StatesGroup

# Replace with your bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"

# Configure the bot and dispatcher
storage = MemoryStorage()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)

class MyFilter(Filter):
   async def check(self, message: types.Message):
      return message.text.isdigit() or message.text.lower() == 'cancel'

class OrderData(StatesGroup):
    GETTING_COUNT = State()

@dp.message_handler(commands=['order'])
async def start_order(message: types.Message, state: FSMContext):
    await message.answer("How many items would you like to order?")
    await OrderData.GETTING_COUNT.set()

@dp.message_handler(state=OrderData.GETTING_COUNT, filter=MyFilter())
async def get_count_handler(message: types.Message, state: FSMContext):
    if message.text.lower() == 'cancel':
       await message.answer("Order cancelled")
       await state.finish()
    else:
        async with state.proxy() as data:
            data['count'] = int(message.text)
        await message.answer(f"Great, {message.text} items, anything else?")
        await state.finish()

@dp.message_handler(state=OrderData.GETTING_COUNT)
async def invalid_count_handler(message: types.Message):
    await message.answer("Please send a number or 'cancel'")

async def on_startup(dispatcher):
    await bot.set_webhook("YOUR_WEBHOOK_URL") # Use your actual webhook URL here

async def on_shutdown(dispatcher):
    await bot.delete_webhook()


if __name__ == '__main__':
     executor.start_webhook(
        dispatcher=dp,
        webhook_path='/',
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host='0.0.0.0',
        port=8080,
    )
```

Here, `MyFilter` acts as a condition to ensure a valid number or the 'cancel' message is provided, and the `invalid_count_handler` catches any invalid message, enhancing the bot’s error handling. This way, instead of a simple state, we have a state with specific input requirements.

**Example 3: Using a Database for state persistence (Redis)**

Finally, let's demonstrate the use of redis, a more robust and common storage option suitable for most production environments.

```python
from aiogram import Bot, Dispatcher, types, executor
from aiogram.contrib.fsm_storage.redis import RedisStorage2
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import redis

# Replace with your bot token and Redis connection details
BOT_TOKEN = "YOUR_BOT_TOKEN"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Configure redis connection
redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
storage = RedisStorage2(redis=redis_conn)

# Configure the bot and dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=storage)


class FeedbackData(StatesGroup):
    GETTING_RATING = State()
    GETTING_COMMENT = State()

@dp.message_handler(commands=['feedback'])
async def start_feedback(message: types.Message, state: FSMContext):
    await message.answer("Please rate our service from 1 to 5")
    await FeedbackData.GETTING_RATING.set()


@dp.message_handler(state=FeedbackData.GETTING_RATING)
async def get_rating_handler(message: types.Message, state: FSMContext):
    if message.text.isdigit() and 1 <= int(message.text) <= 5:
       async with state.proxy() as data:
         data['rating'] = int(message.text)
       await message.answer("Great, please provide any comments")
       await FeedbackData.GETTING_COMMENT.set()
    else:
       await message.answer("Invalid rating, please send a number from 1 to 5")


@dp.message_handler(state=FeedbackData.GETTING_COMMENT)
async def get_comment_handler(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['comment'] = message.text
        rating = data.get('rating')
        comment = data.get('comment')
    await message.answer(f"Thanks for the feedback, you rated us {rating} with comment: {comment}")
    await state.finish()


async def on_startup(dispatcher):
    await bot.set_webhook("YOUR_WEBHOOK_URL") # Use your actual webhook URL here

async def on_shutdown(dispatcher):
    await bot.delete_webhook()


if __name__ == '__main__':
     executor.start_webhook(
        dispatcher=dp,
        webhook_path='/',
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host='0.0.0.0',
        port=8080,
    )

```
Here, we use `RedisStorage2` to store the bot's state in redis. This setup allows state persistence even if the bot is restarted or scaled to multiple instances, which is especially critical in webhook-based systems.

**Recommended Resources:**

For a deep dive into state management and finite state machines, I recommend the following:

1.  **"Introduction to Automata Theory, Languages, and Computation" by John E. Hopcroft, Rajeev Motwani, and Jeffrey D. Ullman:** This classic text provides the theoretical underpinnings of finite state machines, which are crucial for understanding their applications.

2.  **Official aiogram Documentation:** The documentation provides complete and up-to-date information on all the features available within the library, including states and webhook configurations. It’s essential to stay aligned with the latest releases.

3.  **"Designing Data-Intensive Applications" by Martin Kleppmann:** This book offers excellent insights into different data storage and retrieval methods, which is very useful for making choices regarding state persistence.

In conclusion, implementing an fsm using aiogram webhooks is a matter of combining states defined using `StatesGroup`, managing the state context with `FSMContext`, and incorporating robust persistence mechanisms such as redis for real-world deployments. These examples offer a starting point, and further customization will depend on the specific requirements of each project. Remember, maintain a well-defined set of states and transitions for a robust and maintainable bot.
