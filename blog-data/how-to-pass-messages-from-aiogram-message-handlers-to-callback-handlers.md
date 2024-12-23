---
title: "How to pass messages from AIOgram message handlers to callback handlers?"
date: "2024-12-23"
id: "how-to-pass-messages-from-aiogram-message-handlers-to-callback-handlers"
---

Alright, let's dive into this. Handling inter-handler communication in AIOgram, particularly between message and callback handlers, is a common challenge I’ve encountered in several bot projects. It’s a problem that initially seems tricky but becomes quite manageable with the correct approach. You wouldn't want your bot to lose context or require excessive, clunky state management, would you?

One might think of it as coordinating different departments in a company. The message handler, which processes initial user inputs, acts as the customer service team receiving requests. The callback handler, responding to inline keyboard interactions, is more like the specialist team fulfilling those requests based on details. These teams need to communicate, and efficiently, without losing track of what's going on. My past project building an internal task management bot extensively explored various solutions for this, finally settling on a few stable methods I'll walk you through.

The fundamental challenge arises because message and callback handlers are typically independent. A message handler is triggered by a new message text, while a callback handler is activated by a user pressing an inline keyboard button. There's no inherent mechanism for directly sharing data across these different event flows.

The first, and often simplest, method is leveraging the user’s state using the built-in state machine. AIOgram offers powerful state management features through its `StatesGroup` mechanism. This is essentially a class that defines the various stages or states within a conversation. You can store temporary information in these states, accessible across different handlers. Here’s how that might work:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage

# Initialize bot and dispatcher
bot = Bot(token="YOUR_BOT_TOKEN")
storage = MemoryStorage() # You might use Redis or other persistent storage for prod
dp = Dispatcher(bot, storage=storage)

# Define a conversation state
class MyConversation(StatesGroup):
    waiting_for_category = State()
    waiting_for_details = State()

# Message handler to initiate the flow
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message, state: FSMContext):
    await message.reply("Select a category:", reply_markup=types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="Category A", callback_data="category_a")],
        [types.InlineKeyboardButton(text="Category B", callback_data="category_b")]
    ]))
    await MyConversation.waiting_for_category.set()


# Callback handler triggered by inline keyboard
@dp.callback_query_handler(lambda c: c.data in ["category_a", "category_b"], state=MyConversation.waiting_for_category)
async def category_callback_handler(callback_query: types.CallbackQuery, state: FSMContext):
    category = callback_query.data
    await state.update_data(category=category)
    await callback_query.message.edit_text(f"You chose {category}. Now, please enter details.")
    await MyConversation.waiting_for_details.set()
    await callback_query.answer()


# Message handler to retrieve data from the state
@dp.message_handler(state=MyConversation.waiting_for_details)
async def details_handler(message: types.Message, state: FSMContext):
    details = message.text
    data = await state.get_data()
    category = data.get('category')
    await message.reply(f"Category: {category}, Details: {details}")
    await state.finish()

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
```

In this example, we initiate the flow with a `/start` command, transitioning to the `waiting_for_category` state. The `category_callback_handler` then captures the user's category selection, stores it within the state, and moves to `waiting_for_details`. Finally, the `details_handler` retrieves both the category and entered details, processing them before resetting the state using `state.finish()`. This approach maintains context across user interactions, ensuring a coherent flow.

The second technique I frequently use is a more general-purpose approach using an in-memory dictionary or a more persistent cache, acting as a kind of ‘message bus’. This isn’t ideal for large-scale production, given it resides solely in the server's memory but is perfect for smaller-scale applications or as an intermediate approach before implementing more complex options such as Redis. The key here is to assign a unique identifier, often the user id, to the message and then store relevant data keyed to this identifier in a dictionary. Here’s how that might look:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage

# Initialize bot and dispatcher
bot = Bot(token="YOUR_BOT_TOKEN")
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Initialize in memory cache
message_cache = {}

# Message handler to initiate the flow
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    message_cache[user_id] = {"step": 1}
    await message.reply("Select an option:", reply_markup=types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="Option A", callback_data=f"option_a_{user_id}")],
        [types.InlineKeyboardButton(text="Option B", callback_data=f"option_b_{user_id}")]
    ]))

# Callback handler triggered by inline keyboard
@dp.callback_query_handler(lambda c: c.data.startswith("option_"),)
async def option_callback_handler(callback_query: types.CallbackQuery):
    data = callback_query.data.split("_")
    option = data[0]
    user_id = int(data[2])
    
    if user_id in message_cache:
        message_cache[user_id]["option"] = option
        await callback_query.message.edit_text(f"You chose {option}. Now what?")
        await callback_query.answer()
    else:
        await callback_query.answer("Cache missing!", show_alert=True)

# Message handler to access cached values
@dp.message_handler(lambda message: message.from_user.id in message_cache and "option" in message_cache[message.from_user.id])
async def final_handler(message: types.Message):
    user_id = message.from_user.id
    user_data = message_cache.get(user_id)
    
    if user_data:
      option = user_data["option"]
      await message.reply(f"Your selected option was: {option}")
      del message_cache[user_id]
    else:
        await message.reply("Something went wrong.")
    

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
```

Here, we use the user ID as the key in the `message_cache`, storing a dictionary containing the current step and, later, the chosen option. The callback data includes the user ID, so we can retrieve their specific cache entry. While straightforward, this approach should be handled with caution in production, especially with concurrent user activity.

Finally, for more sophisticated applications, utilizing a robust, external caching solution like Redis is advisable. This ensures data persistence and scalability, critical when many users are interacting simultaneously. Redis allows for structured data storage and retrieval, with atomic operations to avoid data inconsistencies. The implementation is similar to our in-memory cache example, but you replace the python dictionary with Redis commands.

```python
import redis
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
# Initialize bot and dispatcher
bot = Bot(token="YOUR_BOT_TOKEN")
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
# Initialize redis cache
redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)

# Message handler to initiate the flow
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    redis_client.set(f"user:{user_id}:step", "1")
    await message.reply("Select an option:", reply_markup=types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="Option A", callback_data=f"option_a_{user_id}")],
        [types.InlineKeyboardButton(text="Option B", callback_data=f"option_b_{user_id}")]
    ]))


# Callback handler triggered by inline keyboard
@dp.callback_query_handler(lambda c: c.data.startswith("option_"),)
async def option_callback_handler(callback_query: types.CallbackQuery):
  data = callback_query.data.split("_")
  option = data[0]
  user_id = int(data[2])

  redis_client.set(f"user:{user_id}:option", option)
  await callback_query.message.edit_text(f"You chose {option}. Now what?")
  await callback_query.answer()
  


# Message handler to access cached values
@dp.message_handler(lambda message: redis_client.exists(f"user:{message.from_user.id}:option"))
async def final_handler(message: types.Message):
    user_id = message.from_user.id
    option = redis_client.get(f"user:{user_id}:option").decode('utf-8')
    await message.reply(f"Your selected option was: {option}")
    redis_client.delete(f"user:{user_id}:option")
    redis_client.delete(f"user:{user_id}:step")
    

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)
```

Here, we store user-specific state in Redis with a user ID prefix. The handlers interact with Redis to both set and get the data. Using Redis provides much better robustness and allows for scaling up the application much easier.

For a deeper dive, I would recommend reading the AIOgram documentation thoroughly, focusing on their state machine. Also, consider examining "Redis in Action" by Josiah L. Carlson for a solid grasp of Redis principles and implementation. These resources were invaluable to me when I was figuring out inter-handler communication. They cover these topics in more detail and offer various strategies beyond what we've discussed. I hope these examples help you navigate this area of AIOgram more effectively.
