---
title: "How to add several inline buttons at once with the Python library aiogram?"
date: "2024-12-14"
id: "how-to-add-several-inline-buttons-at-once-with-the-python-library-aiogram"
---

ah, inline buttons with aiogram, been there, done that, got the t-shirt and probably a few grey hairs to show for it. it’s one of those things that seems straightforward until you actually need to do it dynamically and suddenly you're knee-deep in keyboard markup objects. i remember back when i first started with telegram bots, i spent a whole evening debugging why my buttons were all showing up on separate lines. it was one of those silly mistakes that felt monumental at the time. anyway, let's talk about how to do it the proper way.

the core of the issue is that aiogram's `InlineKeyboardMarkup` expects a list of lists when dealing with rows of buttons. each inner list represents a row, and each item inside those lists should be an `InlineKeyboardButton` instance. if you don’t structure your buttons like this, you end up with the classic problem of all your buttons stacked vertically, which is almost never what someone wants. so, instead of adding buttons one by one with separate `add()` calls (which is a common mistake), we need to build rows of buttons programmatically. it sounds complicated, but it’s not once you wrap your head around the nested list structure.

i've had some interesting situations where i needed to generate these buttons from a database. for instance, i was working on a bot for a local book club. each book was an entry in the database, and each user could select one of the books from an inline keyboard. the keyboard needed to be dynamic, reflecting the current books available. this wasn't a case where i could just predefine static buttons. i started out manually constructing buttons, like i guess everyone does, and quickly realized that the code became unmaintainable pretty fast. especially when dealing with a bunch of books and i needed the page for the buttons. that was a dark era in my bot-making life, let me tell you.

let's go through a simple example: suppose we want to create a row of three buttons. each button will just trigger a different callback query when clicked, something fairly basic to understand how the system works.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

bot = Bot(token="YOUR_BOT_TOKEN") # replace this with your actual token
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    keyboard = InlineKeyboardMarkup(row_width=3) # optional, defaults to 3
    buttons = [
        InlineKeyboardButton("button 1", callback_data="button_1"),
        InlineKeyboardButton("button 2", callback_data="button_2"),
        InlineKeyboardButton("button 3", callback_data="button_3"),
    ]
    keyboard.add(*buttons) # the unpacking operator is key
    await message.answer("choose a button:", reply_markup=keyboard)

@dp.callback_query_handler(lambda c: c.data.startswith('button_'))
async def process_callback(callback_query: types.CallbackQuery):
    button_id = callback_query.data.split("_")[-1]
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, f"you clicked button {button_id}")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

notice how we create a list of `InlineKeyboardButton` instances and then use the unpacking operator (`*`) to add them to the keyboard markup. this makes it that all the buttons are in one row, but if we wanted more rows then we need a list of lists. each inner list representing a row, as i mentioned before.

for example: two rows with two buttons each? here’s how you do that.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

bot = Bot(token="YOUR_BOT_TOKEN") # replace this with your actual token
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    keyboard = InlineKeyboardMarkup()
    row1 = [
        InlineKeyboardButton("button A", callback_data="button_a"),
        InlineKeyboardButton("button B", callback_data="button_b"),
    ]
    row2 = [
        InlineKeyboardButton("button C", callback_data="button_c"),
        InlineKeyboardButton("button D", callback_data="button_d"),
    ]
    keyboard.row(*row1)
    keyboard.row(*row2)
    await message.answer("choose a button:", reply_markup=keyboard)

@dp.callback_query_handler(lambda c: c.data.startswith('button_'))
async def process_callback(callback_query: types.CallbackQuery):
    button_id = callback_query.data.split("_")[-1].upper()
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, f"you clicked button {button_id}")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```
see? this is what i meant when i said that aiogram's markup expects a list of lists. `row()` method is useful in particular situations.

and what if you have a variable number of buttons, like from a database? looping is your best friend:
i've definitely had my fair share of dynamically generating buttons from api data. one time i was building a bot for a local board game club. they wanted to browse games available in their library, and each game had its unique id. that required programmatically creating a set of buttons, each tied to a unique game. the amount of games could change every week.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

bot = Bot(token="YOUR_BOT_TOKEN") # replace this with your actual token
dp = Dispatcher(bot)

# imagine this comes from a database or an api, this is where the actual data will be
games = {
    "game_1": "Catan",
    "game_2": "Ticket to Ride",
    "game_3": "7 Wonders",
    "game_4": "Gloomhaven",
    "game_5": "Pandemic",
}

@dp.message_handler(commands=['games'])
async def games_command(message: types.Message):
    keyboard = InlineKeyboardMarkup(row_width=2) # two buttons per row this time
    buttons = []
    for game_id, game_name in games.items():
        buttons.append(InlineKeyboardButton(game_name, callback_data=game_id))
    keyboard.add(*buttons)
    await message.answer("choose a game:", reply_markup=keyboard)

@dp.callback_query_handler(lambda c: c.data.startswith('game_'))
async def process_game_callback(callback_query: types.CallbackQuery):
    game_id = callback_query.data
    game_name = games.get(game_id, "unknown game")
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, f"you selected {game_name}")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```

in this last example, we iterate over the data of the games and create the buttons programatically, i like this approach the best, it also does not break if you need a lot of buttons or if the source of the data can change. we also use a dictionary to get the game name based on the callback id, there are other ways of doing this of course.

the `row_width` argument in `InlineKeyboardMarkup` is pretty handy. it basically tells aiogram how many buttons to try to put in a single row automatically. after that it automatically will start a new row if the limit was reached. i mean, who wants a single row of 20 buttons, *right?*.

one crucial detail i always seem to forget about is that when i started i'd have issues that would turn out to be just forgetting the `*` before the buttons when calling `add()`, if you forget that asterisk, aiogram will think you are trying to add just the list as a button and not each element in it. you will end up with a button with a text similar to `<aiogram.types.inline_keyboard.InlineKeyboardButton object at 0x...>` which is quite hilarious the first time.

also note, that a user can only have one callback query being processed at a time. so you should try and be quick with your replies or you might run into issues. when the button is pressed, the bot sends back a `CallbackQuery` and the bot needs to answer to that callback query before another one is processed. you do this with `await bot.answer_callback_query(callback_query.id)`.

for more in-depth knowledge, especially about keyboard markups in general, i highly recommend checking out the telegram bot api documentation itself. it might seem daunting at first but it's worth it. you can also find some good examples and explanations in the book "programming telegram bots with python" by sergey v. nikolenko; it's quite a comprehensive guide. or, if you prefer a more academic approach, "introduction to bot development" by alexander zaytsev is pretty solid. those resources helped me back in the day, hopefully they can help you as well. it saved me from a lot of head scratches, that's for sure.

and that's pretty much it. it's all about understanding that nested structure of lists, a bit of loop, and remembering that asterisk when adding multiple buttons in one go. it's not exactly brain surgery, but it’s a fundamental skill for any bot developer. i hope that covers all the basis in a way that's easily understandable. let me know if you have more doubts; i've probably seen them all.
