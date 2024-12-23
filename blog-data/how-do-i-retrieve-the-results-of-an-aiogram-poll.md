---
title: "How do I retrieve the results of an aiogram poll?"
date: "2024-12-23"
id: "how-do-i-retrieve-the-results-of-an-aiogram-poll"
---

Okay, let's tackle this. Retrieving poll results in aiogram can sometimes feel a bit less straightforward than it initially appears, especially if you've spent time mostly handling standard message updates. I remember back in 2021, during an internal project where we were building a custom feedback bot using telegram, I ran into a similar situation; we wanted to use polls for collecting quick sentiment scores and then analyze the data. This experience made me become quite familiar with the nuances of poll updates.

The challenge isn't necessarily in receiving the initial poll creation update, that’s fairly straightforward, but rather in extracting the finalized results, particularly when the poll closes. Telegram, and consequently aiogram, treats active poll states and final poll results as distinct update types. To retrieve results reliably, you need to correctly identify and parse the `PollAnswer` and `Poll` updates.

Firstly, let's break down the mechanics. When a user casts a vote in an active poll, aiogram dispatches a `types.PollAnswer` update. This update contains information about *which* user voted and *what* options they selected. However, this update alone doesn't give you the aggregated, finalized result of the poll; it just gives you the individual responses. For the final results, you need to look for a `types.Poll` update containing `is_closed=True`. This update is usually sent when the poll is explicitly closed, either by the creator or when the timeout is reached.

So, how do you orchestrate this? You typically need to implement two separate handlers: one to capture individual poll answers (`PollAnswer` update) and another to receive the finalized poll data (`Poll` update). The trick is to maintain some state, usually via a database or in-memory data structure (suitable for simple bots or testing), linking poll IDs to your custom logic.

Here is an initial example demonstrating how to capture the individual `PollAnswer` update:

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.dispatcher import filters

# Replace with your bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# A simple dictionary to store individual poll responses
poll_responses = {}

@dp.poll_answer_handler()
async def handle_poll_answer(poll_answer: types.PollAnswer):
    poll_id = poll_answer.poll_id
    user_id = poll_answer.user.id
    chosen_options = poll_answer.option_ids

    if poll_id not in poll_responses:
        poll_responses[poll_id] = {}
    poll_responses[poll_id][user_id] = chosen_options

    print(f"User {user_id} voted for options {chosen_options} in poll {poll_id}")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
```
This snippet demonstrates how to set up the necessary infrastructure. The `poll_answer_handler` is a decorator that tells aiogram to route all `types.PollAnswer` updates to the `handle_poll_answer` function. We collect the user's selection in a dictionary called `poll_responses`.

The above is useful for capturing individual votes, but it does not yet provide the closed, final poll data. Here’s the second piece, showcasing how to get the `types.Poll` update with the final results.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.dispatcher import filters

# Replace with your bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# A simple dictionary to store poll results
poll_results = {}

@dp.poll_handler(filters.PollClosed()) # only process when the poll is closed
async def handle_closed_poll(poll: types.Poll):
    if poll.is_closed:
        poll_id = poll.id
        poll_results[poll_id] = {}
        poll_results[poll_id]['question'] = poll.question
        poll_results[poll_id]['total_voter_count'] = poll.total_voter_count
        poll_results[poll_id]['options'] = []

        for option in poll.options:
            poll_results[poll_id]['options'].append({
                'text': option.text,
                'voter_count': option.voter_count
            })

        print(f"Poll {poll_id} results: {poll_results[poll_id]}")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

```
Here, we are using `dp.poll_handler(filters.PollClosed())`. This makes sure the handler is called only when a poll update arrives and it’s marked as closed. The function `handle_closed_poll` extracts the final poll information, including the question text, the total voters, and voter counts per option. I'm structuring it into a Python dictionary for easy access.

Now, let’s combine these snippets into a full example. This version will store *both* individual responses and the final poll information, giving you a more complete view of the poll data.

```python
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.dispatcher import filters

# Replace with your bot token
BOT_TOKEN = "YOUR_BOT_TOKEN"
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# Dictionaries to store individual poll responses and results
poll_responses = {}
poll_results = {}


@dp.poll_answer_handler()
async def handle_poll_answer(poll_answer: types.PollAnswer):
    poll_id = poll_answer.poll_id
    user_id = poll_answer.user.id
    chosen_options = poll_answer.option_ids

    if poll_id not in poll_responses:
        poll_responses[poll_id] = {}
    poll_responses[poll_id][user_id] = chosen_options

    print(f"User {user_id} voted for options {chosen_options} in poll {poll_id}")


@dp.poll_handler(filters.PollClosed())
async def handle_closed_poll(poll: types.Poll):
    if poll.is_closed:
        poll_id = poll.id
        poll_results[poll_id] = {}
        poll_results[poll_id]['question'] = poll.question
        poll_results[poll_id]['total_voter_count'] = poll.total_voter_count
        poll_results[poll_id]['options'] = []

        for option in poll.options:
            poll_results[poll_id]['options'].append({
                'text': option.text,
                'voter_count': option.voter_count
            })
        
        print(f"Poll {poll_id} final results: {poll_results[poll_id]}")
        if poll_id in poll_responses:
            print(f"Poll {poll_id} user responses: {poll_responses[poll_id]}")



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

```

This consolidated script demonstrates the full flow: you get individual votes and also the final results when the poll closes. Remember that `poll_responses` and `poll_results` dictionaries are stored in memory in this simple example; in a production environment, you'd want to use a proper database to ensure data persistence.

To deepen your understanding of these concepts I'd strongly recommend looking at the official Telegram Bot API documentation, paying particular attention to the `Poll` and `PollAnswer` objects. Also, consulting the `aiogram` official documentation, specifically the part handling updates, filters, and handlers will be highly beneficial. Furthermore, the source code of the aiogram library itself, while a bit more involved, is also a great learning resource for understanding its internal workings. The book "Programming Telegram Bots" by A. Zuev (available on Leanpub) can also be a valuable resource, as it goes into more detail about aiogram and how it interacts with the Telegram API. These resources will give you a solid theoretical understanding and allow you to tackle more complex cases. From personal experience, nothing replaces hands-on coding combined with sound theoretical understanding.
