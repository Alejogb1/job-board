---
title: "How to send a random .pdf file to aigram from a folder?"
date: "2024-12-15"
id: "how-to-send-a-random-pdf-file-to-aigram-from-a-folder"
---

alright, let's tackle this. so, you're aiming to pick a pdf at random from a directory and then ship it off to telegram using aigram, cool. i've bumped into similar things myself, a few times over the years.

first off, grabbing a random file. it's actually simpler than it might look. python’s `random` and `os` modules are our friends here. i remember, back when i was first messing with automating some report generation, i needed to do this exact thing – choose one of the output pdfs for archiving. i started by thinking i needed some complex weighted random selection, which, looking back, is just ridiculous for the task at hand.

here's a basic way to do it:

```python
import os
import random

def get_random_pdf(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return none # or log a warning, or handle this edge case
    return os.path.join(folder_path, random.choice(pdf_files))

#usage
folder = "/path/to/your/pdfs"
random_pdf_path = get_random_pdf(folder)
if random_pdf_path:
    print(f"random pdf chosen: {random_pdf_path}")
else:
    print(f"no pdfs found in {folder}")
```

what this does is grab all the names in the directory and uses a list comprehension to filter them so only files that ends with `.pdf` are considered. then, if that list is empty it returns a `none` object. otherwise uses `random.choice` to pick one at random. i'm using the `os.path.join` function to construct the full path to the file because its a lot less error prone. for example, in windows if you try manually to join path strings with "/" you will get problems.

notice that i'm also lowercasing the extension with `.lower()` to ensure we catch both `.pdf` and `.PDF` cases. little things like this have saved me some debugging time, countless times when dealing with user uploaded files with random casing. i recall a situation at my previous place where we had a system that was supposed to handle file uploads of invoices, and the whole pipeline crashed because half the users were sending `.PDF` and the other half were sending `.pdf`.

now, about the telegram bit using aigram. i’m assuming you’ve got a telegram bot token and you've installed the `aiogram` package. i will not go deep into how to setup your bot with aiogram that is another topic. if you dont, there's plenty of examples in their official documentation. it's good. i was using telebot for some side projects before i discovered aiogram. moving to aiogram was a game changer for my telegram bot development because of its structure and asynchronous nature, handling of several chats at the same time is something that is a lot easier to deal with in aiogram.

i assume that you have your bot running and that you are sending commands to it through a client. a basic implementation for sending the pdf:

```python
import asyncio
from aiogram import bot, dispatcher, types, executor
from aiogram.types import inputfile
from aiogram.contrib.fsm_storage.memory import memorystorage

bot_token = "your_bot_token"
storage = memorystorage()
bot = bot(token=bot_token)
dp = dispatcher(bot, storage=storage)

@dp.message_handler(commands=['sendpdf'])
async def send_random_pdf(message: types.message):
  folder = "/path/to/your/pdfs"
  random_pdf_path = get_random_pdf(folder)

  if random_pdf_path:
      with open(random_pdf_path, 'rb') as pdf_file:
          await bot.send_document(message.chat.id, inputfile.from_read(pdf_file, filename=os.path.basename(random_pdf_path)))
  else:
      await message.reply("no pdfs found, sorry!")

if __name__ == '__main__':
  executor.start_polling(dp, skip_updates=true)
```
here, when the bot receives a `/sendpdf` command, it calls our `get_random_pdf` function, and if there is a valid random pdf file path it sends the pdf as a telegram document.

the key here is `inputfile.from_read`. this reads the file into memory in binary mode and then creates a `inputfile` object that is recognized by aiogram. in my first telegram bot projects i remember i was not using this `inputfile.from_read` method and i was saving files on disk for telegram to find it. it worked but it was totally unnecessary and slower specially for bigger files. the `filename` is just to add the original name to the document. this was needed for a project where i was dealing with medical reports with a standardized nomenclature, and we had to keep the original name of the file for easier traceability.

also, note that i used `with open(...)` to ensure the file gets properly closed after using it. not closing files is something i did a lot in my early coding days, and it comes back to haunt you specially with bigger files where the resources could be locked.

now, let's say you have more complex logic, maybe you want to make sure that the same file is not sent to the user twice or even better, you want to send a set of files to the user. that means that you need some kind of persistance. in this last example i'll implement a more complex stateful example by using an in-memory store, this will not be persistent over restarts but is perfect for demo purposes. i think that the problem of persistance is for another time (maybe using a database).

```python
import asyncio
from aiogram import bot, dispatcher, types, executor
from aiogram.types import inputfile
from aiogram.contrib.fsm_storage.memory import memorystorage
import os
import random

bot_token = "your_bot_token"
storage = memorystorage()
bot = bot(token=bot_token)
dp = dispatcher(bot, storage=storage)

sent_files = {}

def get_random_pdf(folder_path, sent_files_list):
  pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
  available_files = [f for f in pdf_files if os.path.join(folder_path, f) not in sent_files_list]
  if not available_files:
    return none
  return os.path.join(folder_path, random.choice(available_files))

@dp.message_handler(commands=['sendpdf'])
async def send_random_pdf(message: types.message):
    chat_id = message.chat.id
    if chat_id not in sent_files:
        sent_files[chat_id] = []

    folder = "/path/to/your/pdfs"
    random_pdf_path = get_random_pdf(folder, sent_files[chat_id])

    if random_pdf_path:
        with open(random_pdf_path, 'rb') as pdf_file:
            await bot.send_document(chat_id, inputfile.from_read(pdf_file, filename=os.path.basename(random_pdf_path)))
        sent_files[chat_id].append(random_pdf_path)
    else:
        await message.reply("no new pdfs found, all sent")

@dp.message_handler(commands=['reset'])
async def reset_sent_files(message: types.message):
    chat_id = message.chat.id
    sent_files[chat_id] = []
    await message.reply("sent files history reseted for this chat")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=true)
```

in this example there are two things that changed, first, the `get_random_pdf` function accepts a second argument wich is a list of files already sent. it filters out files already sent. then the `send_random_pdf` now keeps track of what files have been sent by using the `sent_files` dictionary. its a simple in-memory key value store. it has a reset command (`/reset`) that clears the `sent_files` for that chat. i know that this is not persistent but is enough for demonstrating the stateful behaviour.

there are tons of edge cases that should be taken into consideration, like handling file errors, large files that could cause memory problems, the size of the telegram files, users that can send a lot of requests at the same time and crash the system. this was my life before getting some experience in the area. that's why there is a whole discipline for devops and systems design. in one of my first projects i created a small script that had a `for` loop that was sending a huge number of requests at the same time to a server. it worked in my machine (of course) and then i deployed that to production with catastrophic results.

for learning more, i'd highly suggest looking into "fluent python" by luciano ramalho for a deep dive into python, covering how the language really works. it's great. for a solid background in telegram bot development, check out the official aiogram documentation, they have good examples. and, if you want to go deep in how to make software more resilient and robust i think that you should take a look at "release it!: design and deploy production-ready software" by michael nyrnberg. this last one is not specifically about python or telegram but it is a very good resource for creating better software.

by the way, i was wondering how to become a librarian? maybe i could have a lot of information at my disposal.

anyway, that’s how i’d approach this task. it's straightforward enough once you know the right tools. let me know if you have any more questions.
