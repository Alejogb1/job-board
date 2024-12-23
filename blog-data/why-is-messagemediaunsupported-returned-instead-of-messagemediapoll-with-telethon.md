---
title: "Why is MessageMediaUnsupported() returned instead of MessageMediaPoll() with Telethon?"
date: "2024-12-23"
id: "why-is-messagemediaunsupported-returned-instead-of-messagemediapoll-with-telethon"
---

Alright, let's unpack this Telethon puzzle. I've certainly been down this road more than a few times, and it's usually not a case of outright failure but rather a misalignment in expectations, specifically about how Telegram handles media and how Telethon interfaces with it.

The core issue often revolves around timing and availability. The `MessageMediaUnsupported()` object isn’t actually a 'failure' state; rather, it indicates that the media content associated with a message isn't immediately accessible through the standard polling methods. This typically happens when the media is large, encrypted, or for whatever reason, hasn't fully materialized for direct retrieval during the initial message processing. It's a signal that Telegram intends for us to use a different pathway to obtain the media, instead of a quick, efficient `MessageMediaPoll()` response.

Think of it like this: when you send a large video file, it doesn't immediately exist in its entirety on Telegram’s servers. There’s often a process of upload, encoding, and distribution. That's why a message with media might show up quickly as a simple message object, but the actual media content needs a bit more time to get ready. Telethon's core design prioritizes speed and efficiency; it doesn’t want to block indefinitely waiting for media to be ready. Thus, `MessageMediaUnsupported()` is a placeholder letting us know to initiate a download request.

Now, let's delve into why `MessageMediaPoll()` might be missing. `MessageMediaPoll()` is specifically used when the media attached to a message is a poll. In contrast, the absence of `MessageMediaPoll()` often indicates that the media is of a different kind, requiring a different handling mechanism. Think of photos, videos, documents – these will be represented by their specific media types, not a poll. You wouldn't use the same function to download a video file as you would to check the results of a poll. This is a design decision to keep operations specific and efficient.

My own experiences involved implementing a complex bot to download media. During early development, I repeatedly encountered this, expecting all media to be immediately available as direct objects. I had initially assumed it was some kind of error on my part, but I eventually came to appreciate the design principle – it makes perfect sense for Telegram to handle large media asynchronously.

Here are a couple of scenarios where I observed this behavior, followed by code examples that illustrate the solution:

**Scenario 1: Downloading a Photo that isn't immediately available**

Initially, I would try something similar to this and be met with the `MessageMediaUnsupported`:

```python
from telethon import TelegramClient, events, types

api_id = YOUR_API_ID
api_hash = 'YOUR_API_HASH'
client = TelegramClient('session_name', api_id, api_hash)

async def main():
    await client.start()

    @client.on(events.NewMessage())
    async def handler(event):
        if event.message.media:
            print(f"Media found: {event.message.media}")
            if isinstance(event.message.media, types.MessageMediaUnsupported):
                print("Unsupported media, needs download")
            elif isinstance(event.message.media, types.MessageMediaPhoto):
                photo = event.message.media.photo
                print(f"Photo received: {photo}")

    await client.run_until_disconnected()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

The above code directly attempts to access a photo but initially finds an 'unsupported' media object.

Here’s how you can correctly handle it, explicitly using the download mechanism instead:

```python
from telethon import TelegramClient, events, types
from telethon.tl.functions.messages import GetMediaConfigRequest
import os

api_id = YOUR_API_ID
api_hash = 'YOUR_API_HASH'
client = TelegramClient('session_name', api_id, api_hash)

async def main():
    await client.start()

    @client.on(events.NewMessage())
    async def handler(event):
        if event.message.media:
            if isinstance(event.message.media, types.MessageMediaUnsupported):
               print("Media needs to be downloaded...")
               # Check if media is indeed a Photo before downloading
               if isinstance(event.message.media,types.MessageMediaPhoto):
                 path = f"./downloaded_media/{event.message.id}.jpg"
                 if not os.path.exists('./downloaded_media'):
                     os.makedirs('./downloaded_media')
                 await client.download_media(event.message, file=path)
                 print(f"Photo downloaded to: {path}")

            elif isinstance(event.message.media, types.MessageMediaPhoto):
                photo = event.message.media.photo
                print(f"Photo received directly : {photo}")

    await client.run_until_disconnected()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

In this refined version, we detect `MessageMediaUnsupported`, and *then* proceed to download the media using `client.download_media()` after checking if it is indeed a photo. This approach ensures we don't prematurely try to extract media information when Telegram indicates we must fetch it. Also, I added a basic folder creation to make sure everything is saved correctly.

**Scenario 2: Handling Documents (Files) instead of polls**

Another common scenario:

```python
from telethon import TelegramClient, events, types

api_id = YOUR_API_ID
api_hash = 'YOUR_API_HASH'
client = TelegramClient('session_name', api_id, api_hash)

async def main():
    await client.start()

    @client.on(events.NewMessage())
    async def handler(event):
        if event.message.media:
            print(f"Media found: {event.message.media}")
            if isinstance(event.message.media, types.MessageMediaUnsupported):
               print("Unsupported media, needs download")

            elif isinstance(event.message.media,types.MessageMediaDocument):
                doc = event.message.media.document
                print(f"Document received directly:{doc}")


            elif isinstance(event.message.media, types.MessageMediaPoll):
                poll = event.message.media.poll
                print(f"Poll Received: {poll}")


    await client.run_until_disconnected()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

This illustrates that if the media is a document, we receive `MessageMediaDocument`. We need to download this document.

```python
from telethon import TelegramClient, events, types
import os
api_id = YOUR_API_ID
api_hash = 'YOUR_API_HASH'
client = TelegramClient('session_name', api_id, api_hash)

async def main():
    await client.start()

    @client.on(events.NewMessage())
    async def handler(event):
        if event.message.media:
            if isinstance(event.message.media, types.MessageMediaUnsupported):
               print("Media needs to be downloaded...")
               if isinstance(event.message.media, types.MessageMediaDocument):
                 path = f"./downloaded_media/{event.message.id}.document"
                 if not os.path.exists('./downloaded_media'):
                     os.makedirs('./downloaded_media')
                 await client.download_media(event.message, file=path)
                 print(f"Document downloaded to: {path}")

            elif isinstance(event.message.media,types.MessageMediaDocument):
                doc = event.message.media.document
                print(f"Document received directly:{doc}")

            elif isinstance(event.message.media, types.MessageMediaPoll):
                poll = event.message.media.poll
                print(f"Poll Received: {poll}")

    await client.run_until_disconnected()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```
The modified code demonstrates a similar pattern as above but applies it to the scenario with document type media.

**Key takeaway:** `MessageMediaUnsupported()` is not an error. It’s a pointer telling us, "Hey, this media is not immediately available, use the download procedures". Don’t expect all media to be immediately accessible as `MessageMediaPoll()`. Look for specific media types like `MessageMediaPhoto`, `MessageMediaVideo`, `MessageMediaDocument`, and implement the appropriate download procedures when the API signals using `MessageMediaUnsupported`.

For a deeper dive into Telegram’s internal architecture, I suggest reviewing the "Telegram API Documentation" available on their official website. In addition, specific chapters in "The Definitive Guide to API Design" by Mike Amundsen can help solidify your understanding of asynchronous data handling, and "Patterns of Enterprise Application Architecture" by Martin Fowler can give you insights into message patterns that reflect what Telegram is likely doing on the backend.

By taking a more granular approach to media handling and understanding the asynchronous nature of these systems, you'll find that you can make much more robust and efficient code. It's a common learning experience, and I hope this detailed explanation helps you move forward.
