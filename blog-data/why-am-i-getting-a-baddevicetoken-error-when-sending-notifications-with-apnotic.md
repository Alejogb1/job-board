---
title: "Why am I getting a BadDeviceToken error when sending notifications with apnotic?"
date: "2024-12-23"
id: "why-am-i-getting-a-baddevicetoken-error-when-sending-notifications-with-apnotic"
---

Let's unpack that `BadDeviceToken` error you're seeing with apnotic, because honestly, it's a common pitfall that I've seen countless times, and yes, I've definitely battled it myself more than once. It usually indicates a mismatch between the device token you're using to send notifications and the actual state of the app and device on apple’s side. It's not that your code is fundamentally broken; it's more about the nuances of how apple's push notification service works, and those nuances can be a pain point if not understood correctly.

To elaborate, the `BadDeviceToken` error typically arises when the token you've acquired is no longer considered valid by apns (apple push notification service). This invalidation can stem from a few specific scenarios. A device token becomes invalid when an app is uninstalled, then reinstalled; if the user explicitly disables notifications for your app, that token will be rejected; if you attempt to send notifications to a token from a sandbox environment with your production environment, or the opposite, you’ll see an error. Finally, there are token rotation policies by apple, tokens don't remain valid forever. These are some key areas. The system might have determined that your app, identified by that particular token, is no longer registered to receive push notifications. Think of it like an expired passport – your access is revoked.

I recall one particularly tricky situation a couple of years ago when debugging a cross-platform app. The issue surfaced when our users started reporting intermittent notification failures. We were using apnotic alongside a custom backend, and the logs were littered with `BadDeviceToken` errors. Initially, we assumed there was a bug in the way we were storing device tokens. We meticulously audited our database, ensuring correct storage and retrieval. However, the errors persisted. After a few days of frustrating troubleshooting, we discovered two distinct problems. One, we had accidentally mixed device tokens from different development and production environments and two, some users had uninstalled and reinstalled the app, generating a new token that wasn't captured correctly on the backend. It was a learning experience to say the least, and that taught me to really pay attention to token lifecycle management.

Now, let's get more technical and dive into some practical coding examples to show you how to handle these kinds of errors. I will use python here but these concepts apply to any language when you are dealing with push notifications and dealing with the apple push notification service.

**Example 1: Handling `BadDeviceToken` Errors Directly**

This code snippet showcases how to catch the `apnotic.errors.BadDeviceToken` error specifically and what you can do to address it, like removing the offending token from your database:

```python
import asyncio
import apnotic
from apnotic.errors import BadDeviceToken

async def send_notification(device_token, message, client):
    try:
        payload = {"aps": {"alert": message}}
        notification = apnotic.Notification(device_token, payload)
        await client.send_notification(notification)
        print(f"Notification sent successfully to {device_token}")

    except BadDeviceToken:
        print(f"Invalid device token detected: {device_token}. Removing from database.")
        # Function to remove device token
        remove_device_token_from_db(device_token) # assume we have an implementation for this.
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    # Mock client setup - replace with your actual client setup
    client_key = "path/to/your/apns.key"
    team_id = "your_team_id"
    key_id = "your_key_id"
    bundle_id = "your_bundle_id"

    client = apnotic.Client(
        key=client_key,
        key_id=key_id,
        team_id=team_id,
        bundle_id=bundle_id
    )
    client.connect()

    # Replace with your list of device tokens
    device_tokens = ["valid_token_1", "invalid_token_2", "valid_token_3"]
    for token in device_tokens:
         await send_notification(token, "Hello from apnotic!", client)

    client.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
```

In this example, the `send_notification` function now includes error handling for `BadDeviceToken`. When apnotic raises this specific error, it's handled, and we might trigger a function (`remove_device_token_from_db`) to purge the invalid token from the database, ensuring it is not tried again. The client setup is an example and you must make sure this is configured correctly.

**Example 2: Differentiating Between Development and Production Tokens**

The next snippet shows a simple example of how to differentiate between sandbox and production environment based on the token itself. This is crucial because tokens from one environment won't work in the other.

```python
import apnotic
from apnotic.errors import BadDeviceToken

def is_sandbox_token(device_token):
   # usually, a token does not directly reveal if it is a sandbox token but this is a simplification.
   # In production you must maintain state to determine the context of the token when you first acquire it.
    return "sandbox" in device_token.lower() # assuming sandbox is present in the token as an example

async def send_notification(device_token, message, client, is_production=False):
    try:
        payload = {"aps": {"alert": message}}
        notification = apnotic.Notification(device_token, payload)
        notification.topic = client.bundle_id if is_production else f"{client.bundle_id}.dev"  # append .dev for sandbox
        await client.send_notification(notification)
        print(f"Notification sent successfully to {device_token}")

    except BadDeviceToken:
        print(f"Invalid device token detected: {device_token}. Check your environment.")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    # Mock client setup - replace with your actual client setup
    client_key = "path/to/your/apns.key"
    team_id = "your_team_id"
    key_id = "your_key_id"
    bundle_id = "your_bundle_id"

    client = apnotic.Client(
        key=client_key,
        key_id=key_id,
        team_id=team_id,
        bundle_id=bundle_id
    )
    client.connect()
     # Example tokens, one from production, one from sandbox
    device_tokens = ["production_token_1", "sandbox_token_2"]

    for token in device_tokens:
        is_prod = not is_sandbox_token(token)
        await send_notification(token, "Hello from apnotic!", client, is_prod)


    client.disconnect()

if __name__ == '__main__':
    asyncio.run(main())

```
In this snippet, the `is_sandbox_token` function provides an overly simplified check; you'd need to incorporate your own logic for discerning between token environments. We then adjust the topic for our notification to append a `.dev` when we are using a development token, this needs to mirror the app identifier setup on the apple developer portal.

**Example 3: Monitoring Token Updates**

This last example highlights an important function we would implement in our clients that handles a token update.

```python
import asyncio

async def handle_token_update(new_token, user_id):
    # implementation
    print(f"New token: {new_token} for user id: {user_id}. Update database record for user")


async def mock_token_generation(user_id):
    #simulate how a device will obtain a new token. This token would need to be stored server side
    await asyncio.sleep(1)
    print("Generating new token")
    return f"new_token_{user_id}"


async def main():
    user_id = "user_123"
    new_token = await mock_token_generation(user_id)
    await handle_token_update(new_token, user_id)


if __name__ == '__main__':
    asyncio.run(main())
```

This function `handle_token_update` simulates what you might implement on the server side to deal with a token update that was initiated by a client. Notice how the new token `new_token` is then passed to this server side method. Usually, in a live application, you would listen to the `didRegisterForRemoteNotificationsWithDeviceToken` delegate method (in swift or objective-c) to acquire the new token and send it to your backend as soon as possible.

To wrap up, addressing the `BadDeviceToken` error requires a thorough understanding of APNS and your own app's notification lifecycle. I'd recommend thoroughly reviewing apple's official documentation on push notifications and the apnotic library’s documentation. "iOS Programming: The Big Nerd Ranch Guide" by Aaron Hillegass offers a comprehensive look into the iOS environment, which can be beneficial, while Apple's official "Local and Remote Notification Programming Guide" is an absolute must-read. Also, pay special attention to the section on provider authentication. Additionally, consider delving into the specifics of apnotic's error handling mechanisms for more clarity. Remember, persistent vigilance over token management will drastically reduce the headaches caused by invalid tokens.
