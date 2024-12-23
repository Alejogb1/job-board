---
title: "How can Rasa handle custom actions after a response is selected?"
date: "2024-12-23"
id: "how-can-rasa-handle-custom-actions-after-a-response-is-selected"
---

Alright, let's tackle this. The issue of managing custom actions *after* a response has been selected in Rasa is something I’ve encountered quite a bit over the years, particularly when building conversational interfaces that go beyond simple question-answer flows. It's not just about generating a text response; it's about orchestrating the complete interaction lifecycle, which often involves executing business logic, updating databases, or interacting with external services. Let’s break down how Rasa facilitates this, avoiding the trap of over-complicating the process.

The key is understanding that Rasa’s architecture provides us with 'hooks,' essentially strategic places within the dialogue management process where we can insert our custom logic. Specifically, the *after-response selection* phase is handled by what Rasa calls custom actions. These actions are not part of the natural language understanding (nlu) or dialogue management (dm) pipelines directly, but they are triggered by the policy engine as an extension of the intent fulfillment.

Think of it like this: the nlu identifies the user's intent, the dm selects a response based on the current dialogue state, and *then* the policy might specify that certain custom actions need to be executed. These actions run in your custom action server – a separate process from the core Rasa server. This separation is essential; it isolates business logic and any potential performance issues with that logic, keeping the core conversational engine responsive. I’ve seen systems falter when actions were embedded directly in the dialogue flow. This architectural choice, of separating the concern, is one of the major reasons Rasa scales well.

The trigger for these custom actions happens in the policy’s predictions; for example, a policy might predict the `utter_greet` template response and simultaneously predict the `action_log_user` custom action. After the text response is delivered (e.g., "Hello, how can I help?"), the `action_log_user` custom action is executed. This is important – the response and the action are distinct events. The response is *sent* to the user while the action is *executed* by the action server.

The mechanics involve your custom action server exposing an endpoint (often over http) that Rasa calls. Within your action server, you implement the code for each of your custom actions. These actions receive data about the current dialogue state (the tracker) and any slots that have been set and can modify that tracker or interact with external systems.

Let's look at a few practical examples.

**Example 1: Logging User Interactions**

Suppose we want to log the user’s intent and the response delivered. This is a straightforward example of an after-response action that doesn't affect the conversation flow directly but is crucial for analytics and debugging. Here's a Python code snippet for such a custom action:

```python
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging

logging.basicConfig(level=logging.INFO)

class ActionLogInteraction(Action):
    def name(self) -> str:
        return "action_log_interaction"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
       intent = tracker.latest_message.get("intent").get("name")
       response_text = dispatcher.latest_output.get("text")
       logging.info(f"User Intent: {intent}, Response: {response_text}")
       return []
```

In this example, `ActionLogInteraction` retrieves the latest user intent and the text of the dispatched response from the tracker and dispatcher respectively. It then logs this information. The key thing is that this logic is *after* the dispatcher actually delivers the message. We also return an empty list ( `[]` ), indicating we are not modifying the conversational flow.

**Example 2: Updating a User's Profile**

Let's consider a more complex scenario: updating a user’s profile in a database. Imagine that after the user provides their name, you want to store that in the database. Here's how that might look:

```python
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging
import requests

logging.basicConfig(level=logging.INFO)

class ActionUpdateUserProfile(Action):
    def name(self) -> str:
        return "action_update_user_profile"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        name = tracker.get_slot("user_name")  # Assuming 'user_name' is a slot
        if name:
            try:
               #In a real implementation I would use a proper database library here
                response = requests.post("http://my-api-server/users", json={"name": name})
                response.raise_for_status()  # Raise error for bad status codes
                logging.info(f"Updated profile for user: {name}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error updating user profile: {e}")

        return []
```

Here, `ActionUpdateUserProfile` retrieves the `user_name` slot, makes a request to an external API, and handles errors appropriately. This action does not directly impact what the user sees but it does alter the state of the world *after* the user input and the response. This is a crucial aspect of actions; they enable your bot to do things, not just say things.

**Example 3: Triggering External Processes**

Lastly, let’s imagine initiating an order processing pipeline after a user confirms their order. This might involve sending a message to a message queue or initiating a process via another API.

```python
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging
import json
import pika

logging.basicConfig(level=logging.INFO)

class ActionProcessOrder(Action):
    def name(self) -> str:
        return "action_process_order"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        order_details = tracker.get_slot("order_details") # Assume order details are stored in a slot.
        if order_details:
            try:
               # Use AMQP library to publish a message to a queue
                connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
                channel = connection.channel()
                channel.queue_declare(queue='order_queue')
                channel.basic_publish(exchange='', routing_key='order_queue', body=json.dumps(order_details))
                logging.info(f"Placed order on queue: {order_details}")
                connection.close()


            except pika.exceptions.AMQPError as e:
                logging.error(f"Error publishing to queue: {e}")

        return []
```

In this instance, after the user confirms an order (and we’ve delivered the confirmation message), the `ActionProcessOrder` publishes the `order_details` to a message queue, an asynchronous step to process the order further. Again, we use a slot, showing the power of managing state within a conversation and using it to interact with the world outside of the bot.

These examples illustrate the core mechanism: you define custom actions in your action server which are invoked by the policy after a response is selected. They have access to the current tracker state and can perform any operations, updating the tracker and interacting with external systems.

To really understand this in detail, I’d recommend diving into the Rasa documentation, particularly the section on custom actions and policies. Also, the book “Natural Language Understanding with Python” by Dale, et al. provides a good foundational understanding of the underlying concepts of conversational AI and helps solidify your understanding of Rasa’s architecture. For a more theoretical treatment, consider reading “Speech and Language Processing” by Jurafsky and Martin which, while broader, gives the necessary context for a deeper comprehension.

In practice, mastering the art of crafting good custom actions lies not just in the code, but in the careful design of your conversational flow and the judicious use of the tracker’s state management. Remember that actions are about *doing,* they are extensions of your bot’s capabilities beyond simply responding to user queries. It's all about creating that seamless and robust conversational experience.
