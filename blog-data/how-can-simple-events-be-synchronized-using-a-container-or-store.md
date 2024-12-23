---
title: "How can simple events be synchronized using a container or store?"
date: "2024-12-23"
id: "how-can-simple-events-be-synchronized-using-a-container-or-store"
---

, let’s tackle this. I remember a particularly gnarly project about five years back where we were building a distributed system for processing real-time sensor data. We had numerous microservices, each responsible for a specific data stream, and the challenge was to get them to react in a coordinated manner to events occurring across the entire system. It wasn't about heavy-duty distributed consensus algorithms here; it was about managing a collection of relatively simple events, and getting the services to play nicely without excessive complexity. This is where the concept of using a centralized container or store for synchronization really shone.

The core idea revolves around maintaining a shared, accessible data structure that holds the current state or the occurrence of specific events. Services subscribe to changes within this container, triggering actions based on these changes. Essentially, this is a publish/subscribe pattern, but implemented with a focus on simple event tracking and notification. Let me walk you through how this can work, along with some specific approaches I've used, and then we'll jump into some illustrative code.

The key isn't to build a heavyweight message queue but rather to create a shared space where events are registered and monitored. This might take the form of a simple dictionary in memory, a lightweight database, or even a specialized data store. The specifics often depend on the scale, performance requirements, and the nature of the events we're synchronizing. In our sensor data system, for instance, a key-value store (like Redis) worked remarkably well because its fast read/write performance and straightforward data structures. The main purpose wasn't data persistence; it was rapid event distribution and synchronization across our various components.

Now, how do we effectively synchronize using such a setup? The workflow typically includes these steps:

1.  **Event Publication:** When an event occurs in a service, that service writes the event's details (such as an event type and associated data) into the shared container. The format of this data is up to the developer, but generally, something like a json or a serialized structure works fine.
2.  **Subscription:** Other services subscribe to changes in this shared container related to the specific events they care about. This could involve polling or more sophisticated mechanisms such as pub/sub functionality if available from the storage solution.
3.  **Change Detection:** The shared container must notify services of new events or updates to existing events. The method of notification could be via a polling mechanism, a dedicated notification system (like Redis Pub/Sub), or another means such as an event emitter pattern if the data structure is contained entirely within application memory.
4.  **Event Handling:** When a service detects a relevant event, it retrieves the event data from the container and executes the necessary logic. This logic is usually very specific to the needs of the service subscribing to the event.

Crucially, this approach isn't designed for high-volume message processing or guaranteed message delivery. If you’re dealing with those conditions, you should start by examining more advanced solutions, such as Apache Kafka, RabbitMQ, or even cloud-based equivalents like AWS Kinesis. However, for simpler use cases, using a container or store allows services to synchronize their behaviors without becoming tangled in intricate communication protocols.

Let's look at some code examples to illustrate this, starting with an in-memory implementation, demonstrating the core concepts.

```python
import threading
import time

class SimpleEventStore:
    def __init__(self):
        self.events = {}
        self.subscriptions = {}
        self.lock = threading.Lock()

    def publish(self, event_type, event_data):
        with self.lock:
            self.events[event_type] = event_data
            if event_type in self.subscriptions:
                for callback in self.subscriptions[event_type]:
                    callback(event_data)

    def subscribe(self, event_type, callback):
        with self.lock:
            if event_type not in self.subscriptions:
                self.subscriptions[event_type] = []
            self.subscriptions[event_type].append(callback)

def event_handler_1(data):
    print(f"Handler 1 received event: {data}")

def event_handler_2(data):
    print(f"Handler 2 received event: {data}")


event_store = SimpleEventStore()
event_store.subscribe("sensor_reading", event_handler_1)
event_store.subscribe("sensor_reading", event_handler_2)

event_store.publish("sensor_reading", {"temperature": 25})
event_store.publish("sensor_reading", {"temperature": 27})
```

In this python snippet, we have an in-memory `SimpleEventStore`. It manages a dictionary to hold events and their subscribers, also utilizing a lock for thread safety. Multiple handlers can subscribe to the same event type and will get notified when a new event occurs. This approach is perfect when the system is relatively small and doesn’t need sophisticated features like event persistence. It’s important to note that this is a simplified version and might not be suitable for a production environment with a large number of handlers and high-event volume.

Let's illustrate using a simplified Redis store as well. This example shows how we can move away from an in-memory event store, into a more robust system where events can be managed externally:

```python
import redis
import json
import time

class RedisEventStore:
    def __init__(self, host='localhost', port=6379):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.channel = 'event_channel' # Redis Pub/Sub channel

    def publish(self, event_type, event_data):
        payload = json.dumps({'type': event_type, 'data': event_data})
        self.redis.publish(self.channel, payload)

    def subscribe(self, callback):
      pubsub = self.redis.pubsub()
      pubsub.subscribe(self.channel)

      for message in pubsub.listen():
        if message['type'] == 'message':
           try:
              payload = json.loads(message['data'])
              callback(payload)
           except json.JSONDecodeError:
                print("Error decoding json payload, skipping message")
                continue

def event_handler(payload):
    print(f"Event Handler received: {payload}")


if __name__ == "__main__":
    store = RedisEventStore()

    # Start the listener in its own thread
    import threading
    subscriber_thread = threading.Thread(target = store.subscribe, args = [event_handler])
    subscriber_thread.daemon = True # Ensure this thread will be killed along with the main thread.
    subscriber_thread.start()


    time.sleep(1) # Give the subscriber time to setup
    store.publish("sensor_reading", {"temperature": 28})
    store.publish("sensor_update", {"status": "online"})
```

This snippet uses Redis pub/sub which is a lightweight, fast, and highly scalable way to distribute messages across applications. The subscriber registers itself to a Redis channel, and any events published will be received as they happen. The message body contains the event type and related data, which is useful for complex systems. The asynchronous nature of redis publish-subscribe means that publishers don't have to wait for acknowledgment from each subscriber. This improves the overall throughput of the system, but also comes with some caveats such as message ordering is not guaranteed, and the loss of messages is a possibility. We handle these in a later example.

For my final example, I want to focus on dealing with the caveats of pub/sub, where we ensure we don't lose events and the subscribers can reliably receive the event. This shows the evolution of our previous example, where we now write events to a stream, instead of a simple pub/sub channel.

```python
import redis
import json
import time
import uuid
from redis.exceptions import ConnectionError

class RedisStreamEventStore:
    def __init__(self, host='localhost', port=6379, stream_name='events'):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.stream_name = stream_name
        self.consumer_group = "my_consumers"

        try:
          self.redis.xgroup_create(self.stream_name, self.consumer_group, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
          if 'BUSYGROUP' not in str(e):
             raise # Re-raise anything that isn't a BUSYGROUP error.

    def publish(self, event_type, event_data):
        payload = json.dumps({'type': event_type, 'data': event_data})
        self.redis.xadd(self.stream_name, {'payload': payload})

    def subscribe(self, consumer_name, callback):
      while True: # handle intermittent disconnections
        try:
          messages = self.redis.xreadgroup(self.consumer_group, consumer_name, {self.stream_name: '>'}, count=1, block=5000)

          for stream, events in messages:
            for event_id, message in events:
               try:
                  payload = json.loads(message["payload"])
                  callback(payload)
                  self.redis.xack(self.stream_name, self.consumer_group, event_id) # Acknowledge the message when processed to allow us to move to the next
               except json.JSONDecodeError:
                    print("Error decoding json payload, skipping message")
                    self.redis.xack(self.stream_name, self.consumer_group, event_id)
                    continue

        except ConnectionError as e:
            print(f"Redis Connection Error: {e}")
            time.sleep(5)
            print("Trying to reconnect to redis...")

def event_handler(payload):
    print(f"Event Handler received: {payload}")


if __name__ == "__main__":
    store = RedisStreamEventStore()

    consumer_id = str(uuid.uuid4())
    import threading
    subscriber_thread = threading.Thread(target=store.subscribe, args=[consumer_id, event_handler])
    subscriber_thread.daemon = True
    subscriber_thread.start()


    time.sleep(1)
    store.publish("sensor_reading", {"temperature": 28})
    store.publish("sensor_update", {"status": "online"})
```

Here, we’ve shifted to using Redis Streams for persistent, ordered message delivery. Subscribers consume events using consumer groups, which ensures that even if a consumer goes offline, it can resume processing from where it left off. The `xack` command is used to acknowledge successful processing of a message, preventing message loss and providing at-least-once semantics. This pattern is more robust than simple pub/sub, and although it introduces slightly more complexity, provides significant improvements for reliability and resilience.

Regarding further learning, I strongly suggest diving into “Designing Data-Intensive Applications” by Martin Kleppmann for a thorough understanding of data systems and distributed architecture. Also, the documentation for Redis itself provides exhaustive details on how their data structures, including Streams, function, which will significantly help when implementing robust solutions for event-driven systems. These two resources should provide you with a very solid technical base to develop your own event synchronization mechanisms, and will provide both the practical knowledge, and the deeper theoretical underpinnings for the choices you make in your designs.
