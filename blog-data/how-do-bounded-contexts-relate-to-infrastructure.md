---
title: "How do bounded contexts relate to infrastructure?"
date: "2024-12-23"
id: "how-do-bounded-contexts-relate-to-infrastructure"
---

Alright, let's tackle this. Thinking back to a project I was on several years ago, the complexities of integrating multiple microservices, each with its own data store, really hammered home the importance of bounded contexts and their direct impact on infrastructure choices. We started with a monolithic application—classic, I know—and the migration to microservices forced us to really confront how those contexts delineated not only our domain logic but also our infrastructure needs. It's not just about drawing boxes on a whiteboard; it's about making practical decisions that affect deployment, data management, and even team structures.

So, how do bounded contexts relate to infrastructure? In essence, a well-defined bounded context acts as a blueprint for its corresponding infrastructural needs. It dictates the types of technologies we use, how we deploy them, and even how we scale them. A bounded context, at its core, encapsulates a specific area of the domain, defining its language, models, and the business logic relevant only within its own boundaries. This separation, far from being just a conceptual exercise, directly influences the technological stack employed. When thinking about the infrastructure, we're not merely considering hardware but also the deployment strategies, databases, message queues, and all the operational necessities that keep the application running.

A key aspect here is data consistency and ownership. Each bounded context should ideally own its data. This principle has direct implications for data storage mechanisms. For instance, if one bounded context primarily manages user authentication, it may utilize a highly optimized key-value store for rapid lookups. Another bounded context handling complex financial transactions may require a robust relational database system that guarantees ACID (Atomicity, Consistency, Isolation, Durability) properties. Trying to share a single database across multiple bounded contexts often leads to tight coupling, schema conflicts, and a higher likelihood of cascading failures – a lesson learned through bitter experience, let me assure you.

This also extends to the deployment strategy. A smaller, simpler bounded context might be perfectly suited for containerized deployment on a platform like kubernetes, while another, with stricter security requirements, might demand a more traditional, isolated server environment. The crucial point is that we're not forcing a single technological solution on the entire application; we’re choosing appropriate tools based on the requirements of each context. This allows us to optimize for performance, maintainability, and security, while also respecting the specific requirements of each distinct area of the application.

The relationship isn't static, however. Changes in the bounded context can drive changes to the infrastructure and vice versa. Let's say, initially, a context could function adequately using an in-memory data store for a small dataset. As the dataset grows, or the requirements shift towards needing persistent data storage or more complex querying, the infrastructure must adapt, leading to potentially a move towards an external database system or even adding dedicated caching mechanisms.

Let’s look at some code examples to make this a bit more concrete. Imagine we have two bounded contexts, ‘User Management’ and ‘Order Processing’.

**Example 1: User Management Context (Data Storage)**

This context deals with user data, primarily requiring fast reads and writes for login and session management:

```python
import redis
import json

class UserStore:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)

    def save_user(self, user_id, user_data):
        user_data_json = json.dumps(user_data)
        self.redis_client.set(f'user:{user_id}', user_data_json)

    def get_user(self, user_id):
        user_data_json = self.redis_client.get(f'user:{user_id}')
        if user_data_json:
            return json.loads(user_data_json)
        return None

    def delete_user(self, user_id):
        self.redis_client.delete(f'user:{user_id}')
```

In this example, redis is a natural choice given its speed and suitability for storing key-value pairs, perfect for handling user sessions. This would likely be deployed in a container alongside the application service.

**Example 2: Order Processing Context (Data Storage)**

This context handles financial transactions and requires robust data integrity.

```python
import sqlite3

class OrderStore:
    def __init__(self, db_path='orders.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                order_items TEXT,
                total_amount REAL,
                order_date TEXT
            )
        ''')
        self.conn.commit()

    def save_order(self, user_id, order_items, total_amount, order_date):
        self.cursor.execute('''
            INSERT INTO orders (user_id, order_items, total_amount, order_date)
            VALUES (?, ?, ?, ?)
        ''', (user_id, json.dumps(order_items), total_amount, order_date))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_order(self, order_id):
        self.cursor.execute('SELECT * FROM orders WHERE order_id = ?', (order_id,))
        order = self.cursor.fetchone()
        if order:
            return {
            'order_id': order[0],
            'user_id': order[1],
            'order_items': json.loads(order[2]),
            'total_amount': order[3],
            'order_date': order[4]
            }
        return None

    def close(self):
      self.conn.close()
```

Here, sqlite3 is used to provide a simple, transactional database. While this is just an example, you can imagine we would use a more robust solution in production. The key takeaway is that the *type* of database is chosen based on the needs of the bounded context.

**Example 3: Inter-Context Communication**

Bounded contexts don't exist in isolation; they need to communicate. Here’s an example of how the order processing context might use a message queue to interact with the user management context.

```python
import pika
import json
from uuid import uuid4


class OrderQueue:
    def __init__(self, queue_name='order_queue', host='localhost'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
        self.channel = self.connection.channel()
        self.queue_name = queue_name
        self.channel.queue_declare(queue=self.queue_name, durable=True)

    def publish_order(self, order_data):
        message_id = str(uuid4())
        message = json.dumps({"message_id": message_id, "order_data": order_data})
        self.channel.basic_publish(exchange='',
                            routing_key=self.queue_name,
                            body=message,
                            properties=pika.BasicProperties(
                              delivery_mode=2, # make message persistent
                            ))
        print(f" [x] Sent order confirmation with id: {message_id}")


    def close(self):
        self.connection.close()

# Example usage in the order context after order processing
# order_queue = OrderQueue()
# order_data = {"order_id": 123, "user_id": 456, "total_amount": 100.00}
# order_queue.publish_order(order_data)
# order_queue.close()
```

Here, I’ve illustrated that the order context might publish a message using a message queue after an order has been processed, perhaps to trigger subsequent actions in another context (notification service, perhaps?). This demonstrates a form of asynchronous communication that minimizes direct coupling. The infrastructure would now need to include the message broker along with the deployment environment.

In conclusion, aligning infrastructure with bounded contexts involves a process of carefully evaluating each domain area, choosing technologies that are well suited to its needs, and creating a loosely coupled system that’s more manageable, scalable, and resilient. The key isn't to fit every problem into the same mould; it’s to acknowledge that different problems require different solutions.

For a more in-depth exploration of this topic, I highly recommend reading "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans; it’s essential for grasping the concepts of bounded contexts. Also, for a good understanding of the practical application of microservices and infrastructure, "Building Microservices" by Sam Newman is an excellent resource. Lastly, "Release It!" by Michael T. Nygard is important reading for understanding the architecture of resilient systems that account for failures across distributed contexts. These resources should prove useful in your further studies.
