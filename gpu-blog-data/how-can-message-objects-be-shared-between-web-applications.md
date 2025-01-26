---
title: "How can message objects be shared between web applications?"
date: "2025-01-26"
id: "how-can-message-objects-be-shared-between-web-applications"
---

Within complex web application architectures, efficient message passing is crucial for enabling communication and data sharing between disparate components. Specifically, facilitating the transfer of message objects—structured data payloads—across application boundaries necessitates careful consideration of serialization, transport protocols, and security. I’ve encountered this challenge repeatedly, particularly when integrating legacy systems with modern microservice-based applications. Based on that experience, I can provide a structured approach to tackle this.

The core issue lies in the heterogeneous nature of web applications. They might be built using different technologies, run in separate execution environments, and operate under varying security contexts. Directly sharing in-memory objects is impossible, necessitating a transformation process for transmittal. This involves serializing the object into a format suitable for transmission, transferring it across the network, and then deserializing it at the receiving end back into a usable object. This overall process is the core component of shared message object implementation.

Let's explore a common scenario: A legacy application written in PHP needs to communicate with a modern React application. The PHP application may generate a message object, for example, a user profile, structured as an associative array, while the React application relies on JavaScript objects. Directly sending the PHP array to React won’t work without some level of conversion and translation.

One practical approach to achieve this message sharing is through a well-defined API that utilizes a standardized message format such as JSON. JSON's ubiquity, human-readable structure, and ease of parsing across diverse platforms make it a prime candidate. The exchange happens over the network typically utilizing HTTP requests.

Let me illustrate with examples.

**Example 1: JSON serialization and HTTP transport**

In this scenario, a PHP backend generates a user object, which then must be sent to a web browser. The PHP code could be:

```php
<?php
// Simulate fetching user data
$user = [
  'id' => 123,
  'name' => 'John Doe',
  'email' => 'john.doe@example.com',
  'role' => 'user'
];

// Set content type to JSON
header('Content-Type: application/json');

// Convert PHP array to JSON string
echo json_encode($user);
```

Here, `json_encode()` in PHP serializes the associative array into a JSON string. The `Content-Type` header is crucial to inform the receiving party—in this case, typically a browser—that the response is JSON. This is the first half of the message sharing, preparing and encoding the message for transmission.

The JavaScript running in the React application can now retrieve the data using `fetch`:

```javascript
fetch('/api/user') // Replace with your actual API endpoint
  .then(response => response.json()) // Parse the JSON response
  .then(user => {
    console.log('Received user:', user); // Use the user object
  })
  .catch(error => console.error('Error fetching user:', error));
```

The `response.json()` method in JavaScript parses the received JSON string into a JavaScript object. This completes the process: the object created in the PHP backend is now usable in JavaScript. Error handling in the `catch` block is crucial for production environments. Note the API endpoint (`/api/user`) here should match the endpoint from the PHP server.

**Example 2: Message Queues for Asynchronous communication**

Direct HTTP requests, such as in example 1, can be problematic when a system needs to handle high volume message transfer. A more robust and scalable way for message passing is to use a message queue. Here, messages are placed on a queue by a "producer" application, and processed by a "consumer" application. This decouples producers from consumers, allowing them to operate independently and enhances scalability.

Consider a scenario where a Python backend handles user registration, then pushes these user details to be processed by an analysis server. We can use RabbitMQ to create this messaging queue system. Here's how the Python registration application can produce a message:

```python
import pika
import json

# User registration data
user_data = {
    'id': 'user-456',
    'username': 'JaneSmith',
    'email': 'jane.smith@example.com'
}
message = json.dumps(user_data)

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='user_registrations')

channel.basic_publish(exchange='', routing_key='user_registrations', body=message)

print(" [x] Sent %r" % message)

connection.close()
```

This snippet connects to a local RabbitMQ instance, declares a queue called `user_registrations`, and publishes the serialized JSON message onto that queue. The crucial part is once again the JSON serialization with `json.dumps`, turning the Python dictionary into a string for transfer. This message is now placed onto the message queue.

A separate Node.js application (consumer) can retrieve this message from the queue:

```javascript
const amqp = require('amqplib/callback_api');

amqp.connect('amqp://localhost', (error0, connection) => {
  if (error0) {
    throw error0;
  }
  connection.createChannel((error1, channel) => {
    if (error1) {
      throw error1;
    }

    const queue = 'user_registrations';

    channel.assertQueue(queue, {
      durable: false
    });

    console.log(" [*] Waiting for messages in %s. To exit press CTRL+C", queue);

    channel.consume(queue, (msg) => {
      const user = JSON.parse(msg.content.toString());
      console.log(" [x] Received:", user);
      // Process user data
    }, {
      noAck: true
    });
  });
});
```

Here, the Node.js application connects to the same RabbitMQ server and consumes messages from the same `user_registrations` queue. The `JSON.parse` function deserializes the message back to a JavaScript object. Again note that the receiving application can now use the received object as it’s own. The message is removed from the queue by the consuming application, indicating the completion of the message transfer.

**Example 3: Using Protocol Buffers for structured data**

While JSON is flexible, it can become verbose, and its schema is implicit. Protocol buffers, or protobufs, provide a more efficient and structured serialization mechanism, especially for complex data structures. It requires defining the structure of data using a `.proto` file, which then generates code for various languages that can handle the serialization and deserialization of these messages.

Let’s assume a .proto file, `user.proto`, which defines a user message:

```protobuf
syntax = "proto3";

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
}
```

The protoc compiler, which can be installed using the package manager of the operating system, can generate a corresponding class in multiple languages, for example Java using this command:
`protoc --java_out=./src/main/java/ user.proto`

Once generated this Java class can be used to serialize/deserialize data. Here's how a Java backend can serialize a user object:

```java
import com.example.UserOuterClass;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;


public class UserSerializer {

    public static void main(String[] args) throws InvalidProtocolBufferException {
        UserOuterClass.User user = UserOuterClass.User.newBuilder()
                .setId(789)
                .setName("Alice Smith")
                .setEmail("alice.smith@example.com")
                .build();

        byte[] serializedUser = user.toByteArray();
      
        String json = JsonFormat.printer().print(user);

        System.out.println(serializedUser);
        System.out.println(json);

        // Assuming data transfer over byte stream
         UserOuterClass.User deserializedUser = UserOuterClass.User.parseFrom(serializedUser);

         System.out.println(deserializedUser.getId());
         System.out.println(deserializedUser.getName());
         System.out.println(deserializedUser.getEmail());
    }
}

```

This code serializes the `user` object to a byte array using `user.toByteArray()`. This is the most compact form of transfer. The serialized byte data can now be transferred over a network stream. The example also shows how the `JsonFormat` class can convert to json format which might be more easily consumable for a receiving web client. Finally, the last part of the main method shows how the transferred byte data can be deserialized back into a usable java object.

On a node.js end, using the appropriate protobuf libraries, the same byte data can be deserialized using generated js code from the proto file. The core principal, of course is the same: message objects are serialized into a data format usable by another language and application, transmitted, and then deserialized back into an object usable in the receiving application.

**Resource Recommendations**

For a deeper understanding, consider studying the following materials:

*   **Data Serialization Formats:** Look into JSON, Protocol Buffers, and Apache Avro for comparison and practical applications.
*   **Message Queues:** Investigate RabbitMQ, Apache Kafka, and Redis Pub/Sub for understanding asynchronous messaging patterns.
*   **API Design:** Learn about REST and GraphQL for designing effective communication interfaces between applications.
*   **Network Protocols:** A basic understanding of HTTP, TCP, and UDP is important for understanding data transport mechanisms.

Implementing shared message object transfer requires a holistic perspective that encompasses message formats, transport, and application needs. The choice depends greatly on specific requirements, such as volume, latency, and data complexity. By focusing on standardization and decoupling, one can effectively enable seamless communication between web applications.
