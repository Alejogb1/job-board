---
title: "How do I call async functions via a PUT API?"
date: "2024-12-23"
id: "how-do-i-call-async-functions-via-a-put-api"
---

Let's tackle this head-on, shall we? The question of calling asynchronous functions within the context of a PUT api handler is a common one, and thankfully, there are several established ways to manage it. I recall a project a few years back, where we were building a user management system. We had a PUT endpoint to update user details, and several of those updates required asynchronous operations—like sending email notifications, updating cached data, or triggering downstream processes. Ignoring the asynchronous nature would have led to unresponsive apis, and that’s something we absolutely wanted to avoid.

The core issue here stems from the non-blocking nature of asynchronous operations. When you invoke an async function, it immediately returns a promise (or a similar construct), which represents the eventual result of the operation. If your api handler returns without waiting for this promise to resolve, the api might send a response prematurely, and those asynchronous side-effects might not execute fully or at all.

There are mainly three patterns that I've consistently used to handle this, each with its own set of tradeoffs. I'll walk you through them with concrete examples.

**Pattern 1: `async`/`await` within the API Handler**

This is perhaps the most straightforward approach. The API handler function itself is marked as `async`, and you use the `await` keyword to pause execution until the asynchronous functions complete. This approach makes your code look and feel almost like synchronous code, which can be easier to read and reason about.

Here's a simplified Node.js example using Express:

```javascript
const express = require('express');
const app = express();
app.use(express.json()); // For parsing application/json

async function updateUserInDatabase(userId, userData) {
    // Assume this is asynchronous and returns a promise.
    return new Promise(resolve => {
      setTimeout(() => {
          console.log(`User ${userId} updated with data:`, userData);
          resolve();
      }, 200); // Simulate a DB call.
    });
}


async function sendUserNotification(userId, userData) {
  return new Promise(resolve => {
    setTimeout(() => {
      console.log(`Notification sent to user ${userId} about update:`, userData);
      resolve();
  }, 100);
  })
}

app.put('/users/:id', async (req, res) => {
    const userId = req.params.id;
    const userData = req.body;

    try {
        await updateUserInDatabase(userId, userData);
        await sendUserNotification(userId, userData)

        res.status(200).json({ message: 'User updated successfully' });
    } catch (error) {
        console.error("Error during user update:", error);
        res.status(500).json({ error: 'Failed to update user' });
    }
});


const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

```

In this example, the `PUT /users/:id` handler is `async`. It first `await`s the completion of `updateUserInDatabase` and then `sendUserNotification`. The response is only sent after these asynchronous operations have completed, or an exception occurs. I've included a `try...catch` block here because this is critical. Async operations can fail. Catching the exceptions here and providing a meaningful response to the caller is necessary.

**Pros of using `async`/`await`:**

*   **Readability:** The code looks cleaner and is easier to understand.
*   **Error Handling:** Using `try...catch` blocks makes synchronous error handling within asynchronous operations straightforward.
*   **Simplicity:** Requires minimal setup, perfect for smaller API functions.

**Cons of `async`/`await`:**

*   **Potential Bottleneck:** If the asynchronous operations take a long time, the API handler is blocked.
*   **Complexity in Chaining Operations:** When dealing with numerous asynchronous operations in series, the code can sometimes get a bit long and might need refactoring.

**Pattern 2: Promise Chaining with `then` and `catch`**

Another approach involves utilizing promise chaining, using `.then()` to execute asynchronous functions sequentially and `.catch()` to manage errors.

Here’s the same use case converted into promise chaining:

```javascript
const express = require('express');
const app = express();
app.use(express.json());

function updateUserInDatabase(userId, userData) {
    return new Promise(resolve => {
        setTimeout(() => {
          console.log(`User ${userId} updated with data:`, userData);
          resolve();
        }, 200);
      });
}

function sendUserNotification(userId, userData) {
    return new Promise(resolve => {
      setTimeout(() => {
        console.log(`Notification sent to user ${userId} about update:`, userData);
        resolve();
    }, 100);
    })
}

app.put('/users/:id', (req, res) => {
    const userId = req.params.id;
    const userData = req.body;

    updateUserInDatabase(userId, userData)
        .then(() => sendUserNotification(userId, userData))
        .then(() => res.status(200).json({ message: 'User updated successfully' }))
        .catch(error => {
          console.error("Error during user update:", error);
          res.status(500).json({ error: 'Failed to update user' });
        });
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
```

Here, the API handler initiates the chain with `updateUserInDatabase`, and then it uses `.then` to trigger `sendUserNotification` when the first promise resolves. Finally, another `.then` sends the successful response or the `.catch` block handles any error that may occur in the chain.

**Pros of Promise Chaining:**

*   **Non-Blocking:** The handler returns immediately, making the application more responsive.
*   **Clear Flow:** The sequential nature of `.then` calls makes it easy to follow the flow of execution.

**Cons of Promise Chaining:**

*   **Error Handling Complexity:** While manageable, `.catch` blocks need to be at the end, and you need to be careful to catch errors from all prior promises.
*   **Readability:** Can become less readable and harder to debug with many chained operations, sometimes referred to as “callback hell”.

**Pattern 3: Background Job Processing (Using Message Queues)**

For scenarios involving long-running or less critical asynchronous operations, it is sometimes best to use a message queue to offload the work to a background job processor. The api handler publishes a message to a queue, and then a separate service picks up the message and performs the actual work asynchronously.

Here’s a simplified example using a hypothetical message queue library:

```javascript
const express = require('express');
const app = express();
app.use(express.json());

// Assume 'messageQueue' is your message queue implementation.
// This would vary depending on the MQ system used (e.g., RabbitMQ, Kafka).
const messageQueue = {
    publish: (queueName, message) => {
        // In a real application this would interact with a message broker.
        console.log(`Message published to ${queueName} queue:`, message)
        // Simulate a delay to make it clear its processing on a different thread.
        setTimeout(() => {
            console.log(`Background process started for`, message);
        }, 10);
    }
}

app.put('/users/:id', (req, res) => {
    const userId = req.params.id;
    const userData = req.body;

    // Publish a message to the queue for further processing.
    messageQueue.publish('user-updates', { userId, userData });

    // Immediately respond to the user
    res.status(202).json({ message: 'User update initiated (processing in background)' });
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

```

In this pattern, the api handler publishes a message to a message queue such as "user-updates", containing the user id and data that needs to be processed. The response is then immediately sent with a "202 Accepted" response code, indicating that the request has been accepted for processing but is not complete. Another worker process listening to this queue will process the message later, performing updates, sending emails and other tasks.

**Pros of using Background Jobs:**

*   **Highly Scalable:** Can distribute the load across multiple workers.
*   **Robustness:** The queue persists messages even if the worker is temporarily unavailable.
*   **Improved API Performance:** The API returns almost immediately.

**Cons of using Background Jobs:**

*   **Increased Complexity:** Requires setting up and managing a message queue infrastructure.
*   **Delayed Operations:** Asynchronous operations do not take place immediately, which may not be ideal for all use cases.
*   **Difficult to Debug:** Debugging distributed systems is more complex.

**Final Remarks:**

Choosing the approach that's most appropriate depends on the specifics of your project, performance requirements, and tolerance for complexity. For simple use cases, `async`/`await` within the API handler is often sufficient. As the number of asynchronous tasks increase, or their complexity grows, you'll find promise chaining provides a more structured approach. For truly asynchronous and time consuming operations, offloading to background processes offers the greatest benefits in terms of performance, robustness, and scalability.

For further learning, I recommend delving into the following resources:

*   **"JavaScript and JQuery: Interactive Front-End Web Development" by Jon Duckett:** This book covers promises in JavaScript in more detail, along with other fundamentals.
*   **"Node.js Design Patterns" by Mario Casciaro and Luciano Mammino:** Explores patterns for building scalable applications, including async programming and message queues in depth.
*  **Papers on Reactive Programming:** Search for "Reactive Streams" or "Reactive Manifesto". These papers give further background into dealing with asynchronous data streams. These resources have helped me in similar scenarios, and I am confident you'll find them beneficial as well.
