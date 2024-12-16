---
title: "How to use async/await instead of THEN and CATCH in Node.js?"
date: "2024-12-16"
id: "how-to-use-asyncawait-instead-of-then-and-catch-in-nodejs"
---

Let’s jump straight into this; I recall a project, circa 2017, involving a complex data pipeline that desperately needed a refactor. It was a tangled mess of `.then()` and `.catch()` blocks, making it incredibly difficult to trace the asynchronous flow of data. At that point, `async/await` was becoming quite prevalent, and frankly, it was a lifesaver. It's not just about syntactic sugar; it fundamentally improves readability and maintainability, especially when dealing with nested asynchronous operations. The move away from traditional promises and their inherent callback structures provides a much more sequential and intuitive coding style.

Let’s start with the core issue; `then` and `catch`, while functional, often lead to what’s commonly referred to as "callback hell" or "promise pyramids." The logic becomes convoluted as you chain multiple asynchronous operations. Errors are handled further down the chain, making it harder to debug. Async/await, built on top of promises, provides a synchronous-looking way to handle asynchronous operations, making the code much cleaner. Instead of working directly with promises, you’re effectively pausing execution at an `await` keyword until the promise resolves or rejects.

First, let's solidify the basic concept. Consider a common scenario: fetching data from an API. With promises, you would structure your code like this:

```javascript
function fetchDataWithPromises(url) {
  return fetch(url)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log("Data:", data);
      return data; // Return the data for further use
    })
    .catch(error => {
      console.error("Fetch error:", error);
      throw error;  // Re-throw the error for handling higher up
    });
}

fetchDataWithPromises('https://api.example.com/data');
```

Now, let's refactor that same functionality using `async/await`. It’s worth noting, that under the hood, it’s still using the same promises, just exposing a more synchronous feel.

```javascript
async function fetchDataWithAsyncAwait(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const data = await response.json();
    console.log("Data:", data);
    return data;  // Return the data for further use
  } catch (error) {
    console.error("Fetch error:", error);
    throw error; // Re-throw error for higher level handling
  }
}

fetchDataWithAsyncAwait('https://api.example.com/data');
```
Notice the `async` keyword preceding the function definition and the `await` keyword before the `fetch` and `response.json()` calls. The code reads far more like traditional synchronous code but handles asynchronous calls appropriately. The entire function is wrapped in a `try...catch` block, which replaces the `.catch` block from the original promise version. Error handling becomes much clearer. If any operation inside the `try` block throws an error (rejects the promise), the code will jump straight into the `catch` block.

The beauty of async/await extends beyond simple one-off asynchronous operations. Its effectiveness shines even brighter when dealing with sequential asynchronous tasks. Let’s take a slightly more complex example involving multiple fetches: fetching user data and then their associated posts.

```javascript
async function fetchUserDataAndPosts(userId) {
  try {
    const userResponse = await fetch(`https://api.example.com/users/${userId}`);
    if (!userResponse.ok) {
      throw new Error(`User fetch error! Status: ${userResponse.status}`);
    }
    const userData = await userResponse.json();
    console.log("User Data:", userData);

    const postsResponse = await fetch(`https://api.example.com/posts?userId=${userId}`);
    if (!postsResponse.ok) {
        throw new Error(`Posts fetch error! Status: ${postsResponse.status}`);
    }
    const postsData = await postsResponse.json();
    console.log("Posts Data:", postsData);
    return { user: userData, posts: postsData };

  } catch (error) {
    console.error("Error in user and posts fetch:", error);
    throw error;
  }
}

fetchUserDataAndPosts(123);
```

This example demonstrates how you can easily await multiple asynchronous operations sequentially within the `async` function without getting buried in `.then()` chains. The code maintains readability and error handling is contained within the `try/catch` block. This sequential flow is extremely powerful in complex application logic where the output of one asynchronous task depends on the outcome of another. This pattern drastically improves the cognitive load for developers reading and maintaining the code.

Crucially, understand that using `async/await` doesn't fundamentally change how promises work. It is built on top of them. It simply offers a cleaner, more synchronous-style API to use them. Any function marked as `async` implicitly returns a promise, meaning you can still use `.then()` and `.catch()` on it, should the need arise. This is very helpful when integrating legacy promise-based code. Also, be mindful of cases where promises can run in parallel. Using `await` inside a loop, for instance, will likely result in sequential, rather than parallel, execution, which can slow things down if not properly managed. There are ways to achieve concurrent execution while leveraging async/await through `Promise.all()` for when multiple asynchronous tasks don't depend on each other, which goes beyond the scope of this question, but is important to keep in mind.

To delve further into the intricacies of asynchronous programming in javascript, I would recommend "JavaScript: The Definitive Guide" by David Flanagan. It provides a comprehensive overview of the underlying mechanics of promises and asynchronous operations in javascript. Another invaluable resource is "Effective JavaScript" by David Herman, which provides practical guidance on improving javascript code, and it has a wonderful chapter on handling asynchronous operations correctly. Additionally, the ECMAScript specification document often details the precise language behaviors, which can be helpful when exploring the fundamental mechanics.

In my past experiences, the transition to `async/await` drastically improved the readability and maintainability of our code. Debugging complex asynchronous logic became far less stressful. While `.then()` and `.catch()` have their place, async/await should be the go-to tool for handling asynchronous logic within javascript due to its inherent clarity and ease of use. Just remember, under the surface, it's still promises at work, and understanding their behavior is essential for writing robust asynchronous code.
