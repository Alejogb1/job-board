---
title: "How to use async/await instead of THEN/CATCH in Node.js?"
date: "2024-12-23"
id: "how-to-use-asyncawait-instead-of-thencatch-in-nodejs"
---

Okay, let's tackle this. It’s a shift that, looking back, drastically cleaned up my code. I vividly recall a project back in 2017 – a distributed service relying heavily on chained promises. The error handling was a nightmare, each `then` followed by another, and nested `catch` blocks creating a labyrinth of callbacks. That’s when I really started appreciating the syntactic sugar that `async/await` brought to the table.

The fundamental issue with `then/catch` (or, more specifically, promise chaining) is that it doesn't always read sequentially, even though the code executes sequentially. It can make control flow and error propagation hard to follow, especially when you're dealing with asynchronous operations that depend on each other. `async/await` allows us to write asynchronous code that looks and behaves more like synchronous code. This dramatically improves readability and makes debugging considerably simpler.

Essentially, `async` marks a function as asynchronous. It implicitly returns a promise. Inside an `async` function, `await` pauses execution until a promise resolves (or rejects). Think of it as a syntactic pause, letting the event loop do its magic. The beauty is, instead of `.then` chaining, you can just assign the resolved value directly to a variable. Error handling, instead of scattered `catch` blocks, becomes more conventional using `try...catch` which is something we're typically very familiar with.

Here’s a simple illustration. Let’s assume we have a function, `fetchData`, which returns a promise that resolves with some data:

```javascript
function fetchData(url) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (url === 'good_url') {
        resolve({ data: 'some data from server' });
      } else {
        reject(new Error('Invalid url'));
      }

    }, 500);
  });
}
```

Now, let's look at how to consume this promise using `then/catch`:

```javascript
function getDataWithThenCatch(url) {
  console.log('Starting fetch with then/catch');
  fetchData(url)
    .then(result => {
      console.log('Data received:', result.data);
      return result.data;
    })
    .then(data => {
      console.log('Processed data:', data.toUpperCase());
    })
    .catch(error => {
      console.error('An error occurred:', error.message);
    });

    console.log('then/catch initiated');
}

getDataWithThenCatch('good_url');
getDataWithThenCatch('bad_url');
```

This approach works, but notice the chained `.then` calls and how the final `catch` block has to handle all errors from the promise chain. The `console.log('then/catch initiated')` will occur before the data is returned. It's not always immediately obvious which `then` is failing at a glance.

Let's contrast that with the `async/await` version:

```javascript
async function getDataWithAsyncAwait(url) {
  console.log('Starting fetch with async/await');
  try {
    const result = await fetchData(url);
    console.log('Data received:', result.data);
    const processedData = result.data.toUpperCase();
    console.log('Processed data:', processedData);
    return processedData; // Return if desired

  } catch (error) {
    console.error('An error occurred:', error.message);
  }
  console.log('async/await initiated');
}


getDataWithAsyncAwait('good_url');
getDataWithAsyncAwait('bad_url');
```

The difference is stark. The code using `async/await` is more straightforward and easy to reason about. Error handling is neatly contained within the `try...catch` block. You can follow the code as if it were synchronous, step by step. Again, the `console.log('async/await initiated')` will also occur before data is received which is one of the common misconceptions when starting with `async/await`.

Now, let's tackle a more complex example. Suppose we need to fetch user data, then based on that user data fetch additional related information. With promises, it becomes intricate. Let’s add a fictional `fetchRelated` function:

```javascript
function fetchRelated(userId) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
          if(userId === 123){
            resolve({ relatedData: 'related data for user 123'});
          } else {
            reject(new Error('User id not found'));
          }
        }, 500)
    });
}
```

Using `then/catch`:

```javascript
function getUserDataWithThenCatch(userId) {
    console.log("then/catch getUserData start");
    fetchData('good_url')
        .then(user => {
            console.log(`User data received. Proceeding to fetch related info: ${user.data}`);
            return fetchRelated(userId);
        })
        .then(related => {
            console.log('Related data:', related.relatedData);
        })
        .catch(error => {
            console.error('An error occurred:', error.message);
        });
    console.log("then/catch getUserData initiated");
}

getUserDataWithThenCatch(123);
getUserDataWithThenCatch(456);
```

And now, the `async/await` alternative:

```javascript
async function getUserDataWithAsyncAwait(userId){
  console.log('async/await getUserData start');
    try{
        const user = await fetchData('good_url');
        console.log(`User data received. Proceeding to fetch related info: ${user.data}`);
        const related = await fetchRelated(userId);
        console.log('Related data:', related.relatedData);
    } catch (error) {
        console.error('An error occurred:', error.message);
    }
    console.log("async/await getUserData initiated");
}

getUserDataWithAsyncAwait(123);
getUserDataWithAsyncAwait(456);
```

See how `async/await` simplifies the flow? The code reads very much like synchronous code with the `await` keyword pausing the execution until the result is available. This clarity makes it much easier to trace the execution path and diagnose issues. The `try/catch` block is still very useful in this example, ensuring a single location handles exceptions from both `fetchData` and `fetchRelated`.

For further learning, I'd suggest looking into "Effective JavaScript" by David Herman, particularly the chapters related to promises and asynchronous programming. This book provides very solid foundations. Also, the JavaScript specification documentation on async functions and promises is invaluable and can provide you with the most current details. Additionally, understanding the mechanics of the event loop – and for that, I’d recommend "Node.js Design Patterns" by Mario Casciaro and Luciano Mammino – is crucial to fully grasp the asynchronous nature of these concepts.

Ultimately, moving to `async/await` over `then/catch` isn’t just about writing less code, it's about writing code that is more expressive, less error-prone, and significantly easier to debug and maintain. From my experience, that makes the shift more than worthwhile.
