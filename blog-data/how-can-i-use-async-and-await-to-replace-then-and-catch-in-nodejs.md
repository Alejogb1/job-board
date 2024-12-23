---
title: "How can I use async and await to replace THEN and CATCH in NodeJS?"
date: "2024-12-23"
id: "how-can-i-use-async-and-await-to-replace-then-and-catch-in-nodejs"
---

Okay, let’s tackle this. I've definitely been in the trenches with asynchronous javascript – more times than I care to recall, actually – and the move from `.then()` and `.catch()` to `async`/`await` was a significant improvement in my workflow and the readability of my codebase. It's not just about syntactic sugar; it fundamentally alters how you reason about and debug asynchronous operations. Let me walk you through it, drawing from past projects and the pain points I've personally encountered.

The core issue stems from how promises are designed. Using `.then()` and `.catch()` introduces nested callbacks, leading to what is often called "callback hell" or "promise chaining hell". While manageable for simple cases, these structures can become incredibly difficult to follow and debug as the complexity increases. `async`/`await` was introduced precisely to mitigate these challenges by allowing us to write asynchronous code that appears more synchronous.

Essentially, `async` declares a function as asynchronous, enabling it to utilize `await`. The `await` keyword, when placed before a promise, causes the function's execution to pause until the promise settles (either resolves or rejects). Once settled, it either returns the resolved value of the promise or throws the rejected reason. This is where the crucial part comes in, it lets you treat promises much like normal, synchronous function calls.

Now, let’s get practical. I’ve had to rewrite countless legacy modules moving from pure promises to async functions. Take, for example, a module that fetches data from an api and does some basic transformation before returning it. Here’s a snippet of what it would look like with only `.then()` and `.catch()`:

```javascript
function fetchDataThenCatch(url) {
  return fetch(url)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      // Simulate some processing
      const transformedData = data.map(item => ({ ...item, processed: true }));
      return transformedData;
    })
    .catch(error => {
      console.error('Error fetching and processing data:', error);
      throw error; // Re-throwing for the caller to handle as they see fit
    });
}

fetchDataThenCatch('https://api.example.com/items')
  .then(processedData => {
      console.log("Processed data:", processedData);
  })
  .catch(error => {
    console.error("Final error handling:", error);
  });
```

This works, but you can see the nested structure which, with added complexity, quickly leads to difficulties in following the control flow. We use multiple `.then()` calls for successive operations and a dedicated `.catch()` block to handle errors across the chain.

Now, here’s how that same functionality looks with `async` and `await`:

```javascript
async function fetchDataAsyncAwait(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const data = await response.json();
    // Simulate some processing
    const transformedData = data.map(item => ({ ...item, processed: true }));
    return transformedData;
  } catch (error) {
     console.error('Error fetching and processing data:', error);
      throw error;
  }
}

async function main(){
  try{
    const processedData = await fetchDataAsyncAwait('https://api.example.com/items');
    console.log("Processed data:", processedData);
  }
  catch(error){
    console.error("Final error handling:", error);
  }
}

main();
```

Notice the difference? Instead of chaining promises, we are now using a `try...catch` block, which is more akin to traditional error handling you would find in synchronous code. Each `await` makes it seem like you're waiting for the previous line to complete, making the code easier to read and follow logically. The error handling is also simplified - the `catch` block manages errors that occurred at any point within the associated `try` block.

The key here is that `await` only works inside an `async` function. It’s an error if you try to use it outside. Also note that an `async` function implicitly returns a promise. So, the caller of `fetchDataAsyncAwait` receives a promise which resolves to the return value of the function or rejects with the thrown error.

Finally, let's demonstrate an example with more complex error handling scenarios. Imagine that in the previous api request example, a specific kind of error needs different handling: for instance, if the api is down we might want to retry, but if it’s a different kind of error, we need to log it and move on:

```javascript
async function fetchDataRetry(url, retries = 3) {
  let attempts = 0;
  while (attempts < retries) {
    try {
      const response = await fetch(url);
       if (!response.ok) {
          if (response.status === 503) { //Service Unavailable
              attempts++;
              console.log(`Service unavailable, retry attempt: ${attempts}`);
              await new Promise(resolve => setTimeout(resolve, 1000 * attempts)); // Exponential backoff
              continue;
            }
         throw new Error(`HTTP error! Status: ${response.status}`);
       }
      const data = await response.json();
      return data;

    } catch (error) {
        console.error(`Error during request: ${error}`);
        if (attempts < retries) {
           attempts++;
            console.log(`Request failed, retry attempt: ${attempts}`);
            await new Promise(resolve => setTimeout(resolve, 1000 * attempts)); // Exponential backoff
            continue;
        }

        throw error;
    }
  }
    throw new Error("Request failed after max retries.");
}


async function main() {
    try {
      const fetchedData = await fetchDataRetry('https://api.example.com/items');
      console.log('Data fetched:', fetchedData);
    } catch (error) {
      console.error('Final Error:', error);
    }
  }

main();
```

This example shows that `async`/`await`, coupled with a `while` loop, makes it easier to handle more complicated cases, such as retries. The `try...catch` block allows us to handle errors granularly, decide on retry logic within the loop, and use `continue` when appropriate. If no more retries are allowed, it exits the loop by throwing an exception, which is then caught in the caller context `main()` function. This is fundamentally easier to reason about compared to attempting the same with multiple `.then` calls and a very complicated chain of `.catch` handlers.

As for further learning, I’d strongly recommend "Effective Javascript" by David Herman. It covers asynchronous programming in detail and goes deeper into javascript's inner workings. Also, for understanding javascript's runtime and event loop, check out "You Don't Know JS" series by Kyle Simpson. Specifically, the "Async & Performance" volume is extremely helpful. The MDN web docs on Promises and Async functions are of course essential resources for day to day development.

In closing, moving from `.then()` and `.catch()` to `async`/`await` will simplify your asynchronous javascript code. It facilitates readability and reduces the complexity of both control flow and error handling by allowing you to write code in a way that is much closer to synchronous coding patterns. This translates to fewer bugs, easier debugging and quicker development cycles.
