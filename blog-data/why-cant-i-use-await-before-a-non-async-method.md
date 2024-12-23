---
title: "Why can't I use `await` before a non-async method?"
date: "2024-12-23"
id: "why-cant-i-use-await-before-a-non-async-method"
---

,  It's a question that surfaces quite frequently, and I've certainly seen my fair share of confusion around it, especially during those early adoption phases of async/await in various projects. From what I've observed over the years, the root cause often stems from a misunderstanding of the fundamental mechanics underpinning asynchronous operations and how `async/await` integrates with that framework.

The core issue is that `await` is a syntactic construct intrinsically tied to asynchronous functions, specifically those declared using the `async` keyword in javascript (and similar in other languages, though I'll focus on javascript for clarity). It's not a generic command that can be used with any method call, but rather a specialized operator designed to pause execution within an `async` function until a promise is resolved or rejected. The `async` keyword, in turn, transforms the function into a state machine, essentially handling the complexities of yielding control back to the event loop while waiting for the underlying asynchronous operation to complete.

Now, why does this matter? Well, if you attempt to use `await` before a non-async function, you're effectively asking the system to pause the execution of the current function based on a process it doesn't recognize as asynchronous. A non-async function returns a value synchronously, not a promise. Thus, there's nothing for `await` to… well… await. It expects a promise, a signal that something asynchronous is happening. Think of it as trying to plug a three-pronged plug into a two-hole outlet – the mechanics just don't match up. The javascript engine, being intelligent (for the most part), spots this mismatch and throws a syntax error.

In practical terms, the absence of the `async` keyword means the function will return either a value directly or will execute its code synchronously within the confines of the call stack. It won't, and can't, return a promise that `await` can monitor. If that returned value happens to be, say, an integer or a string, `await` essentially has nothing to look for, which isn't its intended role. Trying to use `await` there is, put bluntly, a logical fallacy within the async/await paradigm.

Let me illustrate this with a few code snippets. First, an example demonstrating why you **cannot** use `await` before a synchronous function:

```javascript
function syncFunction(value) {
  return value * 2;
}

async function asyncFunction() {
  try {
    // This will throw a syntax error: "await is only valid in async functions"
    const result = await syncFunction(5);
    console.log("Result:", result);
  } catch (error) {
    console.error("Error:", error);
  }
}

asyncFunction(); // will throw error
```

Here, `syncFunction` is a plain javascript function. It executes and returns a value immediately. The attempt to `await` that return leads to the described error, highlighting that `await` cannot interact with synchronous operations directly.

Now, let’s see how a properly setup async function using `await` would work:

```javascript
function asyncOperation(value) {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve(value * 2);
    }, 500); // Simulating an async delay
  });
}

async function correctAsyncFunction() {
  try {
    // This works as it awaits the promise returned by `asyncOperation`
    const result = await asyncOperation(5);
    console.log("Result:", result); // will output after 500ms
  } catch (error) {
    console.error("Error:", error);
  }
}

correctAsyncFunction();
```

In this snippet, `asyncOperation` now returns a promise. This promise resolves after a small delay (simulating I/O or other asynchronous actions). In `correctAsyncFunction`, we properly use `await` to pause execution until that promise has resolved, and the `result` becomes available. This demonstrates the correct usage of async/await, waiting on a promise that represents an asynchronous process.

Finally, let's demonstrate how you might need to refactor existing code for async/await compatibility. Suppose, for some reason, you are stuck with an old library that uses callbacks:

```javascript
function legacyAsyncOperation(value, callback) {
  setTimeout(() => {
    callback(null, value * 2); // simulate node-style callback
  }, 500)
}

async function wrapLegacy() {
  return new Promise((resolve, reject) => {
    legacyAsyncOperation(5, (err, data) => {
        if (err) {
            return reject(err);
        }
        resolve(data);
    });
  });
}

async function useWrappedLegacy() {
  try {
      const result = await wrapLegacy();
      console.log("Result:", result);
    } catch (error) {
    console.error("Error:", error);
  }
}

useWrappedLegacy();
```

In the above code, we have a legacy asynchronous function that uses a callback pattern. We cannot use await directly on this function, therefore a wrapper around it that returns a promise is needed before we can integrate it correctly with async/await.

It's essential to understand that the `async` keyword doesn't *make* a function asynchronous. It merely facilitates the use of `await` within that function and ensures that the function *always* returns a promise. The actual asynchronous operations must still be implemented using promises, callbacks, or other similar constructs.

For anyone delving deeper into this, I'd highly recommend the 'You Don't Know JS' series, particularly the section on 'Asynchronous JavaScript', by Kyle Simpson. It provides a robust explanation of the underlying mechanisms. Also, the EcmaScript specification documents, which define how JavaScript works, will help to have a more nuanced understanding of async/await.

Additionally, "Effective JavaScript" by David Herman offers very practical patterns and tips. By engaging with these resources and understanding the fundamental principles of how promises work, the error of attempting to `await` a synchronous function will become clear, and the benefits of a well-structured asynchronous code becomes more apparent. Hopefully this illuminates the situation.
