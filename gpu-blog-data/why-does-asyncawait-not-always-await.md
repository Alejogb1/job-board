---
title: "Why does Async/Await not always await?"
date: "2025-01-30"
id: "why-does-asyncawait-not-always-await"
---
The common misconception is that the `async`/`await` keywords inherently guarantee synchronous execution within an asynchronous context. I've spent a significant portion of my career debugging production systems where this misunderstanding manifested as subtle, yet critical, concurrency errors. The core issue lies not with the `async`/`await` syntax itself, but rather with what is actually being awaited—specifically, that `await` only pauses execution of the *current* async function until the awaited promise resolves or rejects. It doesn't magically force synchronous behavior onto the underlying asynchronous operations, and understanding this distinction is crucial for writing robust asynchronous code.

To unpack this, consider that `async` functions implicitly return a promise. When you use `await` inside an async function, you are essentially yielding control back to the JavaScript event loop while that promise is pending. JavaScript continues to process other tasks and callbacks, before re-entering the async function to execute the remaining code after the promise resolves. This asynchronous nature remains regardless of how quickly that promise might eventually settle. The primary purpose of `await` is to write asynchronous code that looks more synchronous, improving readability, not to transform its inherent asynchronous behavior.

Furthermore, not every operation you might expect to be asynchronous actually returns a promise. For example, the `.forEach()` method on an array doesn't, and using `async`/`await` within its callback provides no actual pause. It merely creates and executes asynchronous callbacks that run *after* the forEach method completes, possibly in an unpredictable order. Such cases can lead to race conditions and unpredictable results if not carefully managed.

Let’s look at a few concrete code examples to illustrate these points. First, we will examine the correct usage of `async`/`await` with a promise-returning function:

```javascript
async function fetchData() {
    console.log("Fetching data...");
    // Simulating a network request that resolves after 1 second
    const dataPromise = new Promise((resolve) => {
        setTimeout(() => {
            resolve("Data retrieved!");
        }, 1000);
    });

    console.log("Waiting for data...");
    const data = await dataPromise;
    console.log(data);
    return data;
}

async function main() {
    console.log("Starting main function");
    const result = await fetchData();
    console.log("Result from fetchData:", result);
    console.log("Finished main function");
}

main();

// Output:
// Starting main function
// Fetching data...
// Waiting for data...
// (1 second pause)
// Data retrieved!
// Result from fetchData: Data retrieved!
// Finished main function
```

In this example, the `fetchData` function simulates fetching data with a promise that resolves after one second. The `await dataPromise` line pauses the `fetchData` function until the promise resolves. This allows the "Fetching data..." and "Waiting for data..." statements to be printed, followed by the “Data retrieved!” statement after the one second delay, followed finally by the return value and the final “Finished main function” log in main. The key is that `await` correctly handled the asynchronous nature of the `dataPromise`. Note the program did not block until the promise is resolved. Other tasks within the event loop can continue. The `await` does not make the overall process synchronous, only the inner asynchronous function.

The second example shows how improper usage with a non-promise-returning function (specifically a forEach loop) might lead to unexpected behavior.

```javascript
async function processItems() {
  const items = [1, 2, 3];
  console.log("Starting item processing...");

  items.forEach(async (item) => {
    console.log(`Processing item: ${item}`);
      // Simulate work with a delay of 200ms
    await new Promise(resolve => setTimeout(resolve, 200));

    console.log(`Finished processing item: ${item}`);
  });

  console.log("Finished forEach loop.");

  //Additional logic here
    await new Promise (resolve => setTimeout(resolve, 500))
    console.log("Finished function processing.");

}

processItems();

// Output:
// Starting item processing...
// Finished forEach loop.
// Processing item: 1
// Processing item: 2
// Processing item: 3
// (200ms delay)
// Finished processing item: 1
// (200ms delay)
// Finished processing item: 2
// (200ms delay)
// Finished processing item: 3
// (500ms delay)
// Finished function processing.
```

Here, the `forEach` method does not "wait" for each async callback to complete. The "Finished forEach loop." message will be printed before any of the items finish processing. The `await` statements inside each callback only pause that callback's execution, not the `forEach` loop itself. `forEach` completes before any of the inner promises. The asynchronous processing is happening but not in the way someone unfamiliar with asynchronicity would expect. This could lead to unexpected results if any subsequent code depends on all the forEach tasks to be finished.

For the correct asynchronous loop management, one should use `for...of` loops or `Promise.all`, this next example shows how the use of `Promise.all` would help in this case.

```javascript
async function processItemsCorrectly() {
  const items = [1, 2, 3];
  console.log("Starting item processing...");

  const promises = items.map(async (item) => {
    console.log(`Processing item: ${item}`);
    await new Promise(resolve => setTimeout(resolve, 200));
    console.log(`Finished processing item: ${item}`);
    return item * 2;
  });

  const results = await Promise.all(promises);

    console.log("All items processed");
  console.log("Results:", results);

  // Additional logic
    await new Promise (resolve => setTimeout(resolve, 500))
    console.log("Finished function processing.");
}

processItemsCorrectly();

// Output:
// Starting item processing...
// Processing item: 1
// Processing item: 2
// Processing item: 3
// (200ms delay)
// Finished processing item: 1
// (200ms delay)
// Finished processing item: 2
// (200ms delay)
// Finished processing item: 3
// All items processed
// Results: [ 2, 4, 6 ]
// (500ms delay)
// Finished function processing.
```

By utilizing `Promise.all`, we create an array of promises and wait for all of them to complete before continuing. This enables us to manage the asynchronous processing correctly while also returning the results of each item's processing.  This ensures that processing starts for all items, and then the function pauses, only continuing once all item processing is completed. Finally, the function continues and logs its results as expected.

In summary, the reason `async`/`await` might not appear to always “await” is because it is awaiting promises, not blocking the event loop. It pauses execution of the *current* `async` function while awaiting the resolution or rejection of a promise. The underlying asynchronous tasks continue to execute in the background, and the event loop is not blocked. Furthermore, it is crucial to understand which operations actually return promises to avoid misusing `async`/`await` (as illustrated in the foreach example). Using it with non-promise-returning callbacks will not lead to the expected behaviour. Use `for...of` or `Promise.all` to manage sequences of operations which depend on each other.

For further understanding, I strongly advise a deep dive into event loop mechanics and the concepts of promises. Books focusing on JavaScript's asynchronous behavior, such as those by Kyle Simpson, provide in-depth explanations. Practical experience is essential; building and debugging asynchronous applications will further solidify your understanding. Online documentation provided by Mozilla Developer Network (MDN) offers comprehensive details on `async`/`await` and related topics. Mastering these concepts is essential for building efficient and reliable asynchronous Javascript applications.
