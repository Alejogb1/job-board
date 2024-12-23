---
title: "How do you wait for all promises in an array to resolve?"
date: "2024-12-23"
id: "how-do-you-wait-for-all-promises-in-an-array-to-resolve"
---

Let's tackle this. There’s a fundamental asynchronous pattern we encounter all the time in javascript – managing a collection of promises. I’ve seen teams struggle with this, especially those coming from more synchronous backgrounds, and it’s a source of frustration until you really grasp the mechanics. You’ve got an array of promises, each doing some work, and you need to proceed only when *all* of them are complete, regardless of whether they succeed or fail. We're not talking about needing one to finish—we need them *all* done before moving forward. This isn't about individual results; it's about the collective completion signal.

The standard, and frankly, the most robust approach is utilizing `Promise.all`. It does precisely what you need: it takes an array (or any iterable) of promises and returns a *new* promise. This new promise resolves only when all the input promises have resolved successfully, or, crucially, it rejects immediately if *any* of the input promises reject. This is a key point often missed: a single failure cascades, signaling the overarching operation’s failure.

I recall a project a few years back, an e-commerce integration where we needed to pull product details from multiple microservices. Each service returned a promise for a product’s information. We couldn't display anything on the page until *all* the product details had been retrieved. It was precisely this scenario that made `Promise.all` our go-to solution. We had to handle potential network errors gracefully and `Promise.all`’s behavior – fail-fast upon any rejection – forced us to address failure conditions appropriately.

Let's illustrate with some code. Imagine we have a function, `fetchData`, that returns a promise. Each call will simulate fetching a different resource.

```javascript
function fetchData(id, success) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (success) {
        resolve(`Data for id ${id}`);
      } else {
        reject(`Failed to fetch data for id ${id}`);
      }
    }, Math.random() * 500); // Simulate varying network delays
  });
}
```

Now, let’s set up an array of promises and use `Promise.all`:

```javascript
async function processAllData() {
  const promiseArray = [
    fetchData(1, true),
    fetchData(2, true),
    fetchData(3, true),
  ];

  try {
    const results = await Promise.all(promiseArray);
    console.log("All data fetched successfully:", results);
  } catch (error) {
    console.error("Error fetching data:", error);
  }
}

processAllData();
```

In this example, if all fetches are successful, you’ll see an array of resolved values logged. But let's introduce a failure. Changing one of the `fetchData` calls to `fetchData(2, false)` simulates a failure during a fetch:

```javascript
async function processAllDataWithFailure() {
    const promiseArray = [
      fetchData(1, true),
      fetchData(2, false),
      fetchData(3, true),
    ];

    try {
      const results = await Promise.all(promiseArray);
      console.log("All data fetched successfully:", results);
    } catch (error) {
        console.error("Error fetching data:", error);
    }
  }

processAllDataWithFailure();
```

Now the catch block is executed as soon as one of the promises rejects, and the results aren't logged. Notice we used an `async` function to make consuming the promise simpler with `await`. That is the most idiomatic way to use it.

There are situations however where you might not want an operation to fail just because one of the promises rejects, you might instead want to know the outcome of *each* promise individually, regardless if it succeeded or failed. For that, we can employ `Promise.allSettled`.

Let's modify our example one more time to illustrate this, focusing on the practical advantage of using `Promise.allSettled`. Imagine an upload process, where multiple images are uploaded, and the failure of one shouldn't prevent others from being uploaded, and we want a log of each outcome:

```javascript
async function processAllDataWithSettled() {
  const promiseArray = [
      fetchData(1, true),
      fetchData(2, false),
      fetchData(3, true),
  ];

  const results = await Promise.allSettled(promiseArray);
  console.log("All data fetched, settled results:", results);
  results.forEach(result => {
      if(result.status === "fulfilled"){
          console.log(`Resolved: ${result.value}`);
      } else {
          console.log(`Rejected: ${result.reason}`);
      }
  })
}

processAllDataWithSettled();
```

Here, `Promise.allSettled` guarantees that the returned promise will *always* resolve, and it resolves with an array of objects, each representing the outcome of the corresponding input promise. Each object has a `status` property, which is either 'fulfilled' or 'rejected' along with either a `value` property or a `reason` property to contain the relevant result or error message.

Choosing between `Promise.all` and `Promise.allSettled` depends on your specific error handling needs. If any failure constitutes overall operation failure, `Promise.all` is the natural choice. If you need to track the status of each promise, irrespective of success or failure, `Promise.allSettled` is the correct approach. Consider a scenario like multiple API requests where individual failures shouldn't halt the entire operation, or batch uploads as the example showed. `Promise.allSettled` is a very powerful tool for granular control in those cases.

While `Promise.all` and `Promise.allSettled` are the cornerstones for handling multiple promises, it's beneficial to explore the underlying principles of asynchronous programming in JavaScript. I’d recommend delving into *“You Don’t Know JS: Async & Performance”* by Kyle Simpson. This book provides a detailed explanation of promises and asynchronous JavaScript fundamentals that transcends the practical implementation we’ve covered. It’s a deep dive, and it makes all the difference when dealing with more complex asynchronous workflows. In addition, the *“Eloquent JavaScript”* book by Marijn Haverbeke has a great chapter on asynchronous programming that includes a section on promises and is beneficial in solidifying your theoretical understanding. Finally, the Javascript specification (ECMA-262) provides the ultimate authority in understanding javascript behaviors, including promises and asynchronous operations, if you really want to understand the nitty gritty details.

In summary, waiting for all promises to resolve is often solved with `Promise.all` or `Promise.allSettled`. Choose `Promise.all` if any single promise rejection should cause the whole operation to fail. Use `Promise.allSettled` when you need the outcome of each promise, irrespective of success or failure. Knowing the right tool for the job, and their subtle differences, significantly impacts the stability and robustness of asynchronous operations in your code. I’ve seen this difference firsthand – understanding this core concept turns what can often be a complex, error-prone process, into a predictable and manageable part of any application.
