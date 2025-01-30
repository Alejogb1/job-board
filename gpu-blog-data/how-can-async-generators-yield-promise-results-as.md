---
title: "How can async generators yield promise results as they resolve?"
date: "2025-01-30"
id: "how-can-async-generators-yield-promise-results-as"
---
Asynchronous generators in JavaScript, introduced in ES2018, provide a powerful mechanism for handling sequences of asynchronous values. Specifically, the capacity to yield promises from an asynchronous generator and have them effectively resolve before the generator moves to the next `yield` point is central to their utility. This behavior leverages the inherent nature of promises to signal the completion of an asynchronous operation. Essentially, when an async generator yields a promise, the generator's execution pauses until that promise either resolves or rejects. Upon resolution, the resolved value becomes the value of the `yield` expression.

I’ve personally leveraged this functionality numerous times in projects dealing with data streams from APIs or large datasets that required paginated loading. One scenario involved ingesting data from a social media API that imposed rate limits and required multiple requests. Async generators provided an elegant solution to process the API response pages sequentially, waiting for each page to load before moving to the next.

To delve deeper, let's examine the mechanics. An async generator function, designated with the `async function*` syntax, returns an async generator object. This object implements the asynchronous iterator protocol. When the `next()` method of this iterator is called, the generator executes until it encounters a `yield` statement. If the yielded value is a promise, the `next()` call will return a promise that resolves with an object containing the resolved value and a `done` flag indicating the generator state (whether more yields are expected).

The core aspect to comprehend is the inherent behavior of `await` within an async generator. The `yield` keyword, when followed by a promise, implicitly awaits its resolution. The generator pauses execution and the `next()` call's promise will remain pending until the yielded promise is resolved or rejected. This automatic waiting is what makes async generators particularly useful for sequences of asynchronous operations.  If the yielded promise rejects, the generator throws the rejection error, making error handling a key concern.

Let's illustrate this with some code examples.

**Example 1: Fetching Data from Multiple API Endpoints**

```javascript
async function* fetchMultiple(urls) {
  for (const url of urls) {
    console.log(`Fetching from ${url}`);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    yield data;
    console.log(`Data from ${url} fetched and yielded`);
  }
}

async function main() {
  const urls = [
    'https://jsonplaceholder.typicode.com/todos/1',
    'https://jsonplaceholder.typicode.com/todos/2',
    'https://jsonplaceholder.typicode.com/todos/3',
  ];

  for await (const item of fetchMultiple(urls)) {
    console.log("Processing:", item);
  }
}
main();
```

In this snippet, `fetchMultiple` is an async generator that iterates through a list of URLs. Inside the loop, it uses `fetch` to make an HTTP request and, crucially, awaits the promise returned by `response.json()`. It then yields the JSON data which was only available after resolution of promises. The main function then uses an async `for...await...of` loop to consume the yielded values sequentially. Each `yield` produces a promise that is awaited by the `for...await...of` structure. Note the console logs illustrate the asynchronous nature of the execution flow, with fetches and processing interleaving correctly as each promise is fulfilled. The error handling is rudimentary but shows the principle of catching rejections from a network request.

**Example 2:  Simulating Asynchronous Operations with `setTimeout`**

```javascript
function delay(ms, value) {
  return new Promise(resolve => {
    setTimeout(() => resolve(value), ms);
  });
}

async function* asyncOperationSimulator() {
  console.log("Starting async operations");
  yield delay(500, "Operation 1 Complete");
  console.log("Operation 1 yielded");
  yield delay(1000, "Operation 2 Complete");
  console.log("Operation 2 yielded");
  yield delay(750, "Operation 3 Complete");
  console.log("Operation 3 yielded");
}


async function main() {
    for await(const result of asyncOperationSimulator()){
        console.log("Received result:", result);
    }
    console.log("Finished consuming the generator");
}
main()
```

Here, the `asyncOperationSimulator` generator simulates asynchronous processes using `setTimeout` to represent a non-blocking delay. Each `yield` represents a step in the process. Note that the  `console.log` statements and their positioning before and after the yields helps make it explicit when values are yielded by the async generator, and when the consumer receives them. The `for await...of`  loop consumes the results, pausing execution at each iteration to wait for the corresponding promise to resolve. The output shows how code following the `yield` statement waits until the `delay` promise resolves before proceeding.

**Example 3: Handling Potential Errors**

```javascript
async function* errorProneAsyncGenerator() {
  yield Promise.resolve("Initial value");
    try {
        console.log("Starting operation with potential error.");
        const result = await Promise.reject("Error occurred in async process");
        yield result; //This line won't be reached on rejection
    }
    catch (error){
        console.error("Caught an error", error);
        yield "Recovered value after error"
    }
  yield Promise.resolve("Final Value");

}

async function main() {
  for await (const value of errorProneAsyncGenerator()) {
    console.log("Value received:", value);
  }
    console.log("Generator iteration completed");
}
main();

```

This example illustrates exception handling within an async generator. The `errorProneAsyncGenerator` yields a successful promise, then attempts a promise that will reject. The `try...catch` block within the generator gracefully handles the rejection, logs the error, and yields a recovery value.  This demonstrates how errors occurring within promises yielded by the async generator can be caught and handled internally, without causing the consumer iteration to crash. Again, the `for await` construct naturally handles the promises by awaiting their completion.

In terms of resource recommendations to deepen understanding, I'd suggest exploring documentation related to asynchronous iterators and generators found in MDN Web Docs. Furthermore, engaging with more intricate examples in articles and tutorials that explore practical use-cases for async generators is incredibly beneficial, such as handling streaming data or coordinating workflows involving asynchronous operations. Pay close attention to the interplay between `async` and `await` within the context of generators.  Experimenting by creating your own generators is essential for cementing your understanding.  Finally, examine JavaScript design pattern books that discuss asynchronous programming techniques and see how async generators might improve asynchronous flows.
