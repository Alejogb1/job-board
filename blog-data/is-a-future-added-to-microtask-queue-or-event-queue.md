---
title: "Is a Future added to MicroTask queue or Event queue?"
date: "2024-12-16"
id: "is-a-future-added-to-microtask-queue-or-event-queue"
---

Alright, let's tackle this one. I remember vividly a particularly challenging debugging session a few years back, where understanding the nuances of microtask and event queues was absolutely critical. We were dealing with a complex asynchronous data pipeline involving multiple promises and network requests, and the behavior of the ui seemed...well, unpredictable. That experience, more than anything, burned into my brain exactly how futures and their associated asynchronous operations are scheduled.

So, the short answer to your question: a `Future` *itself* is not directly added to either the microtask or event queue. Instead, the *operations resulting from* the completion of a `Future`, typically continuations like `.then()` or `async/await` blocks, are what get scheduled, and these are scheduled onto the microtask queue. Let's unpack that a bit because it's crucial to understand the subtle distinctions.

Think of a `Future` as a promise—a placeholder for a value that might not be available immediately. It represents an eventual result of an asynchronous operation. The key thing is that a `Future`'s operation, like a network call or a file read, typically originates from the event loop, but the *notification* of its completion and subsequent handling are what's orchestrated through the microtask queue.

Here's a more concrete picture. The event loop is the broader mechanism that handles all external events - user interactions, network activity, etc. When an asynchronous operation initiated by a future completes, say, a server sends back a response to a http request, that event goes through the event loop. This event then triggers the resolution or rejection of the associated future, and this resolution or rejection is followed by calling any `.then` clauses or `async/await` blocks. These callbacks are then pushed onto the microtask queue and processed before the event loop moves onto the next tick, ensuring a kind of ‘prioritized’ processing for these asynchronous actions.

To clarify further, the event queue is primarily for external events that the browser is *waiting* for – things like user clicks, network responses, or timers expiring. These cause 'full' event loop iterations. The microtask queue, in contrast, acts as a buffer of more fine-grained, 'high-priority' asynchronous operations. After each 'full' event loop iteration, the browser first goes to microtask queue and handles all jobs within. Because of this, microtasks have priority. They effectively interweave between full event loop iterations. Futures are critical in utilizing this microtask queue effectively.

Let’s look at some illustrative code snippets.

**Snippet 1: Simple Promise Resolution**

```javascript
console.log("Start");

const myPromise = new Promise((resolve) => {
  console.log("Promise constructor");
  resolve("Resolved value");
});

myPromise.then((value) => {
  console.log("Promise then:", value);
});

console.log("End");
```

In this example, the output will be:

```
Start
Promise constructor
End
Promise then: Resolved value
```

The `Promise constructor` logs synchronously, since the constructor body runs immediately. However, the `.then()` callback doesn't get executed right away. Instead, it's scheduled as a microtask. Therefore, `End` logs before the `Promise then` callback which is handled within the microtask queue. This clearly shows that the resolution callback from a promise (a similar construct to future) is queued as a microtask.

**Snippet 2: `async`/`await` Function**

```javascript
async function fetchData() {
  console.log("fetchData start");
  await new Promise((resolve) => setTimeout(resolve, 0));
  console.log("fetchData after await");
  return "Data fetched";
}

console.log("Script start");
fetchData().then((data) => console.log("fetchData resolved:", data));
console.log("Script end");
```

The output is:

```
Script start
fetchData start
Script end
fetchData after await
fetchData resolved: Data fetched
```

Here, the `await` in `fetchData` effectively yields control back to the event loop, although the timer is set for 0. This effectively moves the rest of the `fetchData` function to the microtask queue through the Promise it creates behind the scenes. Observe how "Script end" is printed before the log after await in `fetchData`, and eventually, before the result of `fetchData` promise. Again, it’s the operations *after* the awaited operation, implicitly scheduled as a microtask via the compiler transforms, that are added.

**Snippet 3: Chained `then` with nested Promise**

```javascript
console.log("Start main");

Promise.resolve()
  .then(() => {
    console.log("First then");
    return new Promise((resolve) => {
      console.log("Inner promise constructor");
      resolve("inner resolved")
    }).then(val=> {
      console.log('Inner promise then', val)
    });
  })
  .then(() => {
    console.log("Second then");
  });

console.log("End main");

```

Here, the order of operations is crucial. The output is:

```
Start main
Inner promise constructor
End main
First then
Inner promise then inner resolved
Second then
```

The outer `.then` is queued as microtask. Within this outer `.then`, we create an inner promise and resolve it with its own `.then`. Notice how these resolve after "End main", meaning they are scheduled via the microtask queue. The key point to extract here is how the chained promise continuations and their resolution handlers are placed into microtask queue, in a linear order.

To dig deeper into the specifics of the event loop and microtask scheduling, I’d strongly suggest delving into Jake Archibald’s “Tasks, microtasks, queues and schedules” talk from JSConf.eu. It remains a definitive source. Also, the "HTML Living Standard" document, specifically the section on "event loops", provides the formal definitions and specifications. Additionally, for a book-length treatment of JavaScript's asynchronous programming model, I recommend "You Don't Know JS: Async & Performance" by Kyle Simpson. These are resources that I’ve found extremely helpful over the years.

In conclusion, a `Future`'s associated asynchronous operation itself utilizes the browser's event loop, while the continuations triggered by the `Future`'s resolution or rejection are scheduled onto the microtask queue. It’s this distinction that makes precise control over the order of asynchronous events possible, and critical when building responsive user interfaces. The microtask queue effectively prioritizes updates and completion handlers over other event loop activities. Understanding this interplay is absolutely essential for writing efficient and predictable asynchronous code.
