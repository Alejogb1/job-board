---
title: "Is a Future added to the MicroTask queue or the Event Queue?"
date: "2024-12-23"
id: "is-a-future-added-to-the-microtask-queue-or-the-event-queue"
---

Let’s tackle this one; it's a recurring point of confusion and one that, frankly, I’ve debugged more than a few times in my career. The question of where a `Future` lands—microtask queue or event queue—isn’t quite as straightforward as one might initially think. It's a nuance that developers often grapple with, and understanding it is fundamental for crafting robust asynchronous code, especially in JavaScript-based environments like browsers or Node.js.

The core issue arises from how JavaScript engines handle asynchronous operations. We have, broadly speaking, two types of queues designed to manage these tasks: the event queue (or macrotask queue) and the microtask queue. The event queue deals with macroscopic events like browser rendering, network callbacks, or `setTimeout` calls. On the other hand, the microtask queue is for more immediate, often internally triggered asynchronous tasks like promise resolutions and mutation observer callbacks. This distinction, although conceptually clear, can blur when you bring `Futures` into the mix.

In practice, a `Future` itself isn't directly placed into either queue. Instead, it’s more accurate to say that the actions initiated *by* the `Future` are what end up in these queues. Specifically, it's the *completion* of the `Future` (either successful resolution or rejection) that schedules microtasks. This distinction is significant: a `Future` represents a future computation, not an immediate task.

My first brush with this was when I was working on a complex data processing pipeline in a Node.js backend. We were using custom promise-like objects (long before the native implementation was widely adopted), and things would occasionally execute out of order. This led me down a rabbit hole of queue analysis and debugging. What I realized then, and I’ve seen repeated since, is that `Futures` usually resolve to promises (or something promise-like), and it's the promise machinery that then leverages the microtask queue for their callbacks (the `then()` and `catch()` methods).

Think of it this way: you don’t directly put a `Future` object into the queue; you provide it with work to do, and the result of that work will be channeled to the appropriate queue. The `Future` itself is more of a container holding an eventual value or an error and scheduling that resolution through the microtask queue. The scheduling action itself is synchronous, but the execution of that scheduled task is where the asynchrony lies.

Let’s illustrate with a few examples:

**Example 1: Direct Promise Resolution**

```javascript
console.log("start");

const future = Promise.resolve("future result");

future.then((result) => {
    console.log("Future resolved:", result);
});

console.log("end");
```

Here's what happens: "start" is logged, then the `future` (a resolved Promise) is created. The `then` handler is *immediately* registered, and the resolution of the promise schedules a microtask to call the success handler (the `console.log("Future resolved:", result)` part). “end” is logged. Finally, the microtask is dequeued and executed, producing "Future resolved: future result." Crucially, this resolution happens *after* the current synchronous JavaScript execution context has finished, and *before* the engine can process any new events from the event queue. The key point is that it is not the future itself, but its resolution which produces a microtask.

**Example 2: Future Resolution with a setTimeout**

```javascript
console.log("start");

let resolved;

const future = new Promise(resolve => {
    setTimeout(() => {
        resolve("future result via setTimeout");
        resolved = true;
        console.log("resolve function called")
    }, 0)
});

future.then((result) => {
    console.log("Future resolved:", result);
});


console.log("end");

```

This example is a bit more complex. Here, we wrap a `setTimeout` within a Promise. Initially, "start" is printed and then the promise setup happens. The asynchronous `setTimeout` function is queued in the event queue. "end" is immediately printed. When `setTimeout`’s timer expires (even if it is 0 ms), the callback is pushed to the event queue and is executed when the JavaScript engine is free. It will then resolve the Promise, which in turn schedules a microtask via the `then` method to process the resolved value, resulting in "Future resolved: future result via setTimeout" being printed. We see that "resolve function called" will be logged before "Future resolved..." because that is the sequence of execution of the callbacks from both queues. This showcases how the event queue triggers other microtask queue items.

**Example 3: Future Resolution with a custom async operation**

```javascript

function customAsyncOperation() {
  return new Promise(resolve => {
    console.log("Starting async operation")
    setTimeout(() => {
      console.log("Custom async operation complete");
      resolve("Custom async result");
    }, 100);
  });
}

async function processFuture() {
    console.log("Process started");
    const result = await customAsyncOperation();
    console.log("Future resolved:", result);
    console.log("Process finished");
}


processFuture();
```

Here we introduce an `async` function, which is syntactic sugar over promises. The `customAsyncOperation` function returns a promise that resolves after a brief delay. `processFuture`, an `async` function, effectively creates another layer of Promise wrapping.  The `await` keyword, in particular, desugars to a `then` call.  "Process started" is printed, then the 'Starting async operation' message, then the set timeout is queued on the event queue. When the timer expires, "Custom async operation complete" is logged, and the Promise is resolved. This, as before, creates a microtask for the continuation of `processFuture`. The engine will then pick up that microtask, log "Future resolved: Custom async result", and then "Process finished" will be printed. Again, notice that it is not the initial promise that’s on the microtask queue, it is the continuation of the async function as a result of a `then` call on the promise.

These examples highlight the interplay between event and microtask queues. `Futures`, or more precisely the resolution of the `Futures`, typically interact with the microtask queue using the `then` and `catch` methods for scheduling the callbacks based on the resolution value or error. Event queue items, from `setTimeout`, network requests, and the like, can trigger microtask executions, but they do not directly place items into the microtask queue.

If you want a deeper dive into this, I'd highly recommend exploring the concept of the “JavaScript Event Loop” in detail. You could look at the specifications around the ECMAScript standard itself (ECMA-262), particularly the sections discussing Promises and the Event Loop. Specifically, you might want to check “What every JavaScript developer should know about the event loop” by Philip Roberts, a fantastic talk given during JSConf. Additionally, you could take a look at “Deep JavaScript: Theory and techniques” by Axel Rauschmayer, which provides an extensive look into this particular topic. It’s essential reading for any developer looking to master asynchronous Javascript, in my opinion.

This mechanism isn’t just about the browser's quirks; it applies to Node.js as well, and even some other environments that are based around Javascript. Understanding the nuanced differences between these queues and how `Futures` interact with them is key to building predictable and performant applications. It's one of those topics that feels a bit abstract initially, but once you grasp the details, you’ll find a notable improvement in your code quality and debugging efficiency. The devil, as they say, is in the details.
