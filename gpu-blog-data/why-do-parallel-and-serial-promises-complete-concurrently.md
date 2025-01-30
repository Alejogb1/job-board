---
title: "Why do parallel and serial promises complete concurrently?"
date: "2025-01-30"
id: "why-do-parallel-and-serial-promises-complete-concurrently"
---
The fundamental misconception underlying the question lies in the conflation of promise execution and promise resolution.  While the *resolution* of parallel promises may appear concurrent, the underlying execution, particularly in environments lacking true multi-threading at the JavaScript engine level, often proceeds serially at the engine's discretion.  This is a subtle but crucial distinction I've encountered numerous times while optimizing asynchronous JavaScript applications.

My experience optimizing large-scale data processing pipelines has highlighted this repeatedly. We initially assumed parallel promises, leveraging `Promise.all`, would always deliver true parallelism, resulting in significant performance gains.  However, we discovered that while the *overall completion time* improved, the individual promise execution often adhered to a serial or highly interleaved pattern, dictated by the JavaScript engine's event loop.

**1. A Clear Explanation**

JavaScript's event loop is single-threaded. While promises offer the *illusion* of concurrent execution through asynchronous operations, the engine itself executes tasks sequentially within the event loop.  When you use `Promise.all` with multiple promises, these promises are *scheduled* concurrently. This doesn't imply simultaneous execution on multiple cores. Instead, the engine initiates each promise, placing the related tasks in the event loop's callback queue.  The engine then processes these tasks one after another.  However, because I/O-bound tasks (like network requests or file system operations) release control back to the event loop before their completion, the engine can interleave other tasks. This creates the *impression* of concurrency – the overall time is drastically reduced because the engine efficiently switches between promises instead of waiting for one to fully complete before beginning the next.

Consider a scenario with three promises, each performing a lengthy network request.  If these requests were truly parallel, the completion time would be approximately equal to the time for the longest single request.  However, with a single-threaded event loop, the engine starts the first request, then moves to the second while the first is pending, and so on.  Once a response arrives for a request, its associated `.then()` block is executed. Therefore, the completion of `Promise.all` signals the resolution of *all* promises, not necessarily the simultaneous completion of their underlying operations.

CPU-bound operations, on the other hand, present a different picture. Since these operations consistently occupy the event loop, the benefit of concurrent scheduling diminishes. We observed this during image processing tasks where intensive computations dominated the promise functions. Here, the speedup from `Promise.all` was minimal compared to the I/O-bound scenario.

The perceived concurrency stems from the asynchronous nature of promises and the efficiency of the event loop's task management. The engine smartly handles the asynchronous callbacks and interleaves the execution to maximize throughput.  But the underlying reality remains: the execution is largely sequential, governed by the single-threaded nature of the JavaScript engine.

**2. Code Examples with Commentary**

**Example 1: I/O-bound Operations (Network Requests)**

```javascript
const promises = [
  fetch('http://example.com/data1').then(res => res.json()),
  fetch('http://example.com/data2').then(res => res.json()),
  fetch('http://example.com/data3').then(res => res.json())
];

Promise.all(promises)
  .then(results => {
    console.log('All data received:', results);
  })
  .catch(error => {
    console.error('Error fetching data:', error);
  });
```

In this case, while each `fetch` call is initiated sequentially, the engine efficiently switches between them as responses become available. The overall completion time is significantly faster than sequential `fetch` calls, simulating concurrency.

**Example 2: CPU-bound Operations (Intensive Calculations)**

```javascript
function heavyComputation(n) {
  let result = 0;
  for (let i = 0; i < 100000000; i++) {
    result += Math.pow(i, n);
  }
  return result;
}

const promises = [
  Promise.resolve(heavyComputation(2)),
  Promise.resolve(heavyComputation(3)),
  Promise.resolve(heavyComputation(4))
];

Promise.all(promises)
  .then(results => {
    console.log('Computations completed:', results);
  })
  .catch(error => {
    console.error('Error during computation:', error);
  });
```

Here, the `heavyComputation` function is CPU-intensive.  The apparent concurrency is less noticeable because each calculation holds the event loop for a longer duration, minimizing the benefits of interleaving. The completion time will be closer to the sum of the individual computation times.

**Example 3: Mixed Operations**

```javascript
const promises = [
  fetch('http://example.com/data').then(res => res.json()),
  Promise.resolve(heavyComputation(2)),
  fetch('http://example.com/moreData').then(res => res.text())
];

Promise.all(promises)
.then(results => {
    console.log('Mixed operations completed:', results);
})
.catch(error => {
    console.error('Error during mixed operations:', error);
});

```

This example combines both I/O-bound and CPU-bound tasks.  The engine will interleave the I/O operations efficiently, but the CPU-bound task will create a bottleneck, demonstrating the limitations of simulated concurrency in a single-threaded environment.


**3. Resource Recommendations**

For a deeper understanding of the JavaScript event loop and asynchronous programming, I recommend studying the specifications for promises and asynchronous operations, and exploring advanced JavaScript texts focusing on performance optimization and concurrency patterns.  Furthermore, examining the internals of popular JavaScript runtimes can offer valuable insights into how the engine manages asynchronous tasks.  Lastly, consider investigating alternative approaches to concurrency, like Web Workers, for scenarios demanding true parallel execution beyond the limitations of the event loop.
