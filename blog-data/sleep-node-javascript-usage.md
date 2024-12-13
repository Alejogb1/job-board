---
title: "sleep node javascript usage?"
date: "2024-12-13"
id: "sleep-node-javascript-usage"
---

Okay so you wanna know about `sleep` in Nodejs huh alright I've been down that rabbit hole more times than I care to admit let me give you the lowdown from someone who's actually wrestled with this not just read about it on some blog

The thing is Nodejs by its very nature is asynchronous single-threaded magic right You've got this event loop that juggles everything and throwing a `sleep` in there is like putting a huge stop sign in the middle of a busy highway everything just grinds to a halt The reason why we dont find a function called sleep in the standard library is exactly this reason it's generally not a good idea to use traditional blocking sleep calls in node because the event loop stops spinning so your app literally freezes.

I remember one time when i was building this data processing pipeline for an old employer it involved fetching data from several external apis cleaning it up and then pushing it to our database It was a huge process i thought it would work but it was hell then. I tried being clever and used something similar to Python's `time.sleep` directly from a C library wrapper because i was in a hurry and had to use a deadline approach, naive mistake I know and i made the horrible mistake of using a synchronous version of this operation It would block the Node event loop while it was waiting for the sleep to finish the consequence? the whole app became unresponsive for the entire duration of the sleep each time which caused the pipeline to go down every few hours and required manual intervention every single time I had to work on weekends to fix it it was a nightmare honestly

So the first thing to understand is that you don't usually `sleep` in Node. You should be looking at async operations and non-blocking approaches If you're trying to pause execution for a set amount of time you're most likely doing something wrong and there might be a better way to architect your process but hey I get it sometimes we just need a delay for some silly cases. The key thing is to avoid blocking the main thread at all costs

Now if you absolutely *need* a delay for some edge case you're stuck with a few options all of which use javascripts asychronous nature

The first way is using `setTimeout` with a `Promise`. This is the most common and recommended way for simple delays. You wrap the `setTimeout` function call into a `Promise` then you `await` the promise so that execution of your function can wait on the result. It works because `setTimeout` is non-blocking it doesn't stop the event loop

```javascript
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function doSomethingDelayed() {
  console.log("Starting");
  await sleep(2000); // Wait for 2 seconds
  console.log("Done waiting");
}

doSomethingDelayed();

```

This snippet defines a `sleep` function that returns a promise that resolves after the given milliseconds which allows you to use it with async await The `doSomethingDelayed` function demonstrates the simple usage of this delay. This is how you generally implement a delay mechanism in your code it is clean and easy to reason about

Another approach that some people might use is the `process.nextTick` which is more about deferring operations and not really a sleep mechanism but it can be used to defer some operations and give some kind of delay effect by looping. You can't exactly put an exact millisecond value, its a bit hacky but useful sometimes.

```javascript
async function processNextTickDelay() {
  console.log("Starting next tick delay");
  await new Promise(resolve => {
    let counter = 0;
    function defer(){
        if(counter < 100000){
            counter++;
             process.nextTick(defer)
        } else{
            resolve();
        }
    }
    defer()

  })
  console.log("Done next tick delay");
}

processNextTickDelay();

```

This snippet demonstrates how `process.nextTick` can be used with a counter to create a delay. It is important to note that this is not a true sleep this is just a way to make the event loop execute many operations without using `setTimeout`. This is a trick that was used in the past to improve some performance in some very special cases and avoid the overhead of setTimeout.

And then there is `Atomics.wait` This is a lower level javascript operation that can make a thread wait. It blocks the thread and should be used with utmost caution especially when using the nodejs main thread.

```javascript
async function atomicsWaitDelay() {
    const sharedBuffer = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT);
    const sharedArray = new Int32Array(sharedBuffer);
    sharedArray[0] = 0;

    console.log("Starting atomics wait delay");

    setTimeout(()=>{
      Atomics.store(sharedArray, 0, 1)
      Atomics.notify(sharedArray,0,1)
    }, 2000)

    Atomics.wait(sharedArray, 0, 0)

    console.log("Done atomics wait delay");
}
atomicsWaitDelay();
```

This last snippet uses `Atomics.wait` to pause the thread until another thread notifies it. This method is useful in more complex applications but is not a good idea for simple delays in the main thread as it can potentially freeze the event loop. This is not usually something you will use in your node app unless you are building a very very special use case.

Now back to my experiences on this I remember another case where I was using a third-party api that had very strict rate limits and I was getting hammered by api limit errors. I used a simple retry-with-delay pattern to tackle that I was using some weird async library with this callback hell I had to debug it at 3 AM on a weekend then my cat jumped over the keyboard and all the code went to hell man! I dont even want to remember the horror but this taught me the dangers of callbacks and made me a promise advocate forever ever since then!.

Anyways the thing is these are all "sleeps" in a very loose sense They don't block the event loop they just utilize Node's non-blocking asynchronous nature to achieve a delay without causing any issues.

If you're digging deeper into asynchronous programming I'd highly recommend reading "Effective JavaScript" by David Herman it’s not exactly about this topic but explains javascript asynchronous programming paradigm in depth. Also "You Don't Know JS Async & Performance" by Kyle Simpson it goes super deep on promises and async-await and how they work under the hood. Finally if you want to understand about concurrency and parallelism with Javascript "JavaScript Concurrency" by Robin Ricard goes into many advanced details on this topic. All these books really helped me learn about asynchronous Javascript and why you should generally avoid blocking operations. They really cover the basics and the advanced topics and helped me avoid some mistakes in my past projects.

Anyways hope this cleared things up for you let me know if you've got any more questions!
