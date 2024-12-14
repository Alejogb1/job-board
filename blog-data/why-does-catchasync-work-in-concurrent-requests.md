---
title: "Why does catchAsync work in concurrent requests?"
date: "2024-12-14"
id: "why-does-catchasync-work-in-concurrent-requests"
---

well, i've seen this pattern pop up a fair bit, especially when you're dealing with node.js and promises. i've personally banged my head against it a few times, mostly in my early days building apis. the "catchAsync" pattern, as it's often called, isn't some magic trick. it's actually pretty straightforward once you break it down, and understanding how it interacts with javascript's event loop and async operations is key to see why it works so well with concurrent requests.

let’s start with a common issue. say you have a function that does some async operation, maybe fetching from a database, or calling an external api. if that operation throws an error and you are not careful, it can easily crash your node.js process or lead to unhandled promise rejections. this is a no-go, especially with concurrent requests. imagine hundreds or thousands of requests pouring in and a single unhandled error bring everything down, that's the stuff of nightmares. i recall one time in my previous company when i was working on this internal tool. we were batch-processing user data, and one poorly written async database query caused the whole service to crumble under load. learnt that hard lesson.

the "catchAsync" pattern basically uses higher-order function, a function that returns another function, often leveraging a try/catch block to intercept errors within asynchronous operations. that returned function then gets passed to your route handler. let's say you are in expressjs. let's see a basic example without any fancy error handler:

```javascript
const myAsyncFunction = async () => {
  // imagine this is some async operation
  await new Promise((resolve) => setTimeout(resolve, 100));
    throw new Error('oops, something went wrong');
};

app.get('/badroute', async (req, res) => {
  await myAsyncFunction(); // no error handling here
  res.send('ok, this never happens');
});
```

if you run the code above and hit the `/badroute` endpoint you will end up with an uncaught exception, bad news. that's where catchAsync comes in. now here is how a basic version of catchAsync may look:

```javascript
const catchAsync = (fn) => {
  return (req, res, next) => {
    fn(req, res, next).catch(next);
  };
};

// now using the previous route

app.get('/betterroute', catchAsync(async (req, res) => {
   await myAsyncFunction();
   res.send('ok, this also never happens');
}));
```

in this second example `catchAsync` takes a function `fn`, that we expect to be a function that returns a `promise`, and returns a new function. that new function is designed to be used as a route handler in express or similar web frameworks. inside the new function we execute our original function using `fn(req, res, next)` which returns a promise. then with `.catch(next)` we are handling any error and sending it to the express error handler. the beauty is here `catchAsync` is not blocking. if you send multiple requests to `/betterroute` simultaneously each async function call will be executed and if any errors happen each error will get catch by the `catch` method of each returned promise, and each error will be correctly sent to the `next` express error handler, meaning every concurrent request is handled independently. no interference, no crashes. the event loop keeps chugging along. and since node.js is non blocking, the program does not wait for one request to be finished to continue processing other requests, its not a serial process.

lets dive a little bit more into concurrent requests. the key here is that the javascript event loop is single threaded and nodejs is non blocking. when you have an async operation like a database query or an http request, nodejs doesn't sit there waiting for it. instead, it registers a callback to be executed when that operation is done and keeps processing other things in the event loop. when a request come in, the server immediately picks it up and it does not block the other requests.

think of it like this: you're a restaurant server and several tables place their orders at around the same time. you dont stand there until the first table food is ready before taking another tables order, instead you take all the orders and send them to the kitchen and then come back for the food, once its cooked. the javascript event loop and nodejs work in the same way. nodejs takes the async operations and handles them and is notified once they are finished and processes them. but it does not stop attending other requests in the meantime. each concurrent request gets its own promise chain managed by the `async` and `catchAsync`.

it’s important to note that this pattern relies on the fact that `async` functions implicitly return promises. when an async function throws an error, that error is propagated down the promise chain, becoming a rejected promise. the `catch` method is basically like the emergency brake on a runaway promise, preventing it from causing a total meltdown.

now let me show you a more robust `catchAsync` implementation that includes logging. in a production environment its paramount to log the errors.

```javascript
const catchAsync = (fn, logger) => {
    return (req, res, next) => {
        fn(req, res, next).catch(err => {
           logger.error("an error happened:", err)
           next(err) // passing error to express error handler
        })
    }
}

const myLogger = {
    error: (msg, err) => console.error(msg, err) //replace with your favorite logger
}

app.get('/evenbetterroute', catchAsync(async (req, res) => {
  await myAsyncFunction();
  res.send('ok, this also never happens');
}, myLogger));

```

in this third snippet, i added a `logger`, this is super helpful and in production code you want to use proper logging libraries like winston or similar that can log to external files or other databases.

now the 'why does it work with concurrent requests' part becomes clearer. each request that hits your server triggers a new execution context. these contexts are independent from each other. when you wrap each route handler with `catchAsync` you ensure that each request creates its own isolated promise chain. that isolated chain includes the try/catch logic provided by the `catchAsync` function. if a error occurs during one requests, it is caught only in that request specific try/catch block and never will cause any problem to other concurrent request execution contexts or any other async function execution. the event loop manages these independent async operations, so a single error doesn't bring down other requests, the entire service or the nodejs program.

it's like each request is a different worker bee in a hive, working on its task and each bee has its own little safety net to prevent problems in each task. the hive keeps humming even if some bees encounter little issues.

i once saw a junior dev get really confused about this, thinking that the error handling might only work for the first request, which is not the case. he was trying to debug a concurrency problem, and he was pulling his hairs out for a couple of hours, it was hilarious (not in a bad way obviously, but it was funny to watch). but hey, we’ve all been there. debugging these kind of issues can be very frustrating.

for a deeper theoretical dive, i'd highly recommend reading the writings of douglas crockford on javascript, he goes deep into the inner workings of javascript and it will give you a better understanding of the underlying mechanisms. "javascript the good parts" is a classic and goes very deep. also explore books on concurrency and asynchronous programming in general, such as "concurrent programming in java", even if its java oriented, most concurrency concepts are transferable. understanding the principles behind the event loop, promises, and error handling is fundamental to grokking this pattern, and it's totally worth the effort.
