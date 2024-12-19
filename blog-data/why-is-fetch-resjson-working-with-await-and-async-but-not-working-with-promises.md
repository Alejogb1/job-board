---
title: "Why is fetch res.json() working with await and async but not working with promises?"
date: "2024-12-15"
id: "why-is-fetch-resjson-working-with-await-and-async-but-not-working-with-promises"
---

alright, so you're hitting that classic javascript gotcha with `fetch`, `async/await`, and promises, right? i've been there, trust me. it's a rite of passage for anyone working with asynchronous javascript. let me break down what's going on, from my own experience and with some code examples.

first off, the core of the issue isn't that `res.json()` *doesn't* work with promises. it does. it absolutely returns a promise. the problem arises because the `fetch` api, when used with `.then()` chaining, requires that you handle each promise resolution correctly. it's about the specific timing of when you're trying to access the json data.

let's look at a scenario that probably mirrors what you're experiencing. say you're trying to grab some json data from an api.

here's a common incorrect way of doing it with promises that i used to commit in my early days:

```javascript
fetch('https://api.example.com/data')
  .then(res => {
    console.log(res); // this will be the response object
    return res.json();
  })
  .then(data => {
    console.log(data); // you might think this would work, but...
    // try to do something with data here, it might fail
  });

```

what's happening? in this code, `fetch` returns a promise that resolves with a `response` object. we log that response, and it's all good. then, we use `res.json()`, which itself *returns another promise* that will resolve with the actual javascript object parsed from json. but if you try to treat the `data` as your data inside of the second `.then` without waiting for the promise to be resolved, well it will be undefined and will trigger an error. this can create subtle problems that can be difficult to trace.

the issue isn't with `res.json()` itself, it's that you're trying to immediately use the result of a promise before it's resolved. you're thinking about the data object as it was something instantly fetched but this process is asynchronous, the `data` object will only be filled when the asynchronous operation is completed.

now, compare this to how you typically use `async/await`:

```javascript
async function fetchData() {
    const res = await fetch('https://api.example.com/data');
    console.log(res); // this is the response object
    const data = await res.json();
    console.log(data); // this works as expected
    // you can work with data here
}

fetchData();

```

the `await` keyword is key here. when you `await` `fetch()`, javascript pauses execution of the function until the promise it returns is resolved. only when `fetch()`'s promise is resolved, then its resolved value assigned to the res variable. the same applies to `res.json()`. javascript will pause execution of the function and will not assign the value to the data variable until the promise of `res.json()` has been resolved with the parsed data. this is synchronous in nature from the javascript engine point of view but underneath this the `res.json()` is an asynchronous operation.

this `await` does two things for us: it unwraps the value inside the promise (the `response` object or the data) and it waits until the promise is resolved before proceeding further with the code. this is why with `async/await` it appears to 'just work'. you get the json data when you expect it, without having to manage nested promises. it's simpler, it's cleaner, and for many programmers, it's more intuitive.

i remember back when i first started learning javascript, i went crazy trying to chain those promises in the correct order. i had callbacks nested all over the place, and things were getting out of control. it felt like i was a puppet master with strings tangled all over, it was a debugging nightmare. this is where the beauty of `async/await` came in. it made asynchronous programming so much easier to read and to think about, less headache and more productivity.

think of it like waiting in a line. with promises, you’re constantly getting updates on whether the server is ready. "is it done yet? how about now?". with `async/await`, you just step into line and when you reach the front desk and everything is prepared for you, then you get what you want, without having to actively check every step of the way.

here's another example that highlights the difference using promises that includes error handling that is something you also should be paying attention:

```javascript
fetch('https://api.example.com/data')
    .then(res => {
        if(!res.ok){
            throw new Error(`http error: ${res.status}`);
        }
        return res.json();
    })
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error('there was a problem with the fetch', error);
    });
```

and now the same example using `async/await` syntax:

```javascript
async function fetchData() {
    try{
        const res = await fetch('https://api.example.com/data');
        if(!res.ok){
            throw new Error(`http error: ${res.status}`);
        }
        const data = await res.json();
        console.log(data);
    } catch(error){
        console.error('there was a problem with the fetch', error);
    }
}

fetchData();
```

see how the error handling with `async/await` is more straightforward? we use the usual try/catch blocks. it feels like normal synchronous error handling. with promises, you're dealing with the `.catch()` on each promise chain which can be more verbose.

if you want to go deeper, i'd highly recommend reading up on javascript promises. there's a good book that covers these topics very well called “effective javascript” by david herman and also “javascript: the good parts” by douglas crockford, it can help you understand the intricacies of asynchronous javascript.

the key takeaway is that both methods are using promises under the hood, but the way they handle these asynchronous operations are different. promises are more manual, requiring that you understand the `.then()` chaining and error handling process, while `async/await` simplifies it into a more synchronous looking flow. both of them work, it's just that `async/await` is much more modern and less error prone.

and as a final thought here's a joke for you: why was the javascript developer always so calm? because they knew how to handle all their *promises*!
