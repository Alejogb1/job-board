---
title: "How to do an async await geocode api call within a map with bottleneck?"
date: "2024-12-14"
id: "how-to-do-an-async-await-geocode-api-call-within-a-map-with-bottleneck"
---

so, you're looking to fire off a bunch of geocoding requests inside a map, but want to avoid hammering the api and hitting rate limits, while keeping the code async, eh? i've been down this road more times than i care to count, it's a classic problem, and i can totally relate to the headaches. 

i remember back in my early days, working on a project that involved converting hundreds of addresses to coordinates for a map visualization. i figured, “hey, it’s just a simple loop”. boy, was i wrong. that api started throwing 429s faster than you could say ‘throttling’. learned my lesson the hard way that day: rate limiting is not a suggestion, it’s the law.

the key here isn't about doing things sequentially, which is easy, but rather balancing concurrency with respect to those rate limits. we want to use async/await to keep things tidy, but we need to control how many requests are happening at once. the solution comes in the form of a 'bottleneck', basically an async limiter. it's a pattern i’ve found myself reaching for frequently when dealing with external apis.

the simplest way to do this is not using libraries, but building our custom bottleneck using semaphores. think of it as a traffic controller that manages how many concurrent geocode requests can be active at any time.

here is an example implementation that i cooked up with simple semaphores:

```javascript
class Semaphore {
    constructor(permits) {
        this.permits = permits;
        this.waiting = [];
    }

    async acquire() {
        if (this.permits > 0) {
            this.permits--;
            return;
        }

        return new Promise((resolve) => {
            this.waiting.push(resolve);
        });
    }

    release() {
        if (this.waiting.length > 0) {
            const resolve = this.waiting.shift();
            resolve();
        } else {
            this.permits++;
        }
    }
}


async function geocodeWithSemaphore(address, geocodeFunction, semaphore){
     await semaphore.acquire();
     try {
         const data = await geocodeFunction(address);
         return data;
     } finally {
         semaphore.release();
     }
}


async function batchGeocodeWithSemaphore(addresses, geocodeFunction, maxConcurrent){
      const semaphore = new Semaphore(maxConcurrent);
      const results = await Promise.all(addresses.map(address => geocodeWithSemaphore(address,geocodeFunction,semaphore)));
      return results;
}

//example of a geocode function using google's one
async function googleGeocode(address) {
    const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(address)}&key=YOUR_API_KEY`;
    const response = await fetch(url);
    const data = await response.json();
    if(data.status !== 'OK') throw new Error(`Geocoding failed for ${address} with status ${data.status}`);
    return data.results[0].geometry.location;
}

async function main() {

  const addresses = ["1600 Amphitheatre Parkway, Mountain View, CA", "1 Infinite Loop, Cupertino, CA", "One Microsoft Way, Redmond, WA"];
  
  try {
      const results = await batchGeocodeWithSemaphore(addresses,googleGeocode,3);
      console.log(results);
      //expect to see a list of coordinates
  } catch (error) {
     console.error('error doing geocoding',error);
  }
}

main();
```
this code sets up a basic semaphore implementation. `acquire()` makes a request to get a slot in the semaphore. if there are any slots `available` it gets one, otherwise it waits. `release()` makes available again a slot if there are any waiting requests. `geocodeWithSemaphore` makes use of this by acquiring before calling the api and releases the lock in a finally block to ensure that locks are always released no matter the result of the geocode api call. finally the `batchGeocodeWithSemaphore` just wraps the whole process together, and uses a javascript map to create all the promises and then awaits them all using `promise.all`. we’re limiting the calls to 3 at a time.

now, you can use your own geocoding function. i used google's here. just replace it with your geocoding method. remember to handle error cases and api key correctly or this whole thing will fall over like a house of cards.

another implementation i usually use, which has a slightly different approach and is even more simple to understand, consists in using recursion with a delayed promise. the advantage is that we don't need to use any semaphore classes or external libraries. it's easier to setup and simpler to understand the flow of execution:

```javascript
async function geocodeWithDelay(address, geocodeFunction, delayMs, results = [], index = 0) {
    if (index >= address.length) {
      return results;
    }

    const currentAddress = address[index];
    try {
      const data = await geocodeFunction(currentAddress);
      results.push({address: currentAddress, data: data});
      await new Promise(resolve => setTimeout(resolve, delayMs)); // Delay for next call
    }
    catch(error){
      results.push({address: currentAddress, error: error});
      await new Promise(resolve => setTimeout(resolve, delayMs)); // Delay even if error to respect rate limits
    }


    return geocodeWithDelay(address, geocodeFunction, delayMs, results, index + 1);
}

//example of a geocode function using google's one
async function googleGeocode(address) {
    const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(address)}&key=YOUR_API_KEY`;
    const response = await fetch(url);
    const data = await response.json();
      if(data.status !== 'OK') throw new Error(`Geocoding failed for ${address} with status ${data.status}`);
    return data.results[0].geometry.location;
}


async function main() {
  const addresses = ["1600 Amphitheatre Parkway, Mountain View, CA", "1 Infinite Loop, Cupertino, CA", "One Microsoft Way, Redmond, WA"];
  try {
    const results = await geocodeWithDelay(addresses, googleGeocode, 200); // 200ms delay
    console.log(results);
  } catch(error) {
    console.error('error doing geocoding',error);
  }
}

main();
```

in this implementation, `geocodeWithDelay` takes a delay time parameter (`delayMs`), a list of addresses and the geocoding function. it calls the geocoding api, stores the result, waits the delay time using a promise and if there are any addresses left to process it makes a recursive call. this way ensures no concurrency as requests are done in order, one at a time, but respects a minimum delay to make sure not to trigger the rate limits of your api. is simpler, easier to read, and you are sure that the code is not trying to call all the geocodes in parallel. if an error occurs, the delay is still waited before continuing the recursive call. this approach ensures a controlled flow of requests and is surprisingly efficient for smaller datasets.

finally, and if you really want to go all in and get more control and flexibility with your bottleneck you can use another approach, this time implementing a 'task queue'. it's like having a list of jobs to do and you pull them out one by one while respecting concurrency and rate limits. each task is a promise. the task queue itself is also a promise that resolves when all tasks are done:

```javascript
class TaskQueue {
    constructor(concurrency) {
        this.concurrency = concurrency;
        this.running = 0;
        this.queue = [];
    }

    enqueue(task) {
        return new Promise((resolve, reject) => {
            this.queue.push({ task, resolve, reject });
            this.run();
        });
    }


    run() {
        while (this.running < this.concurrency && this.queue.length) {
            this.running++;
            const { task, resolve, reject } = this.queue.shift();
            task()
                .then(resolve)
                .catch(reject)
                .finally(() => {
                    this.running--;
                    this.run();
                });
        }
    }

     allCompleted(){
        return new Promise((resolve) => {
             const checkQueue = () => {
                if (this.queue.length === 0 && this.running === 0) {
                   resolve();
                  } else {
                    setTimeout(checkQueue, 10);
                 }
            }
              checkQueue();
        });
     }

}

async function batchGeocodeWithQueue(addresses, geocodeFunction, maxConcurrent) {
    const queue = new TaskQueue(maxConcurrent);
    const results = [];
    for (const address of addresses) {
      const task = async () => {
       try {
         const data = await geocodeFunction(address);
         results.push({address, data});
       }
       catch(error) {
          results.push({address,error})
       }
      }
      queue.enqueue(task);
    }
    await queue.allCompleted();
    return results;
}


//example of a geocode function using google's one
async function googleGeocode(address) {
    const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(address)}&key=YOUR_API_KEY`;
    const response = await fetch(url);
    const data = await response.json();
      if(data.status !== 'OK') throw new Error(`Geocoding failed for ${address} with status ${data.status}`);
    return data.results[0].geometry.location;
}


async function main() {
  const addresses = ["1600 Amphitheatre Parkway, Mountain View, CA", "1 Infinite Loop, Cupertino, CA", "One Microsoft Way, Redmond, WA"];
  try{
      const results = await batchGeocodeWithQueue(addresses,googleGeocode,3);
      console.log(results);
  } catch(error) {
      console.error('error doing geocoding', error);
  }
}

main();
```

this `taskqueue` implementation offers great flexibility as you can enqueue as many tasks as you want, and the class will make sure they get called while respecting the concurrency limit. this way you have full control over the queue itself and how they are called in a non-blocking way. this is useful in more complex scenarios where the processing of each geocode may require further processing. i have used this in situations where the geocode is just the beginning of a complex chain of asynchronous api calls. i used this in a personal project where i was gathering data from multiple sources and the `geocoding` was just a small initial step.

remember to always check and respect the api documentation when using public apis. most of them have their own limits, so knowing them in advance is key to not having your code break. also it's important to catch errors to deal with the cases where the geocoding fails for some addresses, as it will happen in real life.

in my experience, i would recommend reading "javascript concurrency" by peter lewis if you want to go deeper into how to use promises in javascript, and also read the "asynchronous javascript" guide in mdn docs if you want to know all the ins and outs of javascript's async capabilities. they are invaluable in building solid async systems. 

i hope this helps you avoid some headaches. the first implementation with the semaphores is my favorite for a simple setup. the second recursive implementation with a delay is easier to understand and requires no extra classes. the last is for more complex cases where you need more control. don’t be shy to experiment a little and see which fits best for your project. remember to always handle those 429 errors and test carefully to not overuse public apis. good luck with your project!
