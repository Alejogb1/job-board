---
title: "How to prevent concurrent Koa API requests from overlapping?"
date: "2024-12-23"
id: "how-to-prevent-concurrent-koa-api-requests-from-overlapping"
---

Alright, let's talk about preventing overlapping concurrent requests in a Koa API. This isn't just theoretical; I've definitely seen this cause headaches in production systems, and finding a good, robust solution can dramatically improve stability. We’re talking about situations where multiple client requests hit your API simultaneously and, if not handled correctly, can lead to data corruption, race conditions, or other undesirable states.

The core challenge stems from the asynchronous nature of javascript and Node.js, combined with the fact that Koa, being a middleware framework, relies heavily on these asynchronous operations. When multiple requests come in close succession, they often end up sharing resources or modifying state in ways that aren't thread-safe, given Node's single-threaded event loop. This is especially pronounced when you have operations that interact with a database or external service. So, the key is to introduce mechanisms that serialise access to these critical sections of your code. I’ve had to deal with this numerous times, particularly in applications handling financial transactions where concurrency errors were simply not an option.

One common approach, and often the most straightforward for many cases, is to leverage request-specific locking using a form of in-memory semaphore or queue system. The idea is to associate a lock with a specific resource or operation and then ensure that only one request can execute code holding that lock at any given time. This can be implemented using various data structures or third-party libraries, and I'll show a basic example later, but first let's discuss why the typical mutex pattern is not always suited for javascript. Mutexes in traditional multi-threaded systems are built to halt and wait on a core; in a single-threaded javascript event loop environment, using a blocking wait for the mutex will simply stop all other operations. Instead, we must carefully craft asynchronous mechanisms that allow other operations to proceed while one request is waiting for a lock to release.

Let’s get down to brass tacks. A crucial step to preventing overlap is *identifying the critical sections of your code.* These are the areas where concurrent modifications can lead to problems, and often involve database updates, file system operations, or any process where shared mutable state is present. Once identified, we need a mechanism to gate access to these sections. This can be done on a per-resource basis, which is usually ideal, or by categorizing and batching, based on the specific application needs.

Now, for some code. Let's start with a relatively basic, though highly functional example using a simple map to store the "locks", using async/await and promises for clarity and non-blocking operation:

```javascript
class RequestLocker {
    constructor() {
        this.locks = new Map();
    }

    async acquire(key) {
        if (!this.locks.has(key)) {
            this.locks.set(key, Promise.resolve());
        }
        let lockPromise = this.locks.get(key);
        let acquirePromise = new Promise(resolve => {
            lockPromise = lockPromise.then(() => resolve());
        });

        this.locks.set(key, acquirePromise);
        return acquirePromise;

    }

    release(key) {
        //No need for explicit release, as the release is implicitly linked to a return from an aquired section
        //this implementation is intended to acquire and then return from a section and that is all that
        //needs to be tracked.
    }
}


const requestLocker = new RequestLocker();

async function processRequest(key, operation) {
    await requestLocker.acquire(key);
    try{
        console.log(`Processing ${key} - Start`);
        await operation();
    } finally {
        console.log(`Processing ${key} - Done`);
    }
}



//example usage in a Koa middleware:
const koaApp = new Koa();

koaApp.use(async (ctx,next)=>{
  const requestId = ctx.request.url;

  const criticalOperation = async ()=>{
        await new Promise(resolve => setTimeout(resolve, 1000));
        console.log(`Operation Completed on ${requestId}`);

        //some other operation to do after the delay, for example some DB update
    };
     await processRequest(requestId,criticalOperation);
     await next();
});



koaApp.listen(3000, () => console.log('Server started on port 3000'));
```
In this snippet, `RequestLocker` holds a map of locks. `acquire` returns a promise that resolves when the lock for the given key is free. Subsequent calls with the same key will get a promise that is resolved after all prior aquire promises for the given key. The `processRequest` function wraps an operation with the acquire and release lock calls, and the `koaApp` demonstrates how you might incorporate this in your Koa middlewares. Run this code with something like `node file_name.js` and hit `localhost:3000/first`, `localhost:3000/second`, `localhost:3000/first` in quick succession. You should notice that the `operation` completion logs for /first happen after eachother and the operation on `/second` occurs seperatly.

While this is functional, it relies on a single global instance of the lock handler. That's usually okay for most applications, and it keeps things simple, but this can cause issues in distributed environments or more complex systems. Also, the above system does not explicitly manage a concept of an *exclusive* lock; it provides serialized access based on key, but not mutual exclusion across keys. In specific scenarios, you might need a more sophisticated mechanism for concurrency.

For instance, consider a situation where you have multiple read operations that can occur concurrently, but only one writer operation should access the data at a given time. You’d need a system that allows for shared read locks and exclusive write locks. Here's a conceptual implementation using a simplified 'ReadWriteLock':

```javascript
class ReadWriteLock {
    constructor() {
        this.readers = 0;
        this.writer = null;
        this.waitingWriters = [];
        this.waitingReaders = [];
    }

    async acquireRead() {
        return new Promise((resolve) => {
            if (this.writer || this.waitingWriters.length > 0) {
                this.waitingReaders.push(resolve);
            } else {
                this.readers++;
                resolve();
            }
        });
    }

    releaseRead() {
        this.readers--;
         this._checkQueue();
    }


    async acquireWrite() {
        return new Promise((resolve) => {
            if (this.readers > 0 || this.writer) {
                this.waitingWriters.push(resolve);
            } else {
                this.writer = resolve;
                resolve();
            }
        });
    }

    releaseWrite() {
        this.writer = null;
        this._checkQueue();
    }

    _checkQueue(){
        if (this.writer) {
            return;
        }
        if(this.waitingWriters.length>0){
             const resolve = this.waitingWriters.shift();
             this.writer = resolve;
             resolve();
        }else if(this.waitingReaders.length>0){
             while(this.waitingReaders.length>0){
                const resolve = this.waitingReaders.shift();
                this.readers++;
                resolve();
             }
         }
    }
}


const rwLock = new ReadWriteLock();

async function readData(){
    console.log("Aquired read lock")
    await rwLock.acquireRead();
    await new Promise(resolve => setTimeout(resolve, 500));
    console.log("Finished read lock");
    rwLock.releaseRead();
}

async function writeData(){
  console.log("Aquired write lock")
    await rwLock.acquireWrite();
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log("Finished write lock");
    rwLock.releaseWrite();
}

async function run(){
    readData();
    readData();
    writeData();
    readData();
    await new Promise(resolve=>setTimeout(resolve,3000)) //wait long enough to run them all
    console.log("done!");
}
run()

```

Notice how `readData()` can run concurrently among itself, but will wait for writes to finish, and write will wait for all read operations to complete before executing.

Another approach involves message queues, such as Redis Pub/Sub or RabbitMQ. These solutions introduce some infrastructure overhead, but they can handle high concurrency with ease and can help with scalability. Here's a simple, conceptual example of how you might use a queue using a simple array:

```javascript
class SimpleQueue {
    constructor() {
        this.queue = [];
        this.processing = false;
    }
    enqueue(item) {
        this.queue.push(item);
        this._processQueue();
    }
    _processQueue() {
        if (this.processing || this.queue.length === 0) return;
        this.processing = true;
        const item = this.queue.shift();
        item()
            .then(() => {
               this.processing = false
                this._processQueue()
             })
             .catch((err)=>{
               this.processing = false
               console.error('Queue error', err)
               this._processQueue()
            })
    }
}

const queue = new SimpleQueue()
//simulate our Koa middleware now
const middlewareSim = (operation) => new Promise((resolve,reject)=>{
    queue.enqueue(()=>operation.then(resolve).catch(reject));
});


async function operation(name, duration){
   console.log(`Start ${name}`);
   await new Promise(resolve => setTimeout(resolve, duration));
   console.log(`Done  ${name}`);
   return;
}

middlewareSim(operation('one',2000))
middlewareSim(operation('two',1000))
middlewareSim(operation('three', 500))
```

The fundamental principle, regardless of implementation, is to serialize access to shared resources. A simple `RequestLocker` as the first example might be ideal for smaller projects and specific needs, while a message queue could be needed for systems requiring more complex infrastructure, perhaps with multiple server instances.

For further information, I'd suggest reading "Patterns of Enterprise Application Architecture" by Martin Fowler for a deeper understanding of concurrency patterns. For more information on javascript async programming, the "You Don't Know JS: Async & Performance" book series by Kyle Simpson is a thorough and well regarded text. Additionally, the official Node.js documentation provides detailed guides on managing asynchrony. The documentation for specific libraries such as `redis` or `amqplib` also offers valuable insights into working with their respective message queues. These resources will provide a more complete picture of the underlying challenges and techniques for preventing concurrent API request overlap. Choosing the right strategy requires an understanding of your system's specific requirements and constraints, but it's a crucial area to master for building robust, reliable web services.
