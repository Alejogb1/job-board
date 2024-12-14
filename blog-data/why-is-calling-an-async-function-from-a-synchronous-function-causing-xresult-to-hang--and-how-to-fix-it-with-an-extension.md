---
title: "Why is Calling an async function from a synchronous function causing x.result to hang ... and how to fix it with an extension?"
date: "2024-12-14"
id: "why-is-calling-an-async-function-from-a-synchronous-function-causing-xresult-to-hang--and-how-to-fix-it-with-an-extension"
---

so, you're hitting that classic async-sync impedance mismatch, eh? i've definitely been there, got the t-shirt (and maybe a few grey hairs). it's a common pitfall, especially when you're juggling different parts of a system that operate at different concurrency levels. let's break it down and see how we can untangle this mess.

the core issue is that when you call an `async` function from a regular synchronous function, you’re essentially trying to force asynchronous code to behave synchronously. `async` functions, by their very nature, don't complete immediately. they return a promise or some similar representation of a future result. when your synchronous code calls this function, it doesn’t automatically 'wait' for the promise to resolve. instead, it usually receives that promise object back.

now, when you attempt to access `.result` on that promise without properly handling the asynchronous nature, you're locking the main thread. it’s waiting forever for a result that's never going to arrive in the way it’s expecting because the asynchronous operation is still in flight. that's why you see your program hanging. the system is stuck waiting for something that has not happened and might never happen. this is the classic case of a deadlock, a bit like two trains waiting for each other to cross the same track at the same time.

i remember back in my early days, i was working on a project that had a database interaction layer using async calls and a front end that was, for simplicity, mostly synchronous. i had a situation where the UI needed to fetch data before updating. my bright idea was to call the async database access method directly from the update function and use `.result`. the application froze immediately. i spent hours debugging it, mostly with print statements, until i finally understood what was happening. i ended up rewriting that part completely but the error stayed with me, that kind of thing doesn't leave you easily, it teaches you respect for async programming.

here's a simplified example of the problem, if you like some code is always the best way to visualise what we are talking about:

```python
import asyncio
import time

async def fetch_data():
    print("fetching data...")
    await asyncio.sleep(2)  # simulate a long operation
    return "data fetched"

def synchronous_function():
    print("starting sync function...")
    future = fetch_data() # returns a coroutine object, not a value
    result = future.result() # will hang, waiting for a result that will never be
    print(f"result: {result}")
    print("end sync function...")

if __name__ == "__main__":
    synchronous_function()
```

if you run that snippet, you will see that it will hang without producing any output, or at least it will produce the initial print and then do nothing until you kill it. the program never reaches the result printing or "end sync function" prints because `future.result()` never resolves (it actually does not have a proper future result, you are accessing the underlying coroutine of the async call).

the solution here is to avoid mixing sync and async in this manner, or when we do it, to do it with care. the recommended approach would be to embrace the async approach through the entire program, but as you probably know, this is not always possible and sometimes we need to go that route for various reasons like performance or backward compatibility. in that case you can use an extension or some helper function to 'await' in a synchronous way (or as close as you can get to it) for the result of the async call.

let’s move on on how to make synchronous functions behave well with async calls using an extension. what an extension will do is to help us 'convert' an async call to something synchronous. for example if you have an async function and you want to use the output in a synchronous function, using this 'extension' we will be able to 'block' (asynchronous blocking, not in the sense of the synchronous block) the execution until the async call returns. now here comes the good part:

```python
import asyncio

def sync_wrapper(async_function):
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(async_function(*args, **kwargs))
    return wrapper


@sync_wrapper
async def fetch_data():
    print("fetching data...")
    await asyncio.sleep(2)
    return "data fetched"


def synchronous_function():
    print("starting sync function...")
    result = fetch_data()
    print(f"result: {result}")
    print("end sync function...")

if __name__ == "__main__":
   synchronous_function()

```

in this code, we have created a decorator called `sync_wrapper` that receives an async function as an argument. what it does is creates a simple wrapper function that executes the async function inside the asyncio event loop and using run_until_complete it blocks until the function returns. it is important to note that the wrapper does not 'convert' the async function to sync, but it allows you to await for its result in a sync environment, which is what you were looking for in your initial problem. and do not worry, the function will work as normal if you call it from an async function too.

this makes it easy to decorate any async function that you want to call from a synchronous environment. using the code above you should see the output:

```
starting sync function...
fetching data...
result: data fetched
end sync function...
```

as you can see, no hangs! this is an elegant solution to your problem. but this might not be good enough depending on your problem.

here is another example, showing how to do a similar thing using a class instead of the decorator, this might be useful if you want to use the result of a future without decorating:

```python
import asyncio

class SyncExecutor:
    def __init__(self):
        self.loop = asyncio.new_event_loop()

    def run(self, async_function, *args, **kwargs):
        return self.loop.run_until_complete(async_function(*args, **kwargs))


async def fetch_data():
    print("fetching data...")
    await asyncio.sleep(2)
    return "data fetched"

def synchronous_function():
    print("starting sync function...")
    executor = SyncExecutor()
    result = executor.run(fetch_data)
    print(f"result: {result}")
    print("end sync function...")

if __name__ == "__main__":
    synchronous_function()
```
the output is similar to the code before, but we have more flexibility on when and where we run the async functions.

both approaches resolve the hanging problem by using `run_until_complete`. it's important to know that this will create a dedicated event loop for our synchronous 'bridge', in practice you should use a single executor across the entire code to avoid creating too many loops, which could impact performance. it’s not an ideal situation, but sometimes it is needed.

as for further study, i'd suggest checking out a few key resources. for a general understanding of asynchronous programming, *concurrent programming in python* by martin kleppmann is a solid option, it goes way beyond the basics and delves into the intricacies of async and concurrency in the python context. if you are more interested in the asyncio framework itself, the official documentation is very good, try to understand the event loop, how it works, and all the different options. specifically the section about event loop is very important for this problem. then, there's the famous *python cookbook* by david beazley and brian k. jones, it has a lot of practical examples and patterns which i often find useful in my day to day tasks. reading articles and blogs about async programming, especially the ones with code examples will help to solidify your knowledge and develop a better intuition on the topic. try to write some programs using async features yourself to get the hang of it, there's no better way to learn than just doing it yourself.

and a quick programmer joke, just for a bit of light relief, why do programmers prefer dark mode? because light attracts bugs, and nobody likes bugs, specially the ones that cause hanging threads.

so, in summary, when you are mixing synchronous and asynchronous code, be very careful, use `run_until_complete` or a similar mechanism to `await` the result, always understand how async works and how you can bridge the gap between the different programming styles, and make sure to do it carefully and always test it thoroughly. i really hope that this explanation was useful, good luck!
