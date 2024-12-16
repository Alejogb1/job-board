---
title: "How do I run two suspended functions and return when the first one finishes?"
date: "2024-12-16"
id: "how-do-i-run-two-suspended-functions-and-return-when-the-first-one-finishes"
---

Alright, let’s tackle this one. It’s a classic concurrency challenge, and I’ve certainly tripped over it a few times in my career—especially back when I was working on that distributed task processor a few years ago. We had similar requirements, needing to execute parallel routines but primarily interested in the result of the fastest completion.

So, your question revolves around running two suspended functions, which I’m interpreting to mean coroutines in a language that supports them—like Kotlin’s coroutines, Python’s asyncio, or even Javascript's async/await. The specific task is to have the main operation return as soon as the first of these two suspended functions completes, essentially canceling the other if necessary, or ignoring its result if not.

The core issue isn't just starting two operations concurrently. It's about intelligently managing their lifecycles and handling their outputs. A naive approach would launch both coroutines and then wait for both to finish, which isn’t what you want. You need something that’s akin to a "race" condition, where the first to finish wins. There are a few solid techniques to achieve this, and I'll illustrate them with examples. We'll look at constructs that enable precisely this behavior: waiting for the first completion while abandoning or discarding results from others.

**First Approach: Explicit Cancellation with `withTimeoutOrNull` (Kotlin Example)**

The most straightforward way, assuming you are working within the kotlin coroutine scope, is using `withTimeoutOrNull` coupled with a structured concurrency pattern and cancellation mechanics. I’ve used this quite often, and it's surprisingly effective for scenarios where a timeout is acceptable. This is, I’d argue, the most explicit and generally safe method.

```kotlin
import kotlinx.coroutines.*

suspend fun taskOne(delayMillis: Long, result: String): String {
    delay(delayMillis)
    println("Task One Completed with $result")
    return result
}

suspend fun taskTwo(delayMillis: Long, result: String): String {
    delay(delayMillis)
     println("Task Two Completed with $result")
    return result
}

suspend fun main() {
    val job = GlobalScope.launch {
        val result = withTimeoutOrNull(500L) {
            coroutineScope {
                val deferredOne = async { taskOne(700L, "Result from Task One") }
                val deferredTwo = async { taskTwo(200L, "Result from Task Two") }
                val fastResult = select {
                    deferredOne.onAwait { it }
                    deferredTwo.onAwait { it }
                }
               fastResult
            }
        }
        println("First result: $result")
    }

    job.join()
}

```

In this snippet, we use `withTimeoutOrNull` to ensure the whole operation doesn't get stuck if neither task completes in time. Within it, we use a `select` expression to retrieve the result from whichever async function finishes first. If `taskTwo` completes before 500ms, it will print its result, otherwise, it will print null due to the timeout in the `withTimeoutOrNull` block. `deferredOne` might still be working in the background, but our main process is not concerned with it.

**Second Approach: Using Asynchronous Tasks and a First-to-Complete Queue (Python Example - Asyncio)**

Here, we'll demonstrate the approach with asyncio in Python. It showcases how to monitor multiple asynchronous tasks and retrieve the result of the first one that completes:

```python
import asyncio

async def task_one(delay, result):
    await asyncio.sleep(delay)
    print(f"Task One Completed with {result}")
    return result

async def task_two(delay, result):
    await asyncio.sleep(delay)
    print(f"Task Two Completed with {result}")
    return result

async def main():
    task1 = asyncio.create_task(task_one(0.3, "Result from Task One"))
    task2 = asyncio.create_task(task_two(0.1, "Result from Task Two"))

    done, _ = await asyncio.wait({task1, task2}, return_when=asyncio.FIRST_COMPLETED)

    if done:
        first_task = done.pop()
        result = first_task.result()
        print(f"First result: {result}")
        #Cancel pending task
        for task in {task1, task2} - done:
           task.cancel()
    else:
       print("No task completed.")


if __name__ == "__main__":
    asyncio.run(main())
```

In this Python example, we create two tasks using `asyncio.create_task`. Then `asyncio.wait` is used with the parameter `return_when=asyncio.FIRST_COMPLETED` to retrieve the completed task. `done` will contain the completed task. We then retrieve the result from that first task, and cancel all the other tasks that might be running. Note that while `wait` can timeout using `timeout` argument, I kept the explanation simple for demonstration purpose.

**Third Approach: Using `Promise.race()` (Javascript Example - Async/Await)**

For the JavaScript folks out there, the `Promise.race()` method is very handy for this specific problem. It's an elegant way to express the race condition between multiple asynchronous operations.

```javascript
async function taskOne(delay, result) {
    await new Promise(resolve => setTimeout(resolve, delay));
    console.log(`Task One Completed with ${result}`);
    return result;
}

async function taskTwo(delay, result) {
    await new Promise(resolve => setTimeout(resolve, delay));
    console.log(`Task Two Completed with ${result}`);
    return result;
}

async function main() {
    try{
    const firstResult = await Promise.race([
        taskOne(700, "Result from Task One"),
        taskTwo(200, "Result from Task Two")
    ]);
    console.log(`First result: ${firstResult}`);
    } catch(error){
        console.error("Error during execution: ", error);
    }
}

main();
```

In this JavaScript example, we use `Promise.race()` to await the promise that settles first (either resolves or rejects). If any one of the provided promises settles (either with success or failure), then that is what the return of `Promise.race` will settle with. The first promise to resolve or reject dictates the settling state of `Promise.race`. This works perfectly in your use case.

**Key Considerations and Further Reading**

When using these constructs, always think about error handling. While the examples are simple for clarity, in a real-world application you would want to handle potential exceptions from the tasks that don't finish first. Additionally, cancellation is crucial. It’s good practice to always have a mechanism to properly cancel or interrupt any long-running or pending operation when its result is no longer required.

For a deeper understanding of concurrency and related patterns, I highly recommend these resources:

*   **"Concurrent Programming in Java: Design Principles and Patterns" by Doug Lea:** While focused on Java, the principles and patterns discussed are broadly applicable to concurrent programming in any language.
*   **"Programming in Lua" by Roberto Ierusalimschy:** Chapter 11, on coroutines, provides a solid, language-agnostic understanding of the concept. Lua is a good language to understand fundamental programming concepts because it's minimalistic.
*   **Official language documentation for your specific language:** For example, the Kotlin documentation on coroutines, Python's asyncio documentation, and JavaScript's documentation for `Promise` are extremely helpful.
*   **"Operating System Concepts" by Silberschatz, Galvin, and Gagne:** This text gives a deep theoretical understanding of concurrency and synchronization. Though it's operating system-focused, it’s beneficial for building robust applications.

In summary, waiting for the first of multiple suspended functions to complete can be elegantly handled with techniques like `withTimeoutOrNull` and `select` (kotlin), `asyncio.wait` with `FIRST_COMPLETED` (python), or `Promise.race()` (javascript). The chosen method will largely depend on your language's coroutine or async mechanisms, but the fundamental concept remains consistent across platforms: gracefully manage multiple tasks, and prioritize the outcome of the first one to complete. I hope this helps, and feel free to ask if you have any more specific scenarios in mind.
