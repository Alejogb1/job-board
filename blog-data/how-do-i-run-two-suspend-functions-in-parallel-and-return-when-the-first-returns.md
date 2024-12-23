---
title: "How do I run two suspend functions in parallel and return when the first returns?"
date: "2024-12-23"
id: "how-do-i-run-two-suspend-functions-in-parallel-and-return-when-the-first-returns"
---

Okay, let's talk about running suspend functions in parallel and returning the result of the first one that completes – a common task in asynchronous programming, and one I've certainly tackled more times than I'd care to count. It's more nuanced than simply launching two coroutines and calling it a day, especially when you need that specific "first to finish" behavior. In my experience, this kind of pattern often crops up when dealing with multiple data sources, or fallback mechanisms.

The core concept revolves around launching the suspend functions within their own coroutines and then employing the power of kotlin’s `async` and `await` mechanisms (or, as a similar analogy from other asynchronous patterns, its equivalent of promises and `await`). The trick, however, isn't just to launch them concurrently; it’s to monitor their results and cancel the slower one once the first completes. This requires a structured approach to coroutine management. Let's get into it, step by step, with code to illustrate.

I recall a project a few years back, we were pulling pricing information from several different APIs. Each API was somewhat unreliable, and we needed to ensure we got a price quickly. Waiting for all the APIs to complete would have introduced unnecessary latency, and sometimes even failed completely due to timeouts. So, our solution, which I'll mirror here with slightly modified examples, employed a 'first-to-finish' strategy.

The problem essentially breaks down into these stages: 1) define suspend functions; 2) Launch coroutines for each with `async`; 3) Await the first to complete and return the results; 4) Manage cancellations to avoid wasted resources.

Let's start with the code examples:

**Example 1: Basic Parallel Execution and First-to-Return**

This example demonstrates the fundamental mechanism. Note that, even if both are `launch`, the `async` block uses await which provides a structured output in the order they are awaited, not necessarily the order they finish.

```kotlin
import kotlinx.coroutines.*
import kotlin.random.Random

suspend fun fetchData(id: Int, delay: Long): String {
    delay(delay)
    return "Data $id: Completed in $delay ms"
}

suspend fun firstToReturn(): String {
    return coroutineScope {
        val deferred1 = async { fetchData(1, Random.nextLong(100, 500)) }
        val deferred2 = async { fetchData(2, Random.nextLong(100, 500)) }

        // Await the first to complete
        val result = withTimeout(1000) {
           awaitAny(listOf(deferred1, deferred2))
        }


        // Cancel the other jobs to avoid wasted resources
        if(deferred1.isActive && deferred1 != result){
            deferred1.cancel()
        }
        if(deferred2.isActive && deferred2 != result){
            deferred2.cancel()
        }


        return@coroutineScope result.await()

    }
}

suspend fun main() = runBlocking {
    val result = firstToReturn()
    println("First result: $result")

}

suspend fun <T> awaitAny(deferreds: List<Deferred<T>>) : Deferred<T> {
    return coroutineScope {
      val firstCompleted = CompletableDeferred<Deferred<T>>()
       deferreds.forEach { deferred ->
           launch {
             deferred.await()
             firstCompleted.complete(deferred)
           }
       }

       firstCompleted.await()
    }
}
```

Here, `firstToReturn` is our core function. We define `fetchData` to simulate our async calls. Inside `firstToReturn`, we launch two coroutines using `async`, each executing `fetchData` with different delays. `awaitAny`, which I'll break down separately, watches their result and return the deferred variable associated with the first completed function. If timeout was reached and no result was received, `withTimeout` will throw an exception. Crucially, upon finding the first result, it cancels the other jobs, saving computational resources. This behavior is important in real-world apps, especially with network requests.

**Example 2: Using a Timeout with cancellation**

Building on example 1, what if the coroutines are taking too long?

```kotlin
import kotlinx.coroutines.*
import kotlin.random.Random

suspend fun fetchData(id: Int, delay: Long): String {
    delay(delay)
    return "Data $id: Completed in $delay ms"
}

suspend fun firstToReturn(): String? {
    return coroutineScope {
        val deferred1 = async { fetchData(1, Random.nextLong(100, 500)) }
        val deferred2 = async { fetchData(2, Random.nextLong(100, 500)) }

        // Await the first to complete, but with a timeout
        val resultDeferred = try {
             withTimeout(300) {
               awaitAny(listOf(deferred1, deferred2))
             }
        } catch(e: TimeoutCancellationException){
             println("Timeout occurred")
            null
        }


        if (resultDeferred != null) {
            if(deferred1.isActive && deferred1 != resultDeferred){
                deferred1.cancel()
            }
             if(deferred2.isActive && deferred2 != resultDeferred){
               deferred2.cancel()
             }
           return@coroutineScope resultDeferred.await()
        } else {
           deferred1.cancel()
           deferred2.cancel()
            return@coroutineScope null
        }
    }
}


suspend fun main() = runBlocking {
   val result = firstToReturn()
   if(result != null) {
       println("First result: $result")
   }else{
       println("no result")
   }

}
suspend fun <T> awaitAny(deferreds: List<Deferred<T>>) : Deferred<T> {
    return coroutineScope {
      val firstCompleted = CompletableDeferred<Deferred<T>>()
       deferreds.forEach { deferred ->
           launch {
             deferred.await()
             firstCompleted.complete(deferred)
           }
       }

       firstCompleted.await()
    }
}
```

Here I have added a timeout using `withTimeout` and handling the exception. Now if the calls are taking too long, you can return a default result or some kind of error, if necessary.

**Example 3: Error Handling**

Finally, we need to deal with potential errors within the suspended functions:

```kotlin
import kotlinx.coroutines.*
import kotlin.random.Random

suspend fun fetchDataWithError(id: Int, delay: Long, shouldThrowError: Boolean): String {
    delay(delay)
    if (shouldThrowError) {
        throw RuntimeException("Error fetching data for $id")
    }
    return "Data $id: Completed in $delay ms"
}


suspend fun firstToReturn(): String? {
    return coroutineScope {
        val deferred1 = async { fetchDataWithError(1, Random.nextLong(100, 500), false) }
        val deferred2 = async { fetchDataWithError(2, Random.nextLong(100, 500), true) }


        val resultDeferred = try {
            withTimeout(300) {
                awaitAny(listOf(deferred1, deferred2))
            }
        } catch(e: TimeoutCancellationException){
           println("Timeout Exception")
           null
        }

        if (resultDeferred != null) {
           if(deferred1.isActive && deferred1 != resultDeferred){
               deferred1.cancel()
           }
            if(deferred2.isActive && deferred2 != resultDeferred){
               deferred2.cancel()
           }
          return@coroutineScope try{
              resultDeferred.await()
          } catch (e: Exception){
              println("Exception inside deferred await")
              null
          }
        } else {
             deferred1.cancel()
            deferred2.cancel()
            return@coroutineScope null
        }
    }
}


suspend fun main() = runBlocking {
    val result = firstToReturn()
    if (result != null) {
        println("First result: $result")
    }else{
        println("No result")
    }
}
suspend fun <T> awaitAny(deferreds: List<Deferred<T>>) : Deferred<T> {
    return coroutineScope {
      val firstCompleted = CompletableDeferred<Deferred<T>>()
       deferreds.forEach { deferred ->
           launch {
             try {
                 deferred.await()
                 firstCompleted.complete(deferred)
             } catch(e: Exception){
                  //we want to ignore this exception because we want the first success, not fail.
                 // We do not complete the future because it's not a success, the await will catch the exception and cancel the job.
             }
           }
       }

       firstCompleted.await()
    }
}
```

In this version, I added a `shouldThrowError` parameter to `fetchDataWithError`. If set to true it throws a runtime exception.  The `awaitAny` function has been modified to handle exceptions by completing the future when the deferred variable succeeds, and ignoring it otherwise. The `resultDeferred.await` also contains a try catch to handle the exceptions thrown by the await call.

The cancellation of the other `async` operations ensures that you aren’t wasting resources once you’ve got the result you were looking for.

**Key Takeaways and Further Reading**

*   **Structured Concurrency:** Kotlin's coroutines provide a structured concurrency model, which simplifies the management of these kinds of scenarios. The use of `coroutineScope` is fundamental for ensuring that all launched coroutines are properly tracked and cancelled.
*   **Cancellation:** Manual cancellation using the `cancel()` method on `Job` is necessary to free up resources when a solution is found.
*   **Error Handling:** It is crucial to plan how you will handle errors within suspend functions launched in coroutines. The `try...catch` blocks are instrumental for managing exceptions thrown during the awaited actions.

For a deeper dive, I would recommend exploring the following:

*   **"Kotlin Coroutines Deep Dive" by Marcin Moskala:** A comprehensive book dedicated to Kotlin coroutines, covering advanced use cases and internals. This is a good resource for understanding the behavior and inner workings.
*   **Official Kotlin Coroutines Documentation:** The documentation provides the most authoritative guide to the behavior of the language features. It's essential for a proper understanding.
*   **"Concurrent Programming on Windows" by Joe Duffy:** While not specific to Kotlin, it presents core concepts of concurrency that are applicable everywhere, especially regarding structured concurrency and cancellation.

These resources should give you a strong base to tackle complex asynchronous patterns. Remember, proper error handling, cancellation of other running routines, and the core concepts of structured concurrency are crucial when operating on async routines, and the above code and reading materials should be invaluable on this journey.
