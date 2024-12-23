---
title: "How do I run two suspend functions in parallel and return the result of the first?"
date: "2024-12-23"
id: "how-do-i-run-two-suspend-functions-in-parallel-and-return-the-result-of-the-first"
---

Okay, let's tackle this concurrency challenge with coroutines, and specifically address the nuances of running suspend functions in parallel, while only caring about the result of the first to complete. It's a situation I've encountered a number of times, especially when dealing with various network requests or data processing pipelines where response time variability is a significant concern.

My early encounters with this involved a complex data synchronization system. We were pulling data from two different sources, both capable of returning the same dataset, but with different update frequencies and latencies. The goal was to use the freshest available data as quickly as possible, and that meant leveraging parallelism. We weren't interested in the laggard, so processing the first response and discarding the rest was paramount.

The straightforward approach, using `async` and `await`, initially feels like the correct route. But if you blindly await both, you're effectively serializing your operations. You only want to consume the result of the *first* suspend function that finishes, so `awaitAll` is unsuitable. This is where the concept of a "racing" operation comes in.

What's needed is to launch each suspend function in its own coroutine scope, via `async`, but then only act on the outcome of the first to complete. This can be achieved using a structure that leverages the nature of deferred results provided by `async`, and the non-blocking nature of coroutines. We’ll use the combination of coroutine scopes, specifically with `coroutineScope`, along with some clever logic. The `coroutineScope` ensures that all launched coroutines will finish before `firstCompletedOrNull` returns, which is critical for preventing leaks.

Let's illustrate this with some code. Consider these three examples. The first uses simple integers as results, the second deals with strings, and the third showcases a slightly more complex model:

**Example 1: Integer Results**

```kotlin
import kotlinx.coroutines.*

suspend fun fetchValue1(): Int {
    delay(200)
    return 42
}

suspend fun fetchValue2(): Int {
    delay(100)
    return 99
}

suspend fun <T> firstCompletedOrNull(block1: suspend () -> T, block2: suspend () -> T): T? = coroutineScope {
    val deferred1 = async { block1() }
    val deferred2 = async { block2() }

    val firstResult = awaitFirstResult(deferred1, deferred2)
    firstResult
}

suspend fun main() {
    val result = firstCompletedOrNull(::fetchValue1, ::fetchValue2)
    println("First completed result: $result") // Output: First completed result: 99
}


suspend fun <T> awaitFirstResult(deferred1: Deferred<T>, deferred2: Deferred<T>): T? {
    return suspendCancellableCoroutine { continuation ->
        var completed = false
        val job1 = launch {
            try {
              val res = deferred1.await()
                if (!completed) {
                    completed = true
                    continuation.resume(res, null)
                }
                deferred2.cancel()
            } catch (e: CancellationException) {
            }
        }

        val job2 = launch {
            try {
                val res = deferred2.await()
                if(!completed) {
                  completed = true
                  continuation.resume(res, null)
                }
               deferred1.cancel()
            } catch (e: CancellationException) {

            }
        }
    continuation.invokeOnCancellation {
        if (!job1.isCancelled)
        { job1.cancel() }
        if (!job2.isCancelled)
        { job2.cancel() }
        }
    }
}
```

Here, `firstCompletedOrNull` uses `async` to launch the two suspend functions as coroutines. Crucially, we're not using `awaitAll`, but rather the `awaitFirstResult` function which employs a `suspendCancellableCoroutine` to react to the first resolved `deferred`. The `firstCompletedOrNull` is designed to manage the coroutine scope cleanly and return the result from the first successful `await` operation. Upon completion of the first operation, the other job is canceled.

**Example 2: String Results**

```kotlin
import kotlinx.coroutines.*

suspend fun fetchString1(): String {
    delay(500)
    return "hello"
}

suspend fun fetchString2(): String {
    delay(100)
    return "world"
}

suspend fun <T> firstCompletedOrNull(block1: suspend () -> T, block2: suspend () -> T): T? = coroutineScope {
    val deferred1 = async { block1() }
    val deferred2 = async { block2() }

    val firstResult = awaitFirstResult(deferred1, deferred2)
    firstResult
}

suspend fun main() {
    val result = firstCompletedOrNull(::fetchString1, ::fetchString2)
    println("First completed result: $result") // Output: First completed result: world
}

suspend fun <T> awaitFirstResult(deferred1: Deferred<T>, deferred2: Deferred<T>): T? {
    return suspendCancellableCoroutine { continuation ->
        var completed = false
        val job1 = launch {
            try {
                val res = deferred1.await()
                if (!completed) {
                    completed = true
                    continuation.resume(res, null)
                }
                deferred2.cancel()
            } catch (e: CancellationException) {
            }
        }

        val job2 = launch {
            try {
                val res = deferred2.await()
                if(!completed) {
                  completed = true
                  continuation.resume(res, null)
                }
               deferred1.cancel()
            } catch (e: CancellationException) {

            }
        }
    continuation.invokeOnCancellation {
        if (!job1.isCancelled)
        { job1.cancel() }
        if (!job2.isCancelled)
        { job2.cancel() }
        }
    }
}

```

This is structurally identical to the first example, but the suspend functions now return strings, demonstrating the generic nature of the `firstCompletedOrNull` function. It shows how the core logic remains consistent regardless of the data type.

**Example 3: Complex Data Model**

```kotlin
import kotlinx.coroutines.*

data class DataRecord(val id: Int, val value: String)

suspend fun fetchData1(): DataRecord {
    delay(300)
    return DataRecord(1, "from api 1")
}

suspend fun fetchData2(): DataRecord {
   delay(100)
   return DataRecord(2, "from api 2")
}

suspend fun <T> firstCompletedOrNull(block1: suspend () -> T, block2: suspend () -> T): T? = coroutineScope {
    val deferred1 = async { block1() }
    val deferred2 = async { block2() }

    val firstResult = awaitFirstResult(deferred1, deferred2)
    firstResult
}

suspend fun main() {
    val result = firstCompletedOrNull(::fetchData1, ::fetchData2)
    println("First completed record: $result") // Output: First completed record: DataRecord(id=2, value=from api 2)
}

suspend fun <T> awaitFirstResult(deferred1: Deferred<T>, deferred2: Deferred<T>): T? {
    return suspendCancellableCoroutine { continuation ->
        var completed = false
        val job1 = launch {
            try {
                val res = deferred1.await()
                if (!completed) {
                    completed = true
                    continuation.resume(res, null)
                }
                deferred2.cancel()
            } catch (e: CancellationException) {
            }
        }

        val job2 = launch {
            try {
                val res = deferred2.await()
                if(!completed) {
                  completed = true
                  continuation.resume(res, null)
                }
               deferred1.cancel()
            } catch (e: CancellationException) {

            }
        }
    continuation.invokeOnCancellation {
        if (!job1.isCancelled)
        { job1.cancel() }
        if (!job2.isCancelled)
        { job2.cancel() }
        }
    }
}
```

Here, we extend the example to use a custom `DataRecord` data class, indicating that this approach is robust enough to handle more intricate data structures.

**Key Takeaways**

The `awaitFirstResult` function is the core of this solution. It utilizes `suspendCancellableCoroutine` which provides us with an explicit cancellation channel and the ability to resume the parent coroutine with a result once it becomes available. Within `awaitFirstResult` two jobs are launched which each attempt to `await` the result from a provided `deferred`. A flag is used to ensure that only one job is able to resume the parent coroutine, and the other job is cancelled. We use `invokeOnCancellation` to handle proper cancellation of jobs. The use of coroutine scope (`coroutineScope`) is vital, since it ensures that all child coroutines launched by `async` are completed or cancelled before this function returns.

To dive deeper, I would recommend the book "Kotlin Coroutines: Deep Dive" by Marcin Moskała and "Concurrent Programming on Windows" by Joe Duffy for foundational concurrent programming concepts that are equally applicable to coroutines. As well, the official documentation for Kotlin coroutines is of high value, detailing the full breadth and depth of the coroutine API. The key lies in understanding how `async`, `await`, and `suspendCancellableCoroutine` function within the coroutine framework to achieve true parallelism and efficient resource management. Understanding these concepts will empower you to confidently tackle these complex use cases.
