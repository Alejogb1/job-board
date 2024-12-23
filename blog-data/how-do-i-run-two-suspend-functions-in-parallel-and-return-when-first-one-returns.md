---
title: "How do I Run two suspend functions in parallel and return when first one returns?"
date: "2024-12-23"
id: "how-do-i-run-two-suspend-functions-in-parallel-and-return-when-first-one-returns"
---

,  I’ve seen this particular problem pop up quite a bit in asynchronous systems, especially when dealing with microservices or data pipelines that rely on concurrent operations. Effectively, you're asking how to launch two coroutines that execute concurrently, but only proceed once *either* of them has completed, disregarding the other's eventual result. The trick here is knowing which kotlin coroutine primitives to use, and understanding their behaviors.

I recall back in my early days on a large data processing team, we were building an ETL system for real-time log ingestion. One crucial component needed to fetch metadata about incoming logs from two separate sources. The first source was typically faster but occasionally experienced latency spikes, while the second was slower but more consistent. The requirement was to immediately use whichever metadata source responded first, abandoning the slower one. This is precisely the scenario we’re discussing, and the standard `async/await` pattern wouldn’t fit because we didn't need *both* results, just the first one.

The key is using `select` expression with coroutines. The `select` expression allows us to race multiple suspendable operations, picking the result of the first one that completes. If you're familiar with the concept of channels, you'll notice `select` works in a conceptually similar way, but it's not limited to just channels; it can be applied to arbitrary suspend functions.

Before we get into the code, a crucial thing to keep in mind is exception handling within the `select` expression. If an exception occurs in *any* of the coroutines, that exception will propagate up unless handled within each coroutine's scope or within the `select` expression, and there won’t be a second chance to recover from other coroutines. The other coroutines might continue, or they might get cancelled, depending on their implementation, but the result won’t propagate.

Here's our first code snippet which demonstrates a basic implementation:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.selects.*

suspend fun fetchDataFromSource1(): String {
    delay(150) // Simulating a faster response
    return "Data from Source 1"
}

suspend fun fetchDataFromSource2(): String {
    delay(300) // Simulating a slower response
    return "Data from Source 2"
}

suspend fun fetchFirstAvailableData(): String = coroutineScope {
    val deferred1 = async { fetchDataFromSource1() }
    val deferred2 = async { fetchDataFromSource2() }

    select {
        deferred1.onAwait { it } // Access the result of deferred1
        deferred2.onAwait { it } // Access the result of deferred2
    }
}

fun main() = runBlocking {
    val result = fetchFirstAvailableData()
    println("First available data: $result") // Expected output is "First available data: Data from Source 1"
}
```

In this example, `fetchDataFromSource1` and `fetchDataFromSource2` are simulated asynchronous functions using `delay`. Within `fetchFirstAvailableData`, we launch them as separate coroutines using `async`, creating `deferred` results. The `select` expression then waits on both of these deferreds via `onAwait`. The first `onAwait` to complete will return its result, and that becomes the final result of the `select` expression, effectively cancelling the other `deferred` .

Now, let's delve into a slightly more practical case, where we need to handle potential errors gracefully. Imagine we are fetching user data from different APIs, each susceptible to temporary failures. The following code snippet demonstrates error handling with a `try-catch` block inside each coroutine, before the select:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.selects.*

suspend fun fetchUserDataFromAPI1(): String {
  delay(100)
    if (Math.random() > 0.3) return "User Data from API 1"
    else throw RuntimeException("API 1 Failed")

}

suspend fun fetchUserDataFromAPI2(): String {
    delay(300)
    if (Math.random() > 0.1) return "User Data from API 2"
    else throw RuntimeException("API 2 Failed")
}

suspend fun fetchUserFastest(): String = coroutineScope {
    val deferred1 = async {
        try {
            fetchUserDataFromAPI1()
        } catch (e: Exception) {
            println("API 1 threw an exception: ${e.message}")
            null
        }
    }

    val deferred2 = async {
        try {
            fetchUserDataFromAPI2()
        } catch (e: Exception) {
           println("API 2 threw an exception: ${e.message}")
            null
        }
    }

    select {
        deferred1.onAwait { it }
        deferred2.onAwait { it }
    }?: "No Data available"

}

fun main() = runBlocking {
    val result = fetchUserFastest()
    println("Fastest user data received: $result")
}
```

Here, both `fetchUserDataFromAPI1` and `fetchUserDataFromAPI2` can potentially throw exceptions. We wrap each call inside an async block with a try catch block. The `null` value means the select block will proceed only with non null results. If none of the results are non-null, the `?:` (Elvis Operator) will return "No Data Available".

Lastly, let's consider a scenario where not all operations return a simple value. Suppose we’re processing sensor data, and each sensor's processing function not only returns data, but also provides a sensor identifier. In the following example, we'll see how to adapt the previous code to handle this case:

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.selects.*

data class SensorData(val sensorId: String, val data: String)

suspend fun processSensorData1(): SensorData {
  delay(200)
    return SensorData("sensor1", "Processed data from Sensor 1")
}

suspend fun processSensorData2(): SensorData {
   delay(400)
    return SensorData("sensor2", "Processed data from Sensor 2")
}

suspend fun getFirstAvailableSensorData(): SensorData = coroutineScope {
    val deferred1 = async { processSensorData1() }
    val deferred2 = async { processSensorData2() }

    select {
        deferred1.onAwait { it }
        deferred2.onAwait { it }
    }
}

fun main() = runBlocking {
    val sensorData = getFirstAvailableSensorData()
    println("First available data from sensor: ${sensorData.sensorId}, Data: ${sensorData.data}")
}
```

This last example demonstrates that you can use `select` with more complex return types. Each of the `processSensorData` functions returns an instance of `SensorData`.

For a deeper understanding of coroutines, I'd recommend reading "kotlin coroutines Deep Dive" by Marcin Moskala. It's a detailed examination of the underpinnings of coroutines in Kotlin. Also, the official Kotlin documentation on coroutines is an indispensable reference point. You’ll find detailed explanations about `select` and other coroutine building blocks there. Another book I often recommend is "Concurrent Programming on Windows" by Joe Duffy, although it's not Kotlin-specific, it gives a great conceptual overview of concurrent programming and patterns, which are helpful when considering more complex scenarios. Lastly, consider looking at the “Reactive Streams Specification” for patterns in reactive programming.

So to summarise, using the `select` expression in combination with `async/await` provides an effective way to run suspend functions in parallel and retrieve the first available result, along with the ability to gracefully manage exceptions, and different return data types. This pattern is very valuable in situations where speed is paramount and waiting for all asynchronous operations is unnecessary or undesirable, as demonstrated in the use cases we've touched on.
