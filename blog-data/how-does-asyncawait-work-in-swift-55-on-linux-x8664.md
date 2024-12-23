---
title: "How does async/await work in Swift 5.5 on Linux x86_64?"
date: "2024-12-23"
id: "how-does-asyncawait-work-in-swift-55-on-linux-x8664"
---

Let's dive into it. I've spent considerable time debugging concurrency issues on Linux servers, particularly when porting some of our iOS code over for backend services using Swift, and understanding how `async`/`await` works, specifically on x86_64, was crucial.

It’s not simply a case of Swift magically working across different architectures. While the language abstracts away many platform-specific details, the underlying implementation of `async`/`await` does interact with the operating system's task scheduling mechanisms, especially on Linux. What I've observed, and what you need to understand, is that Swift's `async`/`await` leverages the underlying concurrency primitives provided by the kernel, often via libraries like `libdispatch` (though not directly, as Swift's concurrency model has its own runtime). It doesn't reinvent the wheel – it builds upon existing, proven systems.

Essentially, `async`/`await` in Swift operates by transforming asynchronous code, which would previously rely heavily on callbacks or completion handlers, into a more straightforward, linear style. Under the hood, this transformation is done by the Swift compiler. When you declare a function as `async`, the compiler generates code that can pause and resume execution. The `await` keyword marks a suspension point, where the current task can be paused, allowing other tasks to execute, and then resumed once the awaited operation completes.

On Linux x86_64, the kernel typically manages threads, and Swift's concurrency runtime works with these threads to achieve parallelism and asynchrony. The specific details can be complex, but the core idea is that when an `await` point is encountered, the current execution context is saved, and the task is placed into a queue. Another task might be selected from this queue to execute, making efficient use of the CPU. Once the awaited operation is complete, the original task is put back on the run queue to continue processing from where it left off.

It's crucial to distinguish this from traditional, callback-heavy asynchronous programming. With callbacks, you have a nested structure of functions, which makes reasoning about program flow difficult. `async`/`await`, however, flattens this out into a sequential, easier-to-follow style. This is a massive improvement for maintainability, especially in complex systems with several interacting asynchronous operations.

Let's illustrate this with a few examples.

**Example 1: Simple Network Request**

Let's say you are using `URLSession` for a basic network request. Historically you would be using completion handlers. However, using `async/await` will make that easier and more readable:

```swift
import Foundation

func fetchData(from urlString: String) async throws -> Data {
    guard let url = URL(string: urlString) else {
        throw NSError(domain: "Invalid URL", code: -1)
    }

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse,
          (200...299).contains(httpResponse.statusCode) else {
        throw NSError(domain: "Invalid Response Code", code: -2)
    }
    return data
}

async func processData() {
    do {
        let data = try await fetchData(from: "https://example.com/data.json")
        print("Data received: \(String(data: data, encoding: .utf8) ?? "")")
    } catch {
        print("Error fetching data: \(error)")
    }
}

Task {
  await processData()
}
```
In this example, `fetchData` is an `async` function, and it uses `await` when fetching the data from the URL. The calling function, `processData`, also needs to be `async` because it uses the `await` keyword for the `fetchData`. Notice how the execution pauses at the `await` and doesn’t block the main thread. The execution will be resumed later when the fetch has finished.

**Example 2: Concurrent Processing**

Now, let’s examine a scenario where multiple network requests need to be processed concurrently:

```swift
import Foundation

func fetchUser(id: Int) async throws -> String {
    let urlString = "https://api.example.com/users/\(id)"
    let data = try await fetchData(from: urlString)
    return String(data: data, encoding: .utf8) ?? "Empty data for user \(id)"
}


func processUsers() async {
    async let user1 = fetchUser(id: 1)
    async let user2 = fetchUser(id: 2)
    async let user3 = fetchUser(id: 3)

    do {
        let users = try await [user1, user2, user3]
        print("User data received: \(users)")
    } catch {
        print("Error fetching users: \(error)")
    }
}

Task {
  await processUsers()
}
```
Here, `async let` is used to start each `fetchUser` request concurrently. The `await [user1, user2, user3]` gathers the results when all the requests finish. This highlights the non-blocking behavior and concurrent capabilities provided by the `async`/`await` feature. All three user requests start almost immediately, and they will be resolved in no particular order, depending on how fast each one completes.

**Example 3: Performing CPU Bound Operations**

Finally, we'll look at a use case where heavy CPU computations are involved. It's important to understand that although `async/await` is non-blocking, CPU-bound operations can still block the execution if they are not done correctly:

```swift

func computeHeavyTask(value: Int) async -> Int {
    var result = value
    for _ in 0..<1_000_000 {
        result = result.bitWidth + 3
    }
    return result
}

async func calculateSums() {
    async let sum1 = computeHeavyTask(value: 10)
    async let sum2 = computeHeavyTask(value: 20)
    async let sum3 = computeHeavyTask(value: 30)

   let result = await [sum1,sum2,sum3]
    print("Sum results are \(result)")
}


Task{
    await calculateSums()
}

```

Here each sum calculation is running concurrently. While `async/await` will allow to manage all of these tasks concurrently in a non-blocking way, they do still use CPU. The operating system and Swift's runtime will be managing and assigning threads to these tasks, which will try to balance the load and not block other important parts of the application.

It’s worth noting that while `async`/`await` provides a more structured way to handle asynchronous operations, it doesn't inherently solve all concurrency issues. Race conditions, deadlocks, and other problems can still occur if the code is not designed carefully.

For a deeper understanding of Swift's concurrency model, I'd strongly recommend reviewing the documentation and session videos from Apple’s Worldwide Developers Conference. I particularly found the documentation on structured concurrency to be very helpful when debugging issues on Linux. Additionally, "Concurrency in Swift" by Vadim Bulavin provides a solid technical deep dive. Understanding concepts such as thread pools and how dispatch queues work is also paramount, and you can find in-depth analysis in Operating System books such as "Modern Operating Systems" by Andrew S. Tanenbaum.

In conclusion, `async`/`await` in Swift 5.5 on Linux x86_64 (and really on any architecture Swift supports) simplifies asynchronous programming by transforming asynchronous code into a more linear style. It leverages the operating system's kernel to manage tasks and ensure non-blocking execution. It's a powerful tool, but as with any concurrency mechanism, it requires careful consideration of its underlying mechanics and potential pitfalls. It is not a magical fix and it does not make all code concurrent, nor should it. It is important to consider the execution context and the type of work being done.
