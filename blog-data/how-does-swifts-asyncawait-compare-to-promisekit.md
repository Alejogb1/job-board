---
title: "How does Swift's async/await compare to PromiseKit?"
date: "2024-12-23"
id: "how-does-swifts-asyncawait-compare-to-promisekit"
---

Alright, let's unpack this. It’s a comparison I’ve actually navigated quite a few times, particularly during a project involving a large-scale data migration pipeline a few years back. We initially leaned heavily on PromiseKit, then gradually transitioned to Swift’s `async`/`await` as the language evolved. My experience with both systems gives me some practical perspective on their differences, and ultimately, their place in modern iOS and macOS development.

Fundamentally, both PromiseKit and `async`/`await` aim to address the same core challenge: managing asynchronous operations. This was a real headache before structured concurrency tools emerged. Callbacks, the old approach, frequently led to nested code pyramids – the infamous "callback hell" – that were difficult to understand, debug, and maintain. PromiseKit offered an elegant solution to that, introducing a cleaner, more linear way to reason about asynchronous flows. You chain asynchronous operations using `then`, `catch`, and similar methods, effectively transforming complex, deeply nested callback structures into something more readable.

Swift’s `async`/`await` is an evolution of that idea, building asynchronous operations directly into the language itself. It's not an external library like PromiseKit, but an intrinsic part of Swift’s concurrency model. This integration means the compiler has far more insight into how these operations are constructed, allowing for more precise error handling and potentially better performance optimization. In essence, `async`/`await` allows you to write asynchronous code that looks and behaves much like synchronous code, removing much of the overhead associated with the promise pattern and callbacks.

One key difference lies in how they handle errors. PromiseKit uses `.catch` blocks at various points in a promise chain to handle errors originating within the preceding promise. While effective, it can still lead to some degree of verbosity, especially if errors need to be dealt with in a more nuanced way, or if different parts of the chain require different error handling mechanisms. With Swift’s `async`/`await`, we primarily use the standard `do-catch` mechanism within an asynchronous context. Errors that are explicitly thrown within an `async` function become an intrinsic part of the control flow, making error handling more predictable and straightforward.

Here's a simple example to illustrate this. Imagine we have a function that fetches user data from an API, then another function to process that data:

**Example 1: Using PromiseKit**

```swift
import PromiseKit

func fetchUserDataPromise(userId: Int) -> Promise<[String: Any]> {
    return Promise { seal in
        // Simulating an async network call
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) {
            let userData = ["id": userId, "name": "John Doe", "email": "john.doe@example.com"]
            seal.fulfill(userData)
        }
    }
}

func processUserDataPromise(userData: [String: Any]) -> Promise<String> {
    return Promise { seal in
         DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) {
             if let name = userData["name"] as? String {
                seal.fulfill("Processed user: \(name)")
            } else {
                seal.reject(NSError(domain: "UserDataError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid user data"]))
            }

         }
    }

}

func executePromiseFlow(userId: Int) {
    fetchUserDataPromise(userId: userId)
        .then { userData in
            processUserDataPromise(userData: userData)
        }
        .done { processedData in
            print("PromiseKit result: \(processedData)")
        }
        .catch { error in
            print("PromiseKit error: \(error.localizedDescription)")
        }
}

// Call the function
executePromiseFlow(userId: 123)
```

Now, let’s look at the equivalent using `async`/`await`:

**Example 2: Using Swift async/await**

```swift
func fetchUserDataAsync(userId: Int) async throws -> [String: Any] {
     return try await withCheckedThrowingContinuation { continuation in
           // Simulate an async network call
           DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) {
            let userData = ["id": userId, "name": "John Doe", "email": "john.doe@example.com"]
            continuation.resume(returning: userData)

        }
     }
}

func processUserDataAsync(userData: [String: Any]) async throws -> String {
   return try await withCheckedThrowingContinuation { continuation in
          DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) {
              if let name = userData["name"] as? String {
                continuation.resume(returning: "Processed user: \(name)")
            } else {
                continuation.resume(throwing: NSError(domain: "UserDataError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid user data"]))
            }
        }
    }
}


func executeAsyncFlow(userId: Int) async {
    do {
        let userData = try await fetchUserDataAsync(userId: userId)
        let processedData = try await processUserDataAsync(userData: userData)
        print("Async/await result: \(processedData)")
    } catch {
        print("Async/await error: \(error.localizedDescription)")
    }
}


Task {
    await executeAsyncFlow(userId: 123)
}
```

Notice how the `async`/`await` version reads more like a synchronous process. The error handling is accomplished through a `do-catch` block, which is a familiar pattern in Swift. You can see how the code looks clearer and there isn't a need to navigate the `then`, `catch` structure that PromiseKit requires. The `withCheckedThrowingContinuation` is required in this example because the code simulates asynchronous calls via GCD, but in practice you'd use native async methods.

Another significant difference is cancellation. PromiseKit has explicit methods and structures to support promise cancellation, involving the use of cancellation tokens and checks within the promises. With `async`/`await`, cancellation is handled more implicitly via cooperative cancellation. If an asynchronous task is canceled (e.g., because the user navigates away from the view), the framework signals the cancellation to cooperating functions, allowing them to clean up resources and exit gracefully. This avoids much of the boilerplate code found in handling cancellation with PromiseKit.

To make it concrete, let's consider how cancellation might look in both scenarios. We'll build on the previous examples by introducing a cancellable network request simulator.

**Example 3: PromiseKit with cancellation**
```swift
import PromiseKit
import Foundation

class CancellableRequest {
    var isCancelled = false
    
    func fetchCancellableData() -> Promise<String> {
      return Promise { seal in
           DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
             if !self.isCancelled {
                seal.fulfill("Data received")
            } else {
                seal.reject(NSError(domain: "CancelledError", code: -999, userInfo: [NSLocalizedDescriptionKey: "Request was cancelled"]))
            }

         }
        
      }

    }


    func cancel(){
        isCancelled = true
    }
}

func runCancellablePromise() {
    let request = CancellableRequest()
    let promise = request.fetchCancellableData()
        .done { data in
            print("PromiseKit Cancellable Data: \(data)")
        }
        .catch { error in
            print("PromiseKit Cancellation Error: \(error.localizedDescription)")
        }
   DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
       request.cancel()
   }
}


// Call
runCancellablePromise()
```

**Example 4: Async/Await with cancellation**

```swift
import Foundation

func fetchCancellableDataAsync() async throws -> String {
   return try await withCheckedThrowingContinuation { continuation in
        
        let task =  DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
             if !Task.isCancelled {
                continuation.resume(returning: "Data received")
             } else {
                 continuation.resume(throwing:  NSError(domain: "CancelledError", code: -999, userInfo: [NSLocalizedDescriptionKey: "Request was cancelled"]))
              }
        }
        continuation.onCancellation = {
            task.cancel()
        }
    }

}


func runCancellableAsync() async {
     do {
          let data = try await fetchCancellableDataAsync()
          print("Async/Await Cancellable Data: \(data)")
        } catch {
            print("Async/Await Cancellation Error: \(error.localizedDescription)")
      }

}

Task {
    let task = Task {
        await runCancellableAsync()
    }
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
        task.cancel()
    }
}

```

In the async/await version, there’s no need for explicit `isCancelled` flags or manual checks within the data fetching. `Task.isCancelled` signals cancellation through the concurrency runtime. In both scenarios, the cancellation happens mid-way through the call, but the async/await handles it more cleanly and directly.

For more in-depth knowledge, I'd recommend looking at the following resources:

*   **“Concurrency Programming with Swift” by Apple:** This is Apple's official guide to understanding Swift’s structured concurrency model, including `async`/`await`. It's the definitive resource for understanding how it works at the language level.
*   **"Effective Concurrency in Swift" by Sundell:** This resource goes into a lot of detail on best practices for using Swift concurrency and provides great insight into handling edge cases and optimizing concurrent code.

In conclusion, while PromiseKit was a valuable tool that significantly improved asynchronous programming practices in Swift, `async`/`await` is a significant advancement. It’s more integrated, more readable, and typically more efficient. Although it took time, migrating to `async`/`await` resulted in a better codebase that was easier to maintain, and I find myself rarely reaching for promise-based solutions these days, unless specifically working with legacy codebases.
