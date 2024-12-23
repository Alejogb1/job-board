---
title: "Why are swift non actor isolated closures causing issues?"
date: "2024-12-23"
id: "why-are-swift-non-actor-isolated-closures-causing-issues"
---

Let's tackle this one. I recall a rather frustrating debugging session back in my days working on a large scale mobile application, which involved a very similar issue. We were aggressively adopting swift's concurrency model, and initially, things looked incredibly promising. However, subtle bugs started creeping in that seemed almost impossible to pinpoint. The root cause, more often than not, was indeed how non-actor isolated closures interacted with the broader concurrency landscape. It's a topic that often gets glossed over but can lead to severe headaches if not understood well.

The fundamental problem arises from the implicit contract surrounding closures in swift, particularly those that are not explicitly actor-isolated. When a closure is not marked as `@MainActor` or part of an actor's isolated context, it defaults to being non-isolated. The term "non-isolated" might seem straightforward, but its behavior within a concurrent system isn't always intuitive. Effectively, it means the closure doesn’t inherently belong to any specific execution context. It can be invoked from any thread, any actor, and at any time, without guarantees of mutual exclusivity, which is a real problem when we start interacting with shared mutable state.

To illustrate, imagine a scenario where we have a shared counter – something ridiculously common. Now, let’s say we pass a non-isolated closure to a concurrent operation that intends to update this counter. Since the closure is non-isolated, there's no implicit synchronization mechanism to prevent race conditions when different parts of our code invoke this closure from various contexts concurrently. It's not about the operation executing, it's about the context the closure resides in. This can lead to inconsistent values and unpredictable application behavior.

Here's a simplified example. This first snippet is the problem case, without proper actor isolation:

```swift
import Foundation

var sharedCounter = 0

func incrementCounter(completion: @escaping () -> Void) {
    DispatchQueue.global().async {
        sharedCounter += 1
        completion()
    }
}


func exampleFunction() {
    for _ in 0..<1000 {
        incrementCounter {
           // Here is our non isolated closure, incrementing sharedCounter
            print("Counter is \(sharedCounter)")

        }
    }

    Thread.sleep(forTimeInterval: 2)
    print("Final counter is: \(sharedCounter)")
}


exampleFunction()

```

If you run this code, you'll likely see inconsistent print statements and a final counter value that’s not reliably 1000, demonstrating the effect of race conditions. The problem here isn’t in `incrementCounter` itself but rather, how the closure captures the shared `sharedCounter` variable without any concurrency safety net and is executed concurrently.

Swift's actors, which provide a level of isolation, are intended to solve exactly this class of problem. A crucial distinction is that while the code inside an actor can still run concurrently, only one task can execute within the actor at any given time. When we use an actor to encapsulate our shared state, we gain the safety mechanisms we desperately need when building concurrent systems.

Let’s re-write that example using an actor:

```swift
import Foundation

actor Counter {
    var value = 0

    func increment()  {
        self.value += 1
        print("Counter is \(value)")
    }

    func get() -> Int{
        return value
    }
}


func exampleActorFunction() async {
    let counterActor = Counter()

    for _ in 0..<1000 {
        Task {
           await counterActor.increment()
        }
    }


   try? await Task.sleep(nanoseconds: 2_000_000_000)

   let finalValue = await counterActor.get()
   print("Final counter is \(finalValue)")
}


Task {
    await exampleActorFunction()
}


```

Here, the actor `Counter` manages access to the shared state `value`. Now, each `increment()` call is guaranteed to be executed in isolation, eliminating the race condition, even if we invoke this method from multiple concurrent tasks. The printing happens from inside the isolated context guaranteeing that a proper value will be printed. We also get a consistent final value.

Another very common scenario occurs with UI updates. We are typically restricted to interacting with UI from the main thread (MainActor). So it is tempting to use non actor isolated closures to call update functions of the UI. Let’s examine this.

```swift
import UIKit

class MyViewController: UIViewController {

    @IBOutlet weak var label: UILabel!

    func updateLabelText(newText: String) {
        DispatchQueue.global().async {
            self.label.text = newText // This will crash
        }
    }


    override func viewDidLoad() {
        super.viewDidLoad()

        for i in 0..<1000 {
             updateLabelText(newText: "\(i)")
        }

    }
}

```

This example is very common and demonstrates the problem well. You are executing the closure from a global queue, and UIKit updates must happen from the main thread. So the app will crash due to that problem. Let's fix it now using `MainActor`.

```swift
import UIKit

class MyViewController: UIViewController {

    @IBOutlet weak var label: UILabel!

    @MainActor
    func updateLabelText(newText: String) {
       self.label.text = newText
    }


    override func viewDidLoad() {
        super.viewDidLoad()
        Task {
            for i in 0..<1000 {
                await updateLabelText(newText: "\(i)")
            }
        }

    }
}
```
Here we mark `updateLabelText` with `@MainActor` to indicate that this function must run on the main thread. The compiler now forces you to call this asynchronously and will not compile the code if you forget the await. This allows you to update UIKit from a global queue and the compiler ensures it is done safely from the main thread. Note the call site also had to change to accommodate this.

The key lesson is that not all code needs actor isolation. If a closure doesn’t access shared mutable state, then a non-isolated closure is perfectly safe. It's about identifying when such isolation is required. If you have a closure that is going to update shared variables it must execute from within an isolated context, either an actor, or `@MainActor`.

For a deeper dive, I'd recommend starting with the official swift documentation about actors and concurrency. Also, the book "Effective Concurrency in Swift" by Bryan Irace offers detailed insights into practical usage and pitfalls, and "Concurrency by Tutorials" from raywenderlich.com is also great resource, they both go into depth about these topics. Understanding the nuances of concurrency is crucial, and these resources will give a much more complete understanding of this often-challenging topic. Remember, the issues caused by non-actor isolated closures aren't always immediately obvious, making a strong grasp of swift’s concurrency model is an essential skill.
