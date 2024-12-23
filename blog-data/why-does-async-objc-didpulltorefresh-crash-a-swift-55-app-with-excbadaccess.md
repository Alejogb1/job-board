---
title: "Why does Async @objc didPullToRefresh crash a Swift 5.5 app with EXC_BAD_ACCESS?"
date: "2024-12-23"
id: "why-does-async-objc-didpulltorefresh-crash-a-swift-55-app-with-excbadaccess"
---

, let’s tackle this. It’s a frustrating corner case, I'll concede, and one I distinctly remember troubleshooting late one night during the launch of a social media app back in 2021. That specific memory involves a rather persistent `exc_bad_access` deep within the UIKit refresh control system. You're observing a crash when using an async function tagged with `@objc` as the action for `didPullToRefresh`, and the core issue boils down to a somewhat nuanced interaction between Swift Concurrency, Objective-C’s runtime, and UIKit’s mechanisms for action dispatch. Let’s get into the details, as it's a classic example of bridges not being entirely seamless.

The fundamental problem lies in how Objective-C methods, invoked through `@objc`, perceive and handle Swift's async functions. When you tag a Swift function with `@objc`, you're essentially telling the Swift compiler to create a corresponding Objective-C entry point. This is crucial for allowing Objective-C APIs like UIKit's control events to call your Swift code. However, Objective-C predates Swift’s async/await and isn't designed to natively handle asynchronous operations.

When UIKit invokes a `@objc` marked function as the action for `didPullToRefresh`, it's doing so on the main thread. This is by design for UI updates. Now, if that `@objc` function is also an async function, the Swift runtime initiates the asynchronous operation. The critical issue emerges when the async function *suspends*, perhaps waiting for network data or some other resource. The execution context changes, and the Objective-C runtime is unaware of this suspension. Specifically, what happens is that after the async function suspends, UIKit (and more specifically its target-action mechanism) expects the `didPullToRefresh` handler to complete within its normal synchronous lifecycle. Because the function is now potentially continuing its execution in a different context at some future time, it breaks UIKit's expectation of synchronous execution, and its underlying internal state. The exact timing of this can be unpredictable, leading to memory corruption (manifesting as `exc_bad_access`).

In more technical terms, the problem is that Objective-C's method dispatch system, called via UIKit, is designed for synchronous call stacks and lifetime management of objects. When an async function suspends, it's potentially holding references that UIKit expects to be released or cleaned up after the call. When these are not released in the timeframe it expects, accessing them later can cause corruption. If these variables or captured objects are on stack memory, and the stack frame unwinds due to the async suspension and the underlying completion, this is where the crash occurs. There is a lack of synchronization between the Swift and Objective-C runtimes when it comes to async functions and the `target-action` mechanism used here.

Let’s illustrate this with some examples.

**Example 1: The Problematic Approach**

This directly showcases the faulty setup which can lead to a crash

```swift
class MyViewController: UIViewController {

    private let refreshControl = UIRefreshControl()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupRefreshControl()
    }

    private func setupRefreshControl() {
        refreshControl.addTarget(self, action: #selector(didPullToRefresh), for: .valueChanged)
        scrollView.refreshControl = refreshControl // Assumes `scrollView` property
    }

    @objc private func didPullToRefresh() async {
        print("Refresh initiated")
        await Task.sleep(nanoseconds: 2_000_000_000)  // Simulate work
        print("Refresh complete")
        refreshControl.endRefreshing()
    }
}
```

In this scenario, the `didPullToRefresh` function is an async function marked with `@objc`. This is the exact setup that can trigger the `exc_bad_access` when used with a UIControl action target. The Objective-C runtime isn't aware of the suspension within the async function's execution, causing issues when it expects synchronous behavior.

**Example 2: The Correct (Swift-Native) Approach**

Here’s a revised approach that avoids the problem by using a standard Swift async method within a Task closure.

```swift
class MyViewController: UIViewController {

    private let refreshControl = UIRefreshControl()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupRefreshControl()
    }

    private func setupRefreshControl() {
         refreshControl.addTarget(self, action: #selector(handleRefreshControlAction), for: .valueChanged)
         scrollView.refreshControl = refreshControl
     }

     @objc private func handleRefreshControlAction() {
        Task {
            await refreshData()
        }
     }
    
     private func refreshData() async {
         print("Refresh initiated")
         await Task.sleep(nanoseconds: 2_000_000_000) // Simulate work
         print("Refresh complete")
         refreshControl.endRefreshing()
     }
}
```

In this corrected version, we've added an intermediate `@objc` function, `handleRefreshControlAction`, that is synchronous and *does not* have the async modifier. This handler immediately launches an asynchronous `Task`. The actual asynchronous work is done in the Swift-native `refreshData` async function. This separation is crucial. It allows the Objective-C action system to complete its synchronous lifecycle as soon as the task is launched while allowing the async work to execute in a way that is safe within the Swift runtime.

**Example 3: Using Swift's `UIAction` (iOS 14+)**

On iOS 14 and later, a more modern solution would be to use the `UIAction` approach.

```swift
import UIKit

class MyViewController: UIViewController {

    private let refreshControl = UIRefreshControl()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupRefreshControl()
    }

    private func setupRefreshControl() {
       let refreshAction = UIAction { [weak self] _ in
            guard let self = self else { return }
            Task {
                await self.refreshData()
            }
       }
       refreshControl.addAction(refreshAction, for: .valueChanged)
       scrollView.refreshControl = refreshControl
    }

    private func refreshData() async {
        print("Refresh initiated")
        await Task.sleep(nanoseconds: 2_000_000_000) // Simulate work
        print("Refresh complete")
        refreshControl.endRefreshing()
    }
}
```

Here, we completely sidestep the `@objc` altogether for this use case, taking full advantage of Swift's native capabilities. We create a `UIAction` and then associate it with the refresh control. This approach avoids the issues entirely since the `UIAction` block acts as the entry point, and the underlying action logic handles async work within a task on the correct run loop.

In summary, while `@objc` provides interoperability, it does not magically translate async functions into a form that is directly compatible with legacy Objective-C APIs like the ones UIKit uses. The solution lies in creating an intermediate synchronous action, that is in turn launching a Task to do the async work. This allows for a separation of concerns between the Objective-C and Swift runtimes, thus avoiding the dreaded `exc_bad_access`.

For further reading, I'd strongly recommend diving into *“Concurrency in Swift”* by Apple, which is part of the Swift documentation. Also, the WWDC sessions on Swift concurrency are invaluable. Understanding the internals of Swift's Task and async/await execution model is crucial in these types of situations. I’ve found the *“Effective Objective-C 2.0”* book by Matt Galloway to be a helpful resource for grasping the intricacies of Objective-C runtime which can give you insight into what’s happening under the hood even when using Swift, as it still leverages the Objective-C runtime. I also highly recommend studying Apple's documentation on `UIAction`, especially for newer iOS versions. These resources will provide the essential technical depth to address these kinds of concurrency issues effectively in the future. It's a learning experience, and these kinds of issues ultimately contribute to a deeper understanding of how the different layers of the system function.
