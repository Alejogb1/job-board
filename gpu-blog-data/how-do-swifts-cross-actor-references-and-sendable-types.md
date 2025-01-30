---
title: "How do Swift's cross-actor references and Sendable types interact?"
date: "2025-01-30"
id: "how-do-swifts-cross-actor-references-and-sendable-types"
---
In Swift, concurrent programming relies on isolating data access to prevent race conditions. Actors provide this isolation by encapsulating state and serializing access through asynchronous methods. The interaction between cross-actor references and the `Sendable` protocol is paramount for maintaining data integrity when working across these isolated contexts. I've spent considerable time debugging concurrency issues related to incorrect `Sendable` conformance, and I can say firsthand that a deep understanding of their interplay is essential for robust Swift applications.

The core concept revolves around the `Sendable` protocol and its implication for data that crosses actor boundaries. A type that conforms to `Sendable` guarantees that it can be safely transmitted across these boundaries. This transmission involves a process conceptually similar to copying the value. However, it's more complex than a simple bitwise copy. The compiler enforces this guarantee, ensuring that if a type is used in a context where a `Sendable` instance is expected (like when passing data to an actor method or closure that runs on a different actor), that type must conform to the `Sendable` protocol. Violations of this constraint lead to compiler errors, alerting developers to potential concurrency issues.

The most common scenario involves data that implicitly conforms to `Sendable`. For example, primitive types like `Int`, `String`, `Bool`, and enumerations without associated values are inherently `Sendable` because they are value types with no internal mutable state that could be affected by concurrent access. Structures and enumerations containing only `Sendable` members also automatically conform to `Sendable`. When passing such values to actor methods, they are effectively copied, ensuring that the actor receiving the value operates on an isolated replica, rather than a shared reference. This replication prevents accidental concurrent modifications from affecting other actors and avoids data corruption.

However, complications arise when dealing with reference types, such as classes. By default, classes are *not* `Sendable`. Passing a class instance to an actor method would expose a single, shared instance to multiple execution contexts, violating the principle of actor isolation. The compiler will flag this with a warning, or if concurrency safety checking is enabled, an outright error. The solution is either to modify the class to conform to `Sendable`, or to use a value type wrapper to transmit the data safely.

The `Sendable` conformance of a class often requires carefully managing its internal state to ensure that any data that could be accessed from concurrent contexts is itself thread-safe. This might involve using synchronization mechanisms like locks or introducing immutable properties for data meant to be accessed across actor boundaries. Alternatively, a simpler solution is often to avoid passing classes across actor boundaries directly. Instead, a value type, such as a struct or an enumeration with associated values, can encapsulate the needed data, and that data will be copied when passed across actor boundaries. The structure can even store a reference to the class but should do it in a way that the class reference is treated as effectively immutable from the actor's point of view. For instance, the class can be designed so that its data cannot be mutated once it has been initialized.

Let's illustrate this with some examples:

**Example 1: Passing a Simple Struct to an Actor**

```swift
actor Counter {
    private(set) var count = 0

    func increment(by amount: Int) {
        count += amount
    }
}

struct IncrementRequest: Sendable {
    let amount: Int
}


let counter = Counter()

func processRequest(request: IncrementRequest) async {
  await counter.increment(by: request.amount)
  print("Counter is: \(await counter.count)")
}


Task {
    await processRequest(request: IncrementRequest(amount: 5))
    await processRequest(request: IncrementRequest(amount: 10))
}
```

In this example, `IncrementRequest` is a simple struct and implicitly conforms to `Sendable`. When `processRequest` is called, the `IncrementRequest` value is copied and passed to the actor’s method. The actor receives a completely independent value, avoiding any concurrency issues. Each `processRequest` call uses different copies of the request data. The `count` within the `Counter` actor is protected from concurrent access by the actor's internal isolation mechanism.

**Example 2: Incorrectly Passing a Non-Sendable Class to an Actor**

```swift
class MutableState {
    var value: Int = 0
}

actor BadActor {
    func update(state: MutableState) {
        state.value += 1
    }
}

let badActor = BadActor()
let sharedState = MutableState()


Task {
    await badActor.update(state: sharedState)
    print("State value is \(sharedState.value)") // Data race is possible!
}
Task {
    await badActor.update(state: sharedState)
    print("State value is \(sharedState.value)") // Data race is possible!
}

```

This code will generate a compiler error (or at least a warning depending on compiler settings). The `MutableState` class is not `Sendable`, and therefore, it’s unsafe to pass an instance to an actor. Even though the actor methods are serialized, the `MutableState` instance is shared across actor instances. This code has the potential for introducing race conditions when both tasks attempt to modify the shared `MutableState.value` concurrently. While the actor ensures serialized access to the *actor's state* it does *not* offer the same guarantee to data it is given.  The compiler forces you to address this issue before you get to runtime.

**Example 3: Using an Immutable Struct Containing Class Data**

```swift
class ImmutableDataHolder {
  private (set) var value: Int

  init(initialValue: Int) {
    self.value = initialValue
  }
}


struct SafeData: Sendable {
    let immutableData: ImmutableDataHolder
}


actor SafeActor {
    func process(data: SafeData) {
        print("Received: \(data.immutableData.value)")

    }
}


let safeActor = SafeActor()
let dataHolder = ImmutableDataHolder(initialValue: 42)
let safeData = SafeData(immutableData: dataHolder)


Task {
    await safeActor.process(data: safeData)
    print("Data was passed!")
}

Task {
  await safeActor.process(data: safeData)
  print("Data was passed!")
}

```
This example demonstrates the correct approach when working with class instances across actors. We encapsulated the class instance in a `SafeData` struct. While we are technically passing a reference to the `ImmutableDataHolder` it is immutable within its scope. The `ImmutableDataHolder` class itself allows only reads and is immutable after the initial construction. The `SafeData` struct and thus the class reference, is `Sendable` because the class reference is treated as effectively read-only in the context of being sent to an actor method. Each call to `safeActor.process` receives a copy of the `SafeData` struct, which contains a shared, but immutable `ImmutableDataHolder` reference.

Understanding the subtle differences between passing value and reference types across actor boundaries is vital. Passing value types (or value type representations of reference types) usually results in a copy, whereas passing a reference type directly to an actor method can lead to concurrency issues if the class is not explicitly declared `Sendable`. The `Sendable` protocol is the mechanism that Swift uses to enforce safe data transfer between actors.

For additional learning resources, I highly recommend exploring Apple's official documentation on Swift concurrency, especially articles related to actors, `Sendable`, and data races. Consider exploring some of the more advanced topics like custom `Sendable` conformance when working with complex class types. Books focusing on concurrent programming in Swift also offer a more detailed and structural view. Furthermore, various online tutorials present practical examples and case studies that can solidify your understanding of these core concepts. Finally, spending time exploring Swift code in existing open source projects can offer practical insights into how these concepts are used in production code.
