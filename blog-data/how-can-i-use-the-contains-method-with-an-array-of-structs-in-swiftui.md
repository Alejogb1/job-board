---
title: "How can I use the .contains method with an array of structs in SwiftUI?"
date: "2024-12-23"
id: "how-can-i-use-the-contains-method-with-an-array-of-structs-in-swiftui"
---

Alright, let’s tackle this. I've encountered this specific challenge quite a few times in my work, particularly when dealing with complex data models in SwiftUI projects. It’s a common stumbling block for many, so you're not alone. The issue essentially boils down to the fact that the `.contains()` method, when used directly with an array of structs, relies on *equality*. Now, structs by default don't inherently know how to compare themselves for equality, unless they adhere to the `Equatable` protocol. This isn't a SwiftUI-specific quirk; it's how Swift works in general. Let me break down how to handle this situation and avoid potential pitfalls.

First, let’s establish the foundational issue: a straightforward `.contains()` check on an array of structs won’t function as you might expect. Assume we have a struct like this, and for illustrative purposes, let's say this was in a past project where I was implementing a task management system:

```swift
struct Task {
    let id: UUID
    let title: String
    let isCompleted: Bool
}
```

Now, if you create an array of `Task` instances and try to use `.contains()` on it, supplying a *different* `Task` instance, even one with seemingly matching property values, it won't work. The default behaviour is to compare object *identities* not their content. The code will effectively always return `false`. This tripped me up a fair bit until I understood what was really happening beneath the surface. This happened during a sprint where we had a complex caching mechanism that relied heavily on matching already existing objects with incoming ones.

To rectify this, we must make our `Task` struct conform to the `Equatable` protocol. This involves defining how two `Task` instances should be considered equal. Typically, you’d compare the fields that *uniquely* identify your struct. In our `Task` case, the `id` is a good candidate:

```swift
struct Task: Equatable {
    let id: UUID
    let title: String
    let isCompleted: Bool

    static func == (lhs: Task, rhs: Task) -> Bool {
        return lhs.id == rhs.id
    }
}
```

Here, we explicitly implement the `==` operator, instructing Swift to consider two `Task` instances equal if their `id` properties are identical. Now `.contains()` will perform as intended when looking for a `Task` with the correct `id`. This method, while straightforward and frequently used, does come with a performance implication as this equality function has to be run for every item in the array until a match is found.

Let’s illustrate with some practical code snippets:

**Snippet 1: The Incorrect Approach (Without Equatable)**

```swift
let task1 = Task(id: UUID(), title: "Grocery Shopping", isCompleted: false)
let task2 = Task(id: UUID(), title: "Grocery Shopping", isCompleted: false) // Different instance

let taskArray = [task1]

let containsTask2 = taskArray.contains(task2) // This will return false, even though properties are the same
print(containsTask2) // Output: false
```

**Snippet 2: The Correct Approach (With Equatable)**

```swift
struct Task: Equatable { // Added Equatable conformance
    let id: UUID
    let title: String
    let isCompleted: Bool

    static func == (lhs: Task, rhs: Task) -> Bool {
        return lhs.id == rhs.id
    }
}


let task3 = Task(id: UUID(), title: "Grocery Shopping", isCompleted: false)
let task4 = Task(id: task3.id, title: "Different Title", isCompleted: true) // Same ID

let taskArray2 = [task3]

let containsTask4 = taskArray2.contains(task4)
print(containsTask4) // Output: true
```

As you can see, by conforming to `Equatable` and defining the `==` operator based on the `id` property, `.contains()` correctly determines if the array contains a task with a matching id. We are now using the identifier of the task, as opposed to the pointer address of the struct in memory. In more complex scenarios, you might compare multiple properties within the equality check based on the business logic.

There is, however, a subtle variation on the `contains` function that might be more useful. If you don’t wish to enforce `Equatable`, or perhaps the definition of equality doesn’t fit with the logic needed for checking membership, you can use the version of `contains` that takes a closure. The closure-based version is more flexible but a bit verbose.

**Snippet 3: Contains with a Closure**

```swift
struct TaskWithoutEquatable {
    let id: UUID
    let title: String
    let isCompleted: Bool
}


let task5 = TaskWithoutEquatable(id: UUID(), title: "Pay Bills", isCompleted: false)
let task6 = TaskWithoutEquatable(id: task5.id, title: "Another title", isCompleted: true)


let taskArray3 = [task5]

let containsTask6 = taskArray3.contains(where: { $0.id == task6.id }) //Using closure version

print(containsTask6) // Output: true
```

Here, we use the `contains(where:)` variant, passing in a closure that defines the comparison logic explicitly. This allows us to check if any element in the array has an `id` that matches the `id` of `task6`, even though `TaskWithoutEquatable` doesn't conform to `Equatable`. This is particularly useful for more fine-grained control over the comparison and for more dynamic conditions. I've used this in scenarios where the definition of 'contains' changes based on different view or user actions.

For deeper understanding, I’d recommend diving into "The Swift Programming Language" book, specifically the sections on protocols, particularly `Equatable`. Additionally, the “Data Structures and Algorithms in Swift” by Ray Wenderlich provides excellent coverage of fundamental data structures and associated operations, allowing you to understand the performance implications of equality checks better. For more detailed knowledge of Swift internals, explore "Advanced Swift" by Chris Eidhof and Ole Begemann. These resources should provide a solid foundation for understanding and effectively using these methods in your projects.

In summary, the `.contains()` method needs clear instructions on how to compare struct instances. You can achieve this through `Equatable` conformance, defining the `==` operator, or using the `contains(where:)` variation with a custom closure. Choosing the method which works best is dependent on the specificity of your needs and how you intend to define equality within the context of the program. Remember that `Equatable` enforces a specific equality definition, while the closure-based approach offers the most flexibility. By mastering these techniques, you'll be equipped to effectively manage collections of structs within your SwiftUI applications, or any Swift project for that matter.
