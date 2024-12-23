---
title: "How do I use `.contains` with an array of structs in SwiftUI?"
date: "2024-12-16"
id: "how-do-i-use-contains-with-an-array-of-structs-in-swiftui"
---

,  I remember distinctly a project a while back, a data visualization app, where we needed to filter a complex dataset represented by an array of custom structs. Performance was paramount, so we had to be meticulous with how we implemented these filters. The naive approach of looping was immediately discarded as unsustainable. The question of using `.contains` efficiently with arrays of structs in SwiftUI is far more nuanced than it first appears. It isn't a straightforward comparison like you might have with simple value types.

The core issue here is that `.contains` by default relies on the `Equatable` protocol, and structs, while they can conform to it, don't automatically do so based on structural equality (i.e., field-by-field comparison). If you simply try to use `.contains` with a struct array without conforming to `Equatable`, you'll get a compile error telling you that your struct doesn't satisfy the requirement. The Swift standard library cannot know, implicitly, what constitutes equality between two instances of your custom type. So, we need to explicitly define it.

Let's dive into an example. Say we have a struct representing a point in a 2D space:

```swift
struct Point: Identifiable {
    let id = UUID()
    let x: Int
    let y: Int
}
```

Now, if we were to create an array of `Point` instances and try to see if a particular point exists in the array using `.contains`, like so:

```swift
let points = [
    Point(x: 1, y: 2),
    Point(x: 3, y: 4),
    Point(x: 5, y: 6)
]
let targetPoint = Point(x: 3, y: 4)
let containsPoint = points.contains(targetPoint) // Compile error!
```

This would fail because `Point` isn't `Equatable`. To fix this, we need to extend our `Point` struct to conform to `Equatable`. Here's the modified struct:

```swift
struct Point: Identifiable, Equatable {
    let id = UUID()
    let x: Int
    let y: Int

    static func == (lhs: Point, rhs: Point) -> Bool {
        return lhs.x == rhs.x && lhs.y == rhs.y
    }
}
```

In this version, we've explicitly defined how two `Point` instances should be considered equal: by comparing their `x` and `y` coordinates. Critically, `id`, though present as an identifiable element, is not considered when comparing points for equality, meaning that two points with differing ids can be considered equal as long as their x and y values match. Now the following code, that previously failed, will run successfully:

```swift
let points = [
    Point(x: 1, y: 2),
    Point(x: 3, y: 4),
    Point(x: 5, y: 6)
]
let targetPoint = Point(x: 3, y: 4)
let containsPoint = points.contains(targetPoint) // Now this works, containsPoint is true
```

This highlights the first key concept: You must implement `Equatable` for custom types when using `.contains` to check for presence based on the type's values rather than its address in memory.

Now, let's consider a scenario where we need more flexibility. Sometimes, you might not want to define an overall equality, or it might not be appropriate for how you want to filter at different points in your code. Let's use a different struct, say `Employee`:

```swift
struct Employee: Identifiable {
    let id = UUID()
    let name: String
    let department: String
    let salary: Double
}
```

In some cases, you might want to check if an employee with a specific name exists, regardless of their department or salary. Using the previous `Equatable` approach could work but requires a specific definition for equality which may not be flexible enough for multiple filter needs. For this use case we can use `.contains(where:)` method:

```swift
let employees = [
    Employee(name: "Alice", department: "Engineering", salary: 100000),
    Employee(name: "Bob", department: "Sales", salary: 80000),
    Employee(name: "Charlie", department: "Engineering", salary: 120000)
]

let targetName = "Bob"
let containsEmployeeName = employees.contains(where: { $0.name == targetName }) // containsEmployeeName is true
```

Here, we're using a closure to define the condition for what constitutes a match. This allows for significantly more dynamic checks without needing to implement full `Equatable` logic across the whole struct. This is the second main take away: The `contains(where:)` variant offers a lot of flexibility if you need to evaluate containment using criteria that are more nuanced or that do not apply across all use cases.

Finally, we might face scenarios involving complex objects with nested structures. In this case, it might be necessary to dive into comparing those nested values. Suppose we have a struct like `Project`:

```swift
struct Address {
    let street: String
    let city: String
}

struct Project: Identifiable, Equatable {
    let id = UUID()
    let name: String
    let address: Address
    
    static func == (lhs: Project, rhs: Project) -> Bool {
        return lhs.name == rhs.name && lhs.address.street == rhs.address.street && lhs.address.city == rhs.address.city
    }
}
```

Here, equality for `Project` is tied not only to the name of the project, but also to its location, which is represented by a nested struct called `Address`. The same principle applies when using `.contains` as when dealing with simple types. This struct implements `Equatable` by recursively evaluating the inner types used. The important lesson here is, when dealing with complex types, ensure that the equality operator, or alternatively your custom logic for `.contains(where:)` considers all relevant nested parameters.

When you’re working with performance-sensitive code, also keep in mind the computational complexity of these operations. While `.contains` itself is generally quite efficient, particularly if the underlying collection is indexed, the complexity inside of your custom `==` function, or your closure, contributes to the overall computational load. Avoid costly operations within the equality check or closure if possible, such as loading data from storage or doing complex calculations repeatedly.

For a more theoretical background and best practices, I'd recommend looking into "Data Structures and Algorithms in Swift" by Marcin Krzyżanowski. Also, reading through the "Swift Standard Library" documentation, especially the sections on `Sequence`, `Collection`, and `Equatable` would be invaluable. These resources provide a solid foundation in understanding these concepts at a deeper level, along with best practices for working with Swift's standard collection types.

In summary, using `.contains` with arrays of structs requires a thoughtful approach. Implement `Equatable` to establish how instances of your struct are considered equal, and use `contains(where:)` for flexible filtering based on specific criteria. Remember that when working with complex or nested structures that equality comparisons need to consider all relevant fields and should remain efficient for performant results. This combination of knowledge and practical application has allowed me to work through many similar situations effectively over the years.
