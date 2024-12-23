---
title: "How can I use `.contains` with an array of Struct in SwiftUI?"
date: "2024-12-23"
id: "how-can-i-use-contains-with-an-array-of-struct-in-swiftui"
---

Okay, let’s tackle this one. I’ve definitely been in this particular corner of SwiftUI before, and it’s a common tripping point, especially when you're moving beyond simple data types. The `.contains()` method, while seemingly straightforward, requires a bit more nuance when you’re dealing with arrays of custom structs. It's not immediately obvious how to tell it what "containment" actually means in that context.

The core issue stems from the fact that SwiftUI’s `.contains()` method, inherited from the `Sequence` protocol, performs an equality check. By default, it compares the memory addresses of two struct instances, not their underlying property values. This means that even if two structs have identical field values, they are considered different because they are distinct objects in memory. You're essentially asking, "Is this *exact* block of memory present in the array?" not "Is there a struct with these *values*?"

To get the behavior you expect, you'll need to conform your struct to the `Equatable` protocol. This protocol requires you to implement the `==` operator, which is where you define what constitutes "equality" for your struct. Once implemented, `contains(_:)` will then use that comparison logic. Let me walk through a few scenarios with code examples, based on some of my past experiences battling this issue.

**Scenario 1: Basic Equality Based on a Single Property**

Imagine you have a struct representing a user, and you want to check if a specific user with a particular ID already exists in your array. This is probably the most common use case.

```swift
struct User: Identifiable, Equatable {
    let id: UUID
    let name: String

    // Equatable conformance. We're using just the id for comparison here
    static func == (lhs: User, rhs: User) -> Bool {
        return lhs.id == rhs.id
    }
}


func checkIfUserExists() {
    let users = [
      User(id: UUID(), name: "Alice"),
      User(id: UUID(), name: "Bob"),
    ]
    let targetUser = User(id: users[0].id, name: "Something Else") // User with matching ID but different name
    
    let exists = users.contains(targetUser) //  will now evaluate true, not false
     
    print("User Exists: \(exists)") // Prints: User Exists: true
}

checkIfUserExists()
```

In this example, we've conformed the `User` struct to `Equatable` and defined that two `User` instances are equal if their `id` properties are equal. Even though `targetUser` has a different `name`, it will still be considered "contained" because the `id` matches a member of the `users` array. This illustrates that you control which fields matter for your comparison.

**Scenario 2: Equality Based on Multiple Properties**

In some situations, a match might require more than just one property being identical. Let's say you have a product, and equality is based on both the product's ID *and* its color.

```swift
struct Product: Identifiable, Equatable {
    let id: Int
    let name: String
    let color: String

    static func == (lhs: Product, rhs: Product) -> Bool {
         return lhs.id == rhs.id && lhs.color == rhs.color
     }
}


func checkProduct() {
  let products = [
     Product(id: 1, name: "T-Shirt", color: "Red"),
     Product(id: 2, name: "Jeans", color: "Blue"),
  ]

  let existingProduct = Product(id: 1, name: "T-Shirt", color: "Red")
  let productWithDifferentColor = Product(id: 1, name: "T-Shirt", color: "Green")
  let productWithDifferentId = Product(id: 3, name: "T-Shirt", color: "Red")

  let firstCheck = products.contains(existingProduct)
  let secondCheck = products.contains(productWithDifferentColor)
  let thirdCheck = products.contains(productWithDifferentId)

    print("First Check \(firstCheck)")  // Prints: First Check true
    print("Second Check \(secondCheck)") // Prints: Second Check false
    print("Third Check \(thirdCheck)") // Prints: Third Check false
}

checkProduct()
```

Here, the `==` operator checks both the `id` and the `color` fields. The `productWithDifferentColor` struct has a matching ID but a different color, resulting in `contains` returning false. Likewise, the product with a different ID fails the check, as is expected.

**Scenario 3: Using `contains(where:)` for Complex Criteria**

Sometimes, your equality check may not fit neatly into the `Equatable`'s `==` method. Perhaps the "containment" criterion is too dynamic or context-specific. This is where `contains(where:)` is beneficial. Instead of directly testing for struct equality, you provide a closure that defines the matching logic. Let's say you want to see if the array contains *any* product that matches a given ID, regardless of color and name.

```swift
struct OrderItem {
   let productId: Int
   let quantity: Int
   let name: String
   let color: String
}


func checkOrderItems() {
    let orderItems = [
       OrderItem(productId: 1, quantity: 2, name: "T-Shirt", color: "Red"),
       OrderItem(productId: 2, quantity: 1, name: "Jeans", color: "Blue")
    ]

    let check = orderItems.contains(where: { item in
            item.productId == 1
        })
        
    let checkFail = orderItems.contains(where: {item in item.productId == 3})
        
    print("Check \(check)")  // Prints: Check true
    print("CheckFail \(checkFail)") // Prints: CheckFail false

}

checkOrderItems()

```

In this final example, we use `contains(where:)` to check if the `orderItems` array contains at least one item with a particular `productId`.  We're not using `Equatable` and, therefore, avoiding the need to define a static function. This approach is useful when you have a very specific check that is unlikely to be reusable.

**Key Takeaways and Recommendations**

*   **`Equatable` is Essential for Simple Equality Checks:** When you need to determine if a struct with particular values exists in the array based on its property values and this check is generally applicable.
*   **`contains(where:)` for Complex Matching Logic:** When the rules for matching are specific to a context or can't be easily expressed as simple equality.

When delving further into this topic, I’d highly recommend diving into the following sources:

*   **"The Swift Programming Language" by Apple:** The official documentation gives a precise overview of `Equatable`, `Sequence`, and collection handling in swift, essential for solidifying your understanding.
*   **"Effective Swift" by Matt Gallagher:** This book dives into best practices, covering topics like how and when to implement protocols effectively in Swift, including `Equatable`, leading to more efficient and readable code.
*   **"Advanced Swift" by Chris Eidhof, Ole Begemann, and Airspeed Velocity:** While more in-depth, this is great for exploring how swift handles protocols under the hood, thus enabling an understanding of why the `.contains` function operates as it does.

Remember, your struct's specific properties dictate how the equality check operates. Always think carefully about which fields define identity in your data model. Using these techniques, you will find you can more effectively use `.contains()` with your arrays of custom structs. You now have the tools to avoid those early frustration points with swift collections.
