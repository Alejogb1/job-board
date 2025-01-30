---
title: "How can I make an @objc function with a parameter type not supported by Objective-C?"
date: "2025-01-30"
id: "how-can-i-make-an-objc-function-with"
---
Objective-C lacks direct support for many modern Swift types, such as tuples, optionals, generics, and Swift-specific enums. Consequently, exposing a Swift function with a parameter using such a type to Objective-C via `@objc` requires an intermediate step of type adaptation. I’ve encountered this constraint numerous times when integrating Swift modules into existing Objective-C codebases, necessitating careful planning to ensure interoperability without compromising type safety or performance. The primary approach involves bridging these unsupported types to compatible Objective-C counterparts, typically using wrapper classes, pre-defined Objective-C types, or through conversion within the function's implementation.

The challenge lies not only in the type mismatch but also in maintaining a clean interface for Objective-C consumers. Objective-C has limited generics, so a generic Swift type, for example, often needs to be represented as a specific concrete class or protocol. The lack of native support for tuples requires creating custom Objective-C classes or structs to encapsulate the data. Similarly, Swift optionals are typically handled by explicitly checking for `nil` in Objective-C, often through a conversion to an equivalent nullable object. These conversions introduce some overhead but are essential for achieving the interoperability goal.

Let’s consider three specific scenarios illustrating how to handle a `CGPoint` tuple, a generic array, and a Swift `enum`.

**Scenario 1: Handling a `(x: Double, y: Double)` tuple parameter.**

Suppose I have a Swift function that accepts a point represented as a tuple of doubles:

```swift
// Swift
func processCoordinates(point: (x: Double, y: Double)) {
    print("Processing point: x=\(point.x), y=\(point.y)")
}
```

This tuple is not directly representable in Objective-C. To expose this function to Objective-C, I need to create an Objective-C-compatible wrapper class or struct, such as the `Point` struct using an associated `create` function in Swift.

```swift
// Swift
@objc
public class PointWrapper: NSObject {
    @objc public let x: Double
    @objc public let y: Double

    @objc public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

extension PointWrapper {
    static func create(_ tuple: (x: Double, y: Double)) -> PointWrapper {
        return PointWrapper(x: tuple.x, y: tuple.y)
    }
}

@objc
public class MyClass: NSObject {
    @objc
    public func processObjectiveCPoint(point: PointWrapper) {
        let swiftPoint = (x: point.x, y: point.y)
        processCoordinates(point: swiftPoint)
    }

    private func processCoordinates(point: (x: Double, y: Double)) {
           print("Processing point from wrapper: x=\(point.x), y=\(point.y)")
    }
}
```

**Commentary:**

Here, I've created an `PointWrapper` Objective-C compatible class with `x` and `y` properties.  My original swift function `processCoordinates` remains unchanged. I then created a new function `processObjectiveCPoint` in the `MyClass` wrapper. This function accepts `PointWrapper` which can be consumed in Objective-C and then converts the wrapper to the swift tuple type for use by the swift internal function `processCoordinates`. This introduces an extra step but maintains type safety and ensures compatibility. The static function on the wrapper is a clean way to convert from the tuple type to the Objective-C consumable type. In Objective-C, I can instantiate and use this wrapper like this:

```objectivec
// Objective-C
PointWrapper *myPoint = [[PointWrapper alloc] initWithX:3.0 y:4.0];
MyClass *myObject = [[MyClass alloc] init];
[myObject processObjectiveCPointWithPoint:myPoint];
```

**Scenario 2: Handling a generic array parameter.**

Now, consider a Swift function that accepts a generic array:

```swift
// Swift
func processArray<T>(items: [T]) {
    print("Processing array with \(items.count) elements")
}
```

Objective-C does not support generic types. A common solution is to create a function specific to supported array types, like strings or numbers. For the sake of demonstration, consider an array of strings. First, an internal, type-specific Swift function can consume the Swift type internally. The function exposed to Objective-C can consume an `NSArray` of strings and then convert to an array of `String`.

```swift
// Swift
@objc
public class MyClass2: NSObject {
    @objc
    public func processObjectiveCStringArray(items: NSArray) {
        let swiftStringArray = items.compactMap{ $0 as? String }
        processArray(items: swiftStringArray)
    }
   private func processArray(items: [String]) {
           print("Processing string array with \(items.count) elements")
   }

}
```

**Commentary:**

The `processObjectiveCStringArray` function accepts an `NSArray` from Objective-C. I then leverage `compactMap` to extract the String values and cast them to the `[String]` type used by the internal swift function. This allows the Swift function to receive the expected type. It also gracefully handles incorrect `NSArray` types. In Objective-C, an array can be passed as follows:

```objectivec
// Objective-C
NSArray *stringArray = @[@"hello", @"world", @"swift"];
MyClass2 *myObject = [[MyClass2 alloc] init];
[myObject processObjectiveCStringArrayWithItems:stringArray];
```

**Scenario 3: Handling a Swift enum parameter.**

Finally, consider the following Swift enum and a function that accepts it as parameter.

```swift
// Swift
enum Status: Int {
    case pending
    case processing
    case completed
}

func handleStatus(status: Status) {
    print("Handling status: \(status)")
}
```

Objective-C does not have built-in support for Swift enums. Here, I use the raw `Int` value of the `enum` to bridge the two languages. I will also create helper functions for conversion.

```swift
// Swift
@objc
public enum ObjectiveCStatus: Int {
    case pending
    case processing
    case completed
}
extension Status {
    static func create(_ value: ObjectiveCStatus) -> Status {
        switch value {
          case .pending: return .pending
          case .processing: return .processing
          case .completed: return .completed
        }
    }
}

@objc
public class MyClass3: NSObject {
    @objc
    public func handleObjectiveCStatus(status: ObjectiveCStatus) {
        let swiftStatus = Status.create(status)
        handleStatus(status: swiftStatus)
    }
    private func handleStatus(status: Status) {
        print("Handling status: \(status)")
    }

}

```

**Commentary:**

I have created an Objective-C compatible enum `ObjectiveCStatus` with corresponding cases.  The `handleObjectiveCStatus` function converts the `ObjectiveCStatus` to a Swift `Status` and passes this to the internal function.  This strategy relies on a raw integer to represent the underlying enum case, which is accessible from Objective-C. The `create` function converts from the Objective-C enum case to the Swift enum case. In Objective-C, usage is as follows:

```objectivec
// Objective-C
ObjectiveCStatus myStatus = ObjectiveCStatusProcessing;
MyClass3 *myObject = [[MyClass3 alloc] init];
[myObject handleObjectiveCStatusWithStatus:myStatus];
```

These examples illustrate common strategies for handling Swift types not natively supported by Objective-C. The primary idea is to create an Objective-C friendly wrapper, or expose a conversion function that performs the necessary conversion.

**Resource Recommendations:**

For further understanding, I recommend consulting the official Swift documentation on interoperability with Objective-C. Additionally, review the documentation for bridging headers and Objective-C runtime environments. Examine existing open-source Swift projects that interface with Objective-C for practical examples of how developers have handled similar issues. Finally, consider a dedicated book on advanced Swift development for deeper insights into cross-language interoperability. These resources will provide a more thorough and practical understanding of this complex topic.
