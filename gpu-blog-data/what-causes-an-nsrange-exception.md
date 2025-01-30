---
title: "What causes an NSRange exception?"
date: "2025-01-30"
id: "what-causes-an-nsrange-exception"
---
NSRange exceptions, commonly encountered when working with strings, attributed strings, and data in Objective-C and Swift, typically arise from the mismatch between a requested range and the actual bounds of the target data structure. As a developer who has spent years debugging iOS applications, I've found these exceptions often stem from subtle errors in index calculations or manipulation of string lengths. A core aspect to grasp is that `NSRange` describes a location (starting index) and a length, both unsigned integers. This inherent design, while efficient, presents potential for errors if not carefully handled.

An `NSRange` exception is essentially the runtime's way of signaling an out-of-bounds access attempt. The operating system, through the Foundation framework, detects that an operation is attempting to read or write data at a memory location not allocated or intended for the data being operated upon. This can manifest as a crash, often with an error message clearly stating an `NSRange` exception, though the specific error message itself may vary slightly depending on the context and which API is triggering the problem. When it comes to strings, these exceptions frequently occur when an `NSRange`'s location or location + length goes beyond the length of the string itself. Similar principles apply to data, arrays, and other ordered collections.

The key issue often isn't a deliberate attempt to access out-of-bounds data, but rather, unintended consequences of range calculations, modifications to data, or unexpected input. For example, a seemingly innocent substring extraction operation could cause an exception if the computed range is based on an incorrect assumption about string content or length after recent string alterations. These errors are frequently caused by fencepost errors, where an index is off-by-one or where an off-by-one error in length leads to accessing data past the last valid element in a string or other container type. In my own development, I often use unit tests that deliberately introduce extreme cases (zero length, large lengths, edge cases) to help detect these kinds of errors during the testing cycle.

The code examples below demonstrate common causes of this exception, alongside practical mitigation strategies:

**Example 1: Substring Extraction with Incorrect Range**

This Swift code attempts to extract a substring using a hardcoded range without verifying the string's length.

```swift
let myString = "Hello, world!"
let badRange = NSRange(location: 7, length: 10)
// Exception will occur, as the length of "Hello, world!" is only 13, making location 7 + length 10 exceed the string bounds.
//let subString = String(myString[Range(badRange, in: myString)!])
// The following, correct code, would avoid the exception:
if badRange.location < myString.count && badRange.location + badRange.length <= myString.count {
    let subString = String(myString[Range(badRange, in: myString)!])
    print(subString) // Prints world!
} else {
    print("Invalid range.")
}
```

**Commentary:**

This first example highlights the risk of using pre-defined ranges without first performing boundary checks. In this situation, the hardcoded `badRange` would cause a runtime error because the length extends past the end of the string. The commented-out code demonstrates the point of failure, and below it is the fixed version using Swift's range conversion and checking that the location is within the string's bounds, and that the end point of the range is also within the string’s bounds. The conditional check is critical; before accessing string contents via a range, it should be verified. This pattern of validation is standard and should always be incorporated in any context where range parameters are not guaranteed to be correct.

**Example 2:  String Manipulation with Calculated Ranges**

Here's another scenario where an `NSRange` exception can occur during string manipulation.

```swift
var mutableString = NSMutableString(string: "Test")
let originalLength = mutableString.length
mutableString.append("ing")
// If I were to use the originalLength here in subsequent steps, an exception will occur.
let range = NSRange(location: 0, length: originalLength)
// Now, this will lead to an NSRange Exception due to the append, causing the length to change.
// mutableString.deleteCharacters(in: range)

// Here is the corrected code.
let currentLength = mutableString.length
let currentRange = NSRange(location: 0, length: currentLength)
mutableString.deleteCharacters(in: currentRange)

print(mutableString)
```

**Commentary:**

This example demonstrates a mutable string where the length changes over time, leading to a possible `NSRange` exception. We begin with a mutable string and store its length.  However, after we append additional characters, the initially calculated range based on `originalLength` becomes invalid as the string has grown. Attempting to delete characters based on the obsolete length would lead to an `NSRange` exception as the length we want to delete has exceeded the total length of the updated string. In the code above, I have commented out the portion that would cause the error. The fixed code recalculates the current length and then creates a valid range based on the current length. This emphasizes the importance of always using current values for length, especially when dealing with mutable objects.

**Example 3: Data Subrange Handling**

The principles that cause `NSRange` exceptions with strings apply to other data structures as well such as data objects.

```swift
let myData = Data(bytes: [0x01, 0x02, 0x03, 0x04, 0x05])
let badDataRange = NSRange(location: 2, length: 5) // Location 2 + Length 5 goes beyond the data length 5

// The following will cause an NSRange exception.
// let subData = myData.subdata(in: Range(badDataRange, in: myData)!)

// The corrected version:
if badDataRange.location < myData.count && badDataRange.location + badDataRange.length <= myData.count{
    let subData = myData.subdata(in: Range(badDataRange, in: myData)!)
    print(subData) // Prints <030405>
} else {
    print("Invalid data range")
}

```

**Commentary:**

This demonstrates an `NSRange` exception with Data types. Similar to string handling, we create a data object. The `badDataRange` is initialized with a location and length that would, if used, result in a range exceeding the bounds of the data object. The error is caused by the request to access 5 bytes starting from index 2 of a data object that has only 5 total bytes of data. The fix mirrors the string fixes by applying conditional checks based on the data object length to ensure a valid range. These examples showcase that range-checking best practices extend across different data types and that a rigorous approach to validating range values is essential to prevent exceptions.

To mitigate these types of errors, I would recommend becoming deeply familiar with the debugging tools available for the respective IDE. Specifically, setting breakpoints before any range calculations, then stepping line-by-line is useful in identifying the exact location of error. In cases where the source code does not contain the error, examining the call stacks in the debugger is extremely important to identify which specific function call led to the error, as this might point to a different function passing in a faulty range value.

For additional learning on the topic I would recommend focusing on documentation provided by Apple. Thoroughly reviewing the Apple documentation for `NSString`, `NSMutableString`, `String`, `Data`, and the `NSRange` structure itself is extremely valuable. Beyond the formal documentation, there are several books on iOS development that extensively cover common errors and best practices, which also includes dealing with `NSRange` exceptions. Books like "Programming in Objective-C" by Stephen Kochan, and other introductory or advanced books on iOS and Swift programming can offer valuable insight and perspectives on these kinds of issues. These references collectively provide a more comprehensive understanding of `NSRange` handling, going beyond the immediate issue and expanding to include best practices, common anti-patterns, and effective debugging methods.
