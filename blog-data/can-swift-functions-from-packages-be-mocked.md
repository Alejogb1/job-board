---
title: "Can Swift functions from packages be mocked?"
date: "2024-12-23"
id: "can-swift-functions-from-packages-be-mocked"
---

Alright, let's unpack this. Mocking functions from packages in Swift – it's a topic that's come up more times than I can count in my years of building iOS and macOS applications, and I've seen some, let's say, 'interesting' approaches to handling it. The short answer is: yes, absolutely, but the *how* is what we need to focus on, because it's not always straightforward. It’s a crucial skill for robust unit testing, and avoiding its pitfalls will save you a lot of debugging headaches later on.

I remember one project, about five years back, where we integrated a third-party package for handling network requests. Initially, everything worked fine. Then, as the project grew, we started noticing intermittent failures in our unit tests. We were mocking our local code thoroughly, but the network logic from that package, which we were using in multiple components, was causing chaos. We couldn't reliably test different scenarios, like network errors or different responses, which was a nightmare. That's when I really had to dive deep into effective mocking strategies for packages.

Now, the core challenge lies in the fact that Swift doesn't have built-in, direct mocking mechanisms like other languages might. We don't have a `mock` keyword or something similar. Instead, we rely on techniques like protocol-based dependency injection and, in some cases, more advanced approaches like using type erasure. The main goal is to decouple our code from the concrete implementation provided by the package, which allows us to substitute our custom, controllable implementations during testing.

Let's start with the most common and highly recommended approach: using protocols. Imagine this third-party package provides a `NetworkClient` struct with a function like `fetchData(from url: URL) async throws -> Data`. Instead of directly using `NetworkClient` everywhere, we’d define a protocol:

```swift
protocol NetworkService {
    func fetchData(from url: URL) async throws -> Data
}
```

Then, we make the `NetworkClient` conform to this protocol:

```swift
// Assuming 'ThirdPartyPackage' is where the original NetworkClient lives
import ThirdPartyPackage

extension ThirdPartyPackage.NetworkClient: NetworkService {
    func fetchData(from url: URL) async throws -> Data {
        return try await self.fetchData(from: url)
    }
}
```

Now, in our application code, we don't refer to `ThirdPartyPackage.NetworkClient` directly; instead, we use the `NetworkService` protocol. This allows us to inject different implementations: the real one in production and a mock one in our tests. Here’s how we might construct a mock:

```swift
class MockNetworkService: NetworkService {
    var dataToReturn: Data?
    var errorToThrow: Error?
    var urlsCalled: [URL] = []

    func fetchData(from url: URL) async throws -> Data {
        urlsCalled.append(url)
        if let error = errorToThrow {
            throw error
        }
        guard let data = dataToReturn else {
            throw NSError(domain: "MockNetworkService", code: 1, userInfo: [NSLocalizedDescriptionKey: "No data to return"])
        }
        return data
    }
}
```

This `MockNetworkService` allows us to control the returned data, errors, and even track which URLs were accessed. This is powerful. We could then use it in our tests like this:

```swift
func testMyDataFetchingFunction() async throws {
    let mockService = MockNetworkService()
    let testData = "Test Data".data(using: .utf8)!
    mockService.dataToReturn = testData
    let testURL = URL(string: "https://example.com/data")!

    // Somewhere, the system uses NetworkService protocol, so we inject our mock
    let myObject = MyObject(networkService: mockService)
    let result = try await myObject.loadData(from: testURL) // Assume loadData uses NetworkService

    XCTAssertEqual(result, testData)
    XCTAssertEqual(mockService.urlsCalled.first, testURL) // Ensure correct url was used
}
```

This example illustrates the power of protocol-based dependency injection. We’ve completely decoupled our test from the concrete `NetworkClient` provided by the third-party package.

Sometimes, however, you might be working with a package where the function you want to mock isn't neatly wrapped in a protocol, or the package doesn't offer interfaces explicitly. In these scenarios, we can resort to slightly more sophisticated techniques. One approach is creating a wrapper class or function using closures. While it might add an extra layer, it allows you to control how that function is executed during testing. Let’s illustrate: Suppose the package has a static function `Utility.process(data: Data) -> Result<String, Error>`. You can't directly conform static functions to protocols.

First, create a wrapper:

```swift
class UtilityWrapper {
    typealias ProcessClosure = (Data) -> Result<String, Error>
    var processData: ProcessClosure

    init(processData: @escaping ProcessClosure = Utility.process) {
        self.processData = processData
    }

    func process(data: Data) -> Result<String, Error> {
        return processData(data)
    }
}
```

And now, a mock wrapper:

```swift
class MockUtilityWrapper: UtilityWrapper {
    var resultToReturn: Result<String, Error>?
    var dataPassed: Data?

    override init(processData: @escaping (Data) -> Result<String, Error> = { _ in .failure(NSError(domain: "Mock", code: 0)) }) {
        super.init(processData: processData)
    }
    
    override func process(data: Data) -> Result<String, Error> {
        dataPassed = data
        if let result = resultToReturn {
            return result
        }
        return .failure(NSError(domain: "Mock", code: 1))
    }
}
```

Finally, use the wrapper in your code, injecting the appropriate version:

```swift
func processDataFunction(data: Data, utility: UtilityWrapper) -> String? {
    let result = utility.process(data: data)
    switch result {
    case .success(let string):
        return string
    case .failure:
        return nil
    }
}


func testProcessingData() {
    let testData = "Some data".data(using: .utf8)!
    let mockUtility = MockUtilityWrapper()
    mockUtility.resultToReturn = .success("Processed data")

    let processedString = processDataFunction(data: testData, utility: mockUtility)
    XCTAssertEqual(processedString, "Processed data")
    XCTAssertEqual(mockUtility.dataPassed, testData)

}
```

A third, more involved approach, is using type erasure. While it can achieve the desired mocking effect, it should be used with caution as it introduces more complexity. Its primary purpose here would be to bypass limitations with static or global functions. For instance, if that `Utility.process` function was a global one, not associated with a struct, creating an abstraction over it with type erasure allows you to substitute the implementations for testing. I won't detail it with code here because it is significantly more complex and can introduce potential performance overhead. If type erasure is a path you're interested in exploring, I strongly advise looking into advanced literature and examples to fully understand its ramifications.

For furthering your understanding, I highly recommend looking at “Dependency Injection in Swift” by Jon Reid and “Working Effectively with Unit Tests” by Jay Fields. These resources will give you a solid grounding in dependency injection and its essential role in mock testing. The Swift programming language documentation itself is also valuable, especially when delving into protocols and generics.

In summary, while mocking functions from packages in Swift isn't a first-class feature, the techniques available, such as protocol-based dependency injection and wrapper classes, give you the necessary control for effective unit testing. It’s about decoupling your code, making it testable, and ensuring you have a solid foundation that is resistant to changes in those dependencies. Through careful design and application of these practices, you'll avoid the common pitfalls of coupled tests and achieve more robust software.
