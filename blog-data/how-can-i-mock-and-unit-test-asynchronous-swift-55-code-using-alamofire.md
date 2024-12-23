---
title: "How can I mock and unit test asynchronous Swift 5.5 code using Alamofire?"
date: "2024-12-23"
id: "how-can-i-mock-and-unit-test-asynchronous-swift-55-code-using-alamofire"
---

Alright, let's tackle this. Been down this road more times than I can count, especially when dealing with network layers in Swift. Mocking and unit testing asynchronous code, particularly when Alamofire is involved, definitely has its nuances. It’s not as straightforward as synchronous code, but with the right approach, it’s entirely manageable. I’ve personally seen several projects bogged down by a lack of proper testing for their network interactions, so this is a topic I feel strongly about.

The core challenge, as you probably know, comes from the asynchronous nature of network requests. Alamofire operates on background threads, and your unit tests are typically run on the main thread. To properly test this, you've got to effectively control and observe the asynchronous behavior. Simply put, we need to isolate our network layer so we're not making actual API calls during tests, and we need to handle how the results are returned.

My first, and arguably most crucial recommendation, is to design your networking layer using protocols. This isn't a fancy abstraction for the sake of it. This is about decoupling your implementation from the specific network library you're using. You should create protocols that describe the behavior of your network calls rather than relying directly on concrete classes. Here's how I typically set things up:

```swift
protocol NetworkService {
    func fetchUsers(completion: @escaping (Result<[User], Error>) -> Void)
    func postUser(user: User, completion: @escaping (Result<User, Error>) -> Void)
}

struct User: Codable {
    let id: Int
    let name: String
}

```
This `NetworkService` protocol defines the contracts we’ll adhere to. Note, I’m using the Swift `Result` type for error handling; it's cleaner and more explicit. Now, your actual implementation using Alamofire would conform to this protocol.

```swift
import Alamofire

class AlamofireNetworkService: NetworkService {
    func fetchUsers(completion: @escaping (Result<[User], Error>) -> Void) {
        AF.request("https://api.example.com/users").responseDecodable(of: [User].self) { response in
            switch response.result {
            case .success(let users):
                completion(.success(users))
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }

    func postUser(user: User, completion: @escaping (Result<User, Error>) -> Void) {
      AF.request("https://api.example.com/users", method: .post, parameters: user, encoder: JSONParameterEncoder.default).responseDecodable(of: User.self) { response in
            switch response.result {
            case .success(let postedUser):
                completion(.success(postedUser))
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
}
```

This example, while basic, shows the actual usage of Alamofire. However, remember *this is what we do not want to use during our unit tests*. Our tests should be fast, predictable, and not dependent on network connectivity. That's where mocking comes in.

For mocking, we’ll create a test double that also conforms to `NetworkService`. This double won't make actual network requests; instead, it will return predefined data or simulate errors. Let’s create a basic mock:

```swift
class MockNetworkService: NetworkService {
    var usersResult: Result<[User], Error>?
    var postUserResult: Result<User, Error>?

    func fetchUsers(completion: @escaping (Result<[User], Error>) -> Void) {
        if let result = usersResult {
            completion(result)
        }
    }
    func postUser(user: User, completion: @escaping (Result<User, Error>) -> Void) {
        if let result = postUserResult {
            completion(result)
        }
    }

    func setFetchUsersResult(result: Result<[User], Error>){
        usersResult = result
    }
    func setPostUserResult(result: Result<User,Error>){
        postUserResult = result
    }
}
```

Now, for your unit tests, you inject your `MockNetworkService` where you would normally use `AlamofireNetworkService`. The beauty here is that your test setup can control the outcome of the network calls, making your tests deterministic. Here's a basic example of what a test might look like using XCTest:

```swift
import XCTest

class UserService {
    private let networkService: NetworkService

    init(networkService: NetworkService) {
        self.networkService = networkService
    }

    func getUsers(completion: @escaping (Result<[User], Error>) -> Void) {
        networkService.fetchUsers(completion: completion)
    }

    func createUser(user: User, completion: @escaping (Result<User,Error>) -> Void) {
        networkService.postUser(user: user, completion: completion)
    }
}

class UserServiceTests: XCTestCase {
    func testFetchUsersSuccess() {
        let mockNetworkService = MockNetworkService()
        let userService = UserService(networkService: mockNetworkService)
        let expectedUsers = [User(id: 1, name: "John Doe"), User(id: 2, name: "Jane Doe")]
        mockNetworkService.setFetchUsersResult(result: .success(expectedUsers))
        let expectation = XCTestExpectation(description: "Fetch users success")
        userService.getUsers { result in
            switch result {
            case .success(let users):
                XCTAssertEqual(users.count, 2)
                XCTAssertEqual(users[0].id, 1)
                XCTAssertEqual(users[1].name, "Jane Doe")
                expectation.fulfill()
            case .failure(_):
               XCTFail("Expected success, but got failure")
            }
        }
        wait(for: [expectation], timeout: 1.0)

    }

    func testCreateUserSuccess() {
        let mockNetworkService = MockNetworkService()
        let userService = UserService(networkService: mockNetworkService)
        let user = User(id: 3, name: "Peter Pan")
        mockNetworkService.setPostUserResult(result: .success(user))
        let expectation = XCTestExpectation(description: "Post user success")
        userService.createUser(user: user) { result in
            switch result {
            case .success(let postedUser):
                XCTAssertEqual(postedUser.id, 3)
                XCTAssertEqual(postedUser.name, "Peter Pan")
                expectation.fulfill()
            case .failure(_):
                XCTFail("Expected success, but got failure")
            }
        }
        wait(for: [expectation], timeout: 1.0)
    }


    func testFetchUsersFailure() {
        let mockNetworkService = MockNetworkService()
        let userService = UserService(networkService: mockNetworkService)
        let expectedError = NSError(domain: "TestErrorDomain", code: 100, userInfo: [NSLocalizedDescriptionKey: "Test error message"])
        mockNetworkService.setFetchUsersResult(result: .failure(expectedError))
         let expectation = XCTestExpectation(description: "Fetch users failure")
        userService.getUsers { result in
            switch result {
            case .success(_):
                XCTFail("Expected failure, but got success")
            case .failure(let error):
              let nsError = error as NSError
                XCTAssertEqual(nsError.domain, "TestErrorDomain")
                XCTAssertEqual(nsError.code, 100)
                XCTAssertEqual(nsError.localizedDescription, "Test error message")
               expectation.fulfill()
            }
         }
        wait(for: [expectation], timeout: 1.0)
    }
}
```
I’ve used `XCTestExpectation` to handle the asynchronous nature of the tests, waiting for the callbacks to be executed before asserting the results. This pattern has served me incredibly well across multiple projects. This allows you to test the behavior of your code given various network outcomes (successes, failures, and different data) without actually relying on the network at all.

For deeper insight into these kinds of testing practices, I would highly recommend checking out *Growing Object-Oriented Guided by Tests* by Steve Freeman and Nat Pryce. It's a classic on unit testing and testing concepts. Also, *Working Effectively with Legacy Code* by Michael Feathers has very useful tips on how to test code that hasn't been designed with testability in mind and is highly useful when dealing with existing network implementations. For a specific view on protocols and dependency injection in swift, I recommend you study the material on the Swift documentation, specifically articles related to Protocol-oriented programming.

The key takeaways here are: abstract your network layer behind a protocol, create mocks conforming to that protocol to simulate network responses in your tests, and utilize XCTestExpectations to handle asynchronous results in tests. Remember, your tests should be as reliable and fast as possible, and following this pattern can help achieve that.
