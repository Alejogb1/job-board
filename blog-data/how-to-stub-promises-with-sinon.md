---
title: "How to stub promises with sinon?"
date: "2024-12-16"
id: "how-to-stub-promises-with-sinon"
---

, let's talk about stubbing promises with sinon. I've had my fair share of dealing with asynchronous testing, and let me tell you, stubbing promises correctly can save a *lot* of headache. It's not always immediately obvious how to do it, especially when you first start using sinon. It’s one of those things where a little practice and understanding of what's happening under the hood goes a long way.

The core issue when testing with asynchronous operations, like those returning promises, is maintaining control over the execution flow. You need to isolate the unit under test and control what gets returned from its dependencies – otherwise, you are venturing into integration test territory, which can get complex very quickly. That's where sinon's stubbing capabilities become invaluable, allowing us to meticulously mock the behaviour of promise-returning functions. We want to ensure our unit behaves correctly irrespective of the actual underlying asynchronous operations that it depends on.

There are several ways to stub a promise-returning function, but the general idea is to use `sinon.stub()` to intercept the call to the function you want to control, and then configure it to return a promise that either resolves with a predefined value or rejects with an error. The first thing that comes to mind is that many developers first encounter the need for this after attempting to test their code that leverages the `async`/`await` syntax which, under the hood, still utilizes promises. It's not a black box, though it often feels like it!

Let's look at a few different scenarios, starting with a simple resolved promise.

**Example 1: Stubbing a function to resolve with a value**

Suppose we have a service function, `fetchUserData`, which returns a promise that resolves with user data:

```javascript
// service.js (hypothetical example)
async function fetchUserData(userId) {
    // Pretend this is an API call
    return new Promise(resolve => {
        setTimeout(() => {
            resolve({ id: userId, name: 'Test User' });
        }, 50); // Simulate latency
    });
}

module.exports = { fetchUserData };
```

And now, imagine we have a component using this:

```javascript
// component.js
const { fetchUserData } = require('./service');

async function displayUserName(userId) {
    const userData = await fetchUserData(userId);
    return userData.name;
}

module.exports = { displayUserName };
```

Now here’s how we would test this using `sinon.stub()`, specifically mocking the `fetchUserData` function to resolve with a specific value rather than the real async operation. We use `resolves()` after our `stub()` definition which allows us to indicate the value the stubbed function will resolve to.

```javascript
// component.test.js
const sinon = require('sinon');
const { fetchUserData } = require('./service');
const { displayUserName } = require('./component');
const assert = require('assert');

describe('displayUserName', () => {
    it('should display user name correctly', async () => {
        const stub = sinon.stub(fetchUserData, 'fetchUserData');
        stub.resolves({ id: 123, name: 'Stubbed User' });

        const userName = await displayUserName(123);
        assert.strictEqual(userName, 'Stubbed User');
        stub.restore(); // Restore the original method, don't forget this!
    });
});
```

In the above example, the `fetchUserData` function, instead of actually performing any asynchronous operation, will immediately return a resolved promise with the defined value, meaning that our test runs synchronously and the underlying async operation is never executed. This isolates the `displayUserName` logic for unit testing.

**Example 2: Stubbing a function to reject with an error**

Sometimes, we need to test error handling. Stubbing to reject a promise allows us to simulate various scenarios such as API failures. Let’s say we need to simulate that `fetchUserData` function failing due to a network issue or other problem. We can utilize the `.rejects()` method within our stub:

```javascript
// component.test.js (modified)
const sinon = require('sinon');
const { fetchUserData } = require('./service');
const { displayUserName } = require('./component');
const assert = require('assert');

describe('displayUserName', () => {
    it('should handle error when user data fetching fails', async () => {
        const stub = sinon.stub(fetchUserData, 'fetchUserData');
        stub.rejects(new Error('Network Error'));

        try {
            await displayUserName(123);
            assert.fail('Expected an error to be thrown');
        } catch (error) {
            assert.strictEqual(error.message, 'Network Error');
            stub.restore();
        }
    });
});
```

Here, the stubbed function `fetchUserData` immediately returns a rejected promise carrying an error. We wrap the `displayUserName` call within a `try...catch` block to assert that the function correctly handles the rejected promise.

**Example 3: Stubbing different outcomes based on parameters**

In some cases, the response from a function depends on its parameters. Let’s imagine a more complex scenario where the API might return different values for different user ids. In that case we would use the `.withArgs()` method in the stubbed method chain.

```javascript
// component.test.js (modified further)
const sinon = require('sinon');
const { fetchUserData } = require('./service');
const { displayUserName } = require('./component');
const assert = require('assert');

describe('displayUserName', () => {
    it('should display correct user name based on id', async () => {
        const stub = sinon.stub(fetchUserData, 'fetchUserData');
        stub.withArgs(123).resolves({ id: 123, name: 'User One' });
        stub.withArgs(456).resolves({ id: 456, name: 'User Two' });


        let userName = await displayUserName(123);
        assert.strictEqual(userName, 'User One');
        userName = await displayUserName(456);
        assert.strictEqual(userName, 'User Two');
        stub.restore();
    });
});
```

In this test, we are stubbing the same function with different responses depending on the arguments that are passed to it. This showcases the flexibility of the library and allows us to mock complex interactions within our units.

When it comes to further reading, I’d highly recommend checking out Martin Fowler's "Mocks Aren't Stubs" article to further clarify the differences between mocks and stubs. Also, "Test-Driven Development: By Example" by Kent Beck is a seminal book that provides further insights on effective unit testing, including how to design testable systems. Finally, if you want to delve into the intricacies of testing asynchronous code, "Effective JavaScript" by David Herman, although not strictly about testing, does a great job at explaining how JavaScript's asynchronous operations work, which is critical when working with promises. Knowing the underlying mechanisms is as crucial as the libraries that help with testing.

Stubbing promises correctly isn’t simply a matter of syntax; it's about understanding the control you need in asynchronous unit tests. Use `.resolves()` for success scenarios, `.rejects()` for simulating errors, and `.withArgs()` to handle conditional responses based on inputs. And always, *always*, remember to restore the original function with `stub.restore()` after the test is complete. Failure to restore stubs can create subtle, hard-to-track issues in your subsequent tests. It’s all about meticulous testing to build robust applications.
