---
title: "How to stub promises using Sinon?"
date: "2024-12-23"
id: "how-to-stub-promises-using-sinon"
---

Alright, let's tackle stubbing promises with Sinon. This is a topic I've spent a good amount of time with, having debugged quite a few flaky test suites back in my days working on a complex microservices architecture, where proper mocking and stubbing were absolutely paramount. A lot of our interactions were asynchronous, relying heavily on promises, and getting the stubs *just so* was critical. Let's dive into how to do this effectively, keeping things practical and focused on real-world scenarios.

The fundamental idea behind stubbing promises is to replace an actual asynchronous function, which returns a promise, with a controllable substitute. This substitute allows us to dictate how that promise resolves or rejects, giving us granular control over the asynchronous behavior in our tests. Sinon, thankfully, provides robust tools for accomplishing this.

I'll start with the basic approach and then move into some more nuanced scenarios. The key is understanding that a promise essentially has two paths: resolution with a value or rejection with an error. Stubbing needs to handle both.

Let's consider a simple example where you have a function, let's call it `fetchUserData`, that returns a promise:

```javascript
function fetchUserData(userId) {
    // Imagine this makes an actual HTTP call, but let's
    // simulate a successful response for now
    return new Promise(resolve => {
        setTimeout(() => {
           resolve({ id: userId, name: 'Test User' });
        }, 100);
    });
}
```

Now, let’s say we have a testing scenario where we want to examine a component that utilizes this `fetchUserData` function. Without proper stubbing, your tests would be unnecessarily dependent on the availability of the service it's trying to interact with, slowing tests down and making them unreliable. This is where Sinon steps in. We’d want to stub `fetchUserData` to control its outcome.

Here’s an example of stubbing this function to resolve with a predefined value:

```javascript
const sinon = require('sinon');

// We'll assume fetchUserData is available
// from some module, say, 'api-client.js',
// so we import it here for demonstration.
const apiClient = { fetchUserData };

describe('Component using fetchUserData', () => {
   it('should handle successful user fetch', async () => {
        const userId = '123';
        const stub = sinon.stub(apiClient, 'fetchUserData');

        stub.resolves({ id: userId, name: 'Stubbed User' }); // Stub resolves

        const result = await apiClient.fetchUserData(userId);

        expect(result).to.deep.equal({ id: userId, name: 'Stubbed User' });
        expect(stub).to.have.been.calledWith(userId); // Verify that it was called as intended
        stub.restore(); // Important to clean up stubs after usage
    });
});
```

In this first code block, we’re using `sinon.stub` to replace the original `fetchUserData` function on the `apiClient` object with our substitute. The critical part here is `stub.resolves({ id: userId, name: 'Stubbed User' })`. This instructs our stub to resolve the promise it returns with the provided object, completely bypassing the original implementation of the `fetchUserData` method. We also check to ensure the stub was called with the correct user id and finally, `stub.restore()` cleans up the stub, returning the original function to the `apiClient` object.

But what about when you need to simulate a failure? That’s equally critical. Here's a demonstration on how to handle rejected promises with stubbing:

```javascript
const sinon = require('sinon');

const apiClient = { fetchUserData };


describe('Component using fetchUserData', () => {
    it('should handle errors during user fetch', async () => {
         const userId = '123';
         const stub = sinon.stub(apiClient, 'fetchUserData');
         const errorMessage = 'Failed to fetch user';

         stub.rejects(new Error(errorMessage)); // Stub rejects with an Error

         let caughtError;
         try {
            await apiClient.fetchUserData(userId);
         } catch(e) {
           caughtError = e;
         }


         expect(caughtError).to.be.an('error');
         expect(caughtError.message).to.equal(errorMessage);
         expect(stub).to.have.been.calledWith(userId);
         stub.restore();
     });
});
```

In this second code example, instead of `resolves()`, we use `rejects(new Error(errorMessage))`. This forces our stubbed promise to reject with the specified error, allowing our tests to verify the error handling logic within our code. This ensures that the components consuming our `fetchUserData` method correctly manage exceptional scenarios.

Sometimes, you might encounter scenarios where the promise resolution needs to be dynamic. For instance, you might want to make the resolve or reject behavior dependent on the input arguments passed to the function. Here's how to achieve that:

```javascript
const sinon = require('sinon');
const apiClient = { fetchUserData };


describe('Component using fetchUserData', () => {
 it('should handle fetch based on input params', async () => {
      const stub = sinon.stub(apiClient, 'fetchUserData');

      stub.callsFake((userId) => { // Stub with callsFake and dynamic resolving
          if (userId === 'validUser') {
              return Promise.resolve({ id: userId, name: 'Valid User' });
          } else {
              return Promise.reject(new Error('Invalid user id'));
          }
      });

      let validResult = await apiClient.fetchUserData('validUser');
      expect(validResult).to.deep.equal({ id: 'validUser', name: 'Valid User' });

      let caughtError;
      try {
           await apiClient.fetchUserData('invalidUser');
      } catch(e) {
        caughtError = e;
      }
      expect(caughtError).to.be.an('error');
      expect(caughtError.message).to.equal('Invalid user id');

      expect(stub).to.have.been.calledTwice;
      stub.restore();
    });
});
```
Here, `callsFake` allows us to define a function that will be executed every time the stub is called. This function receives the arguments passed to the stub and we can then decide the resolution or rejection based on these arguments. This introduces the concept of conditional or dynamic stubbing of promises, greatly increasing the testability of complex flows.

When working with Sinon stubs, it’s important to remember that you are not altering the original function in any permanent way. The `stub.restore()` ensures that any changes you make are temporary and are confined within the scope of the test suite. Failing to do this can lead to test pollution where changes persist beyond what was intended.

For further exploration into best practices for stubbing and mocking in JavaScript, I'd recommend “Test Driven Development: By Example” by Kent Beck. Though it focuses more broadly on TDD, it provides essential principles that underpin effective stubbing. Additionally, the testing documentation of popular frameworks like Jest and Mocha often contains useful examples that could help deepen the understanding of stubbing asynchronous code, as they often rely on similar concepts, or you could explore the advanced features of Sinon itself by going through the official documentation. The key here is practice and a thoughtful approach to each test case. Over time, you'll develop the intuition needed to create robust and reliable test suites. I hope these examples, along with the cited resources, provide a good starting point for you.
