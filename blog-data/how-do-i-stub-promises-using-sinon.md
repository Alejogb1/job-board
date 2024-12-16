---
title: "How do I stub promises using Sinon?"
date: "2024-12-16"
id: "how-do-i-stub-promises-using-sinon"
---

Let’s tackle this. Over the years, I've spent more than my share of time debugging asynchronous tests, and effective stubbing of promises, especially using a tool like Sinon, has proven to be absolutely critical. It's not merely about making tests pass; it's about ensuring our test scenarios accurately reflect the real-world behaviors we're trying to validate, isolating units of code effectively. I recall one particularly challenging project involving a complex data pipeline, where inconsistent network responses made testing almost impossible. That's where Sinon’s power really came into play, allowing us to simulate all manner of promise resolutions and rejections consistently, ultimately providing a much more robust testing suite.

Essentially, stubbing promises with Sinon means replacing an actual function that returns a promise with a controlled simulation of a promise. This allows us to dictate how the promise resolves (with a value) or rejects (with an error), thereby controlling the flow of asynchronous operations and the subsequent behavior of the code under test. It's about creating predictable test environments where you can isolate the component you’re focusing on. It cuts out external dependencies and helps zero in on potential bugs in your current code.

The fundamental concept relies on Sinon's `stub()` method, which, in its most basic form, replaces a function with a mock function. The magic happens when you configure this mock function to return a promise. There are a couple of ways to do this, depending on whether you need a promise that resolves, rejects, or both.

Let’s look at a few illustrative examples:

**Example 1: Stubbing a Resolving Promise**

Imagine a function, `fetchUserData`, which is supposed to return a promise that resolves with user data. Here’s how you might stub it using Sinon:

```javascript
const sinon = require('sinon');

async function fetchUserData(userId) {
  // Assume this is a real network call that returns a promise.
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({ id: userId, name: "Test User" });
    }, 50); // Simulate some asynchronous delay
  });
}

async function processUserData(userId) {
    const userData = await fetchUserData(userId);
    return `User ${userData.name} (ID: ${userData.id}) processed.`;
}


describe('processUserData', () => {
  it('should process user data correctly when fetchUserData resolves', async () => {
    const userId = 123;
    const stubbedUserData = { id: userId, name: 'Stubbed User' };
    const fetchUserDataStub = sinon.stub();
    fetchUserDataStub.resolves(stubbedUserData);

    // Replace the real fetchUserData with the stub
    const originalFetchUserData = fetchUserData; // Keep a reference
    fetchUserData = fetchUserDataStub;

    const result = await processUserData(userId);

    // Restore the original function for other tests
    fetchUserData = originalFetchUserData;


    expect(result).toEqual('User Stubbed User (ID: 123) processed.');
    expect(fetchUserDataStub.calledOnce).toBe(true);
    expect(fetchUserDataStub.firstCall.args[0]).toBe(userId);

  });
});
```

In this example, `sinon.stub()` creates a stub that replaces the real implementation of `fetchUserData`. The critical part is `.resolves(stubbedUserData)`, which configures the stub to return a promise that immediately resolves with the value `stubbedUserData`. By replacing the real `fetchUserData` function with this stub, our test can control precisely what the promise will return. I've included verification of `calledOnce` and argument checking to make sure the stub was called correctly. This is a crucial step as well. Without it, there is no guarantee that the stub was actually ever called. Remember, replacing the original function is very important; without it, you would be calling the real implementation instead.

**Example 2: Stubbing a Rejecting Promise**

Now, let’s consider a scenario where we want to simulate a failed request. We can use `.rejects()` for that.

```javascript
const sinon = require('sinon');

async function fetchDataFromApi(url) {
  return new Promise((resolve, reject) => {
        setTimeout(() => {
             reject(new Error('Network Error'));
        }, 50);
  });
}

async function handleApiCall(url) {
  try {
    const data = await fetchDataFromApi(url);
    return data;
  } catch (error) {
    throw new Error(`Failed to fetch data from api: ${error.message}`);
  }
}

describe('handleApiCall', () => {
  it('should handle rejected promises from fetchDataFromApi correctly', async () => {
      const url = 'https://api.example.com/data';
      const fetchApiStub = sinon.stub();
      const errorMessage = "Simulated API Error";
      fetchApiStub.rejects(new Error(errorMessage));

      const originalFetchDataFromApi = fetchDataFromApi;
      fetchDataFromApi = fetchApiStub;

    await expect(handleApiCall(url)).rejects.toThrow(`Failed to fetch data from api: ${errorMessage}`);
      expect(fetchApiStub.calledOnce).toBe(true);
      expect(fetchApiStub.firstCall.args[0]).toBe(url);

        fetchDataFromApi = originalFetchDataFromApi;
  });
});
```

Here, `fetchApiStub.rejects(new Error(errorMessage))` makes the stub return a promise that is immediately rejected with an error message, emulating a failed network call. This allows us to test the error handling logic within `handleApiCall`. I've once again included checks for call count and the input parameters. I’ve also used `expect().rejects.toThrow()`, which is the recommended approach for testing promise rejections in most test frameworks.

**Example 3: Stubbing with Dynamic Resolution**

Sometimes, you might need more nuanced control over promise behavior, perhaps resolving based on the input arguments. For that, you can leverage the `callsFake()` method to supply a function that dynamically decides how to resolve.

```javascript
const sinon = require('sinon');

async function makeCalculation(a, b, operation) {
    return new Promise(resolve => {
        setTimeout(() => {
            if (operation === 'add') {
                resolve(a + b);
            } else if (operation === 'subtract') {
              resolve(a - b);
            } else {
                resolve(null);
            }
          },50)
      });
}

async function calculate(a,b,operation){
  const result = await makeCalculation(a,b,operation);
  return `Calculation result is: ${result}`;
}


describe('calculate', () => {
  it('should dynamically calculate based on the stubbed operation', async () => {
    const addStub = sinon.stub();
      addStub.callsFake((a, b, operation) => {
        if(operation === 'add'){
            return Promise.resolve(a+b);
        } else if(operation === 'subtract') {
          return Promise.resolve(a-b);
        }
        return Promise.resolve(null);
      });

        const originalMakeCalculation = makeCalculation;
    makeCalculation = addStub;


    const addResult = await calculate(5, 3, 'add');
    const subResult = await calculate(5,3, 'subtract');
    const invalidResult = await calculate(5,3, 'multiply');
    
      makeCalculation = originalMakeCalculation;

    expect(addResult).toEqual('Calculation result is: 8');
    expect(subResult).toEqual('Calculation result is: 2');
      expect(invalidResult).toEqual('Calculation result is: null');

      expect(addStub.callCount).toBe(3);
        expect(addStub.firstCall.args).toEqual([5,3,'add']);
         expect(addStub.secondCall.args).toEqual([5,3,'subtract']);
    expect(addStub.thirdCall.args).toEqual([5,3,'multiply']);

  });
});
```

Here, we've set up a more elaborate simulation using `callsFake()`. The function provided to `callsFake()` allows us to inspect the arguments passed to `makeCalculation` and return a different promise based on those values. We’re effectively creating a dynamic mock that responds differently based on test inputs, thereby providing flexibility in test design. Once again, I've included checks for the correct call count and parameters to ensure the stub is working as intended. It also demonstrates how the same stub can be used multiple times with varied input, resulting in different resolved values.

For further study, I’d highly recommend exploring the official Sinon.js documentation thoroughly. It’s incredibly well-written and comprehensive. Beyond that, “Test-Driven Development: By Example” by Kent Beck is a fundamental resource that will give you a deeper understanding of the underlying test principles that make practices such as stubbing so impactful. Additionally, any good book focused on testing JavaScript applications, like “Effective JavaScript” by David Herman, will likely have a good explanation and practical examples of test doubles, like stubs. Understanding the principles of software testing along with mastery of tools like Sinon is essential.

In closing, effective promise stubbing with Sinon isn’t just about avoiding actual API calls; it’s about crafting precise, focused tests that truly evaluate the behavior of your code under various scenarios. The examples above should give a solid foundation for your work, but remember that continual practice and refining your testing strategy is key to producing robust and maintainable software.
