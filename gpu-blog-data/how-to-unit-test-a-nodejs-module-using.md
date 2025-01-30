---
title: "How to unit test a Node.js module using Mocha and Chai?"
date: "2025-01-30"
id: "how-to-unit-test-a-nodejs-module-using"
---
Unit testing forms the bedrock of reliable software, and in my experience developing Node.js modules, Mocha coupled with Chai provides a robust framework for verifying individual components. Specifically, I've found that a well-structured test suite, using these tools effectively, drastically reduces the incidence of regressions and promotes a more maintainable codebase.

The core principle is isolation: each unit test should focus on a single, atomic piece of functionality, typically a function or a method. This requires meticulous setup and teardown routines to ensure tests don't inadvertently affect each other, and that test state is properly initialized and reset between each test execution.

Mocha serves as the test runner; it discovers and executes test suites, providing a structured environment and reports test results. Chai, on the other hand, is an assertion library; it offers a set of expressive methods to verify the expected outcome of a given test against the actual outcome of the code being tested. It's this combination that allows for clear, easily understandable test specifications.

Let's illustrate this with a few practical examples based on a hypothetical module I've worked on, one handling basic date manipulation. The module is intended to be simple, comprising three functions: `addDays`, `subtractDays`, and `formatDate`.

**Example 1: Testing `addDays` Functionality**

First, consider the `addDays` function which increments a given date by a specified number of days. The following code block demonstrates a unit test using Mocha and Chai:

```javascript
const { addDays } = require('../src/date-utils');
const { expect } = require('chai');

describe('addDays', () => {
  it('should add days correctly to a date', () => {
    const initialDate = new Date('2024-01-15T00:00:00.000Z');
    const daysToAdd = 5;
    const expectedDate = new Date('2024-01-20T00:00:00.000Z');
    const result = addDays(initialDate, daysToAdd);
    expect(result).to.eql(expectedDate);
  });

    it('should handle negative day additions correctly', () => {
      const initialDate = new Date('2024-01-15T00:00:00.000Z');
      const daysToAdd = -3;
      const expectedDate = new Date('2024-01-12T00:00:00.000Z');
      const result = addDays(initialDate, daysToAdd);
      expect(result).to.eql(expectedDate);
  });

  it('should handle adding zero days correctly', () => {
      const initialDate = new Date('2024-01-15T00:00:00.000Z');
      const daysToAdd = 0;
      const expectedDate = new Date('2024-01-15T00:00:00.000Z');
      const result = addDays(initialDate, daysToAdd);
      expect(result).to.eql(expectedDate);
  });


    it('should not mutate the original date object', () => {
      const initialDate = new Date('2024-01-15T00:00:00.000Z');
      const daysToAdd = 5;
      addDays(initialDate, daysToAdd);
      const originalDateCheck = new Date('2024-01-15T00:00:00.000Z');
      expect(initialDate).to.eql(originalDateCheck);
    });
});
```
In this test suite, `describe('addDays', ...)` groups together tests specifically for the `addDays` function.  Each `it('should ...', ...)` block represents a single test case, which tests a particular aspect of functionality. In the first test, we provide a known initial date, the number of days to add, and the expected result.  Chai's `expect(result).to.eql(expectedDate)` verifies that the actual result matches the expected result, using deep equality for date object comparison. The second and third tests check behavior for negative days and zero day increments respectively. A crucial test, the fourth, verifies that the `addDays` function operates immutably, ensuring it doesn’t change the original input date object, which is essential for avoiding side effects.

**Example 2: Testing `subtractDays` Functionality**

The `subtractDays` function should behave similarly to `addDays` but decrement days from the input date, and testing it also follows a similar structure.

```javascript
const { subtractDays } = require('../src/date-utils');
const { expect } = require('chai');

describe('subtractDays', () => {
  it('should subtract days correctly from a date', () => {
    const initialDate = new Date('2024-01-20T00:00:00.000Z');
    const daysToSubtract = 5;
    const expectedDate = new Date('2024-01-15T00:00:00.000Z');
    const result = subtractDays(initialDate, daysToSubtract);
    expect(result).to.eql(expectedDate);
  });

  it('should handle negative day subtraction correctly', () => {
      const initialDate = new Date('2024-01-12T00:00:00.000Z');
      const daysToSubtract = -3;
      const expectedDate = new Date('2024-01-15T00:00:00.000Z');
      const result = subtractDays(initialDate, daysToSubtract);
      expect(result).to.eql(expectedDate);
  });

  it('should handle subtracting zero days correctly', () => {
      const initialDate = new Date('2024-01-15T00:00:00.000Z');
      const daysToSubtract = 0;
      const expectedDate = new Date('2024-01-15T00:00:00.000Z');
      const result = subtractDays(initialDate, daysToSubtract);
      expect(result).to.eql(expectedDate);
  });


   it('should not mutate the original date object', () => {
    const initialDate = new Date('2024-01-20T00:00:00.000Z');
    const daysToSubtract = 5;
    subtractDays(initialDate, daysToSubtract);
     const originalDateCheck = new Date('2024-01-20T00:00:00.000Z');
    expect(initialDate).to.eql(originalDateCheck);
  });
});
```
This suite mirrors the structure of the `addDays` tests, verifying the core subtract function, negative arguments, zero arguments, and again the crucial non-mutating behavior of the function. This promotes a consistent, reliable testing strategy across module functions.

**Example 3: Testing `formatDate` Functionality**

Finally, consider `formatDate`, which is intended to format a date object into a specific string.

```javascript
const { formatDate } = require('../src/date-utils');
const { expect } = require('chai');

describe('formatDate', () => {
  it('should format a date as YYYY-MM-DD', () => {
    const date = new Date('2024-01-20T12:34:56.000Z');
    const expectedFormat = '2024-01-20';
    const result = formatDate(date);
    expect(result).to.equal(expectedFormat);
  });

    it('should handle a different date correctly', () => {
        const date = new Date('2023-12-31T23:59:59.999Z');
        const expectedFormat = '2023-12-31';
        const result = formatDate(date);
        expect(result).to.equal(expectedFormat);
    });

    it('should handle a date with single digit month and day', () => {
        const date = new Date('2024-02-03T10:00:00.000Z');
        const expectedFormat = '2024-02-03';
        const result = formatDate(date);
        expect(result).to.equal(expectedFormat);
    });

  it('should throw error if provided a non-date argument', () => {
     expect(() => formatDate("not a date")).to.throw(Error);
    });
});
```

The test cases verify that the date is formatted in the expected ‘YYYY-MM-DD’ format, including various scenarios for different input dates and ensure correct handling of leading zeros. This function is specifically tested for an error scenario when provided an invalid date, using chai's `.to.throw(Error)` assertion. This ensures robustness by confirming that the function handles invalid arguments gracefully.

**Resource Recommendations**

For gaining further insight, exploring the official Mocha and Chai documentation is highly recommended; both provide exhaustive information on their respective features. In addition, I've found that reviewing code examples from open-source Node.js projects often demonstrates best practices for applying these testing tools in practical settings. Understanding testing methodologies is crucial; I would also advise exploring books focusing on software testing principles. Specifically, pay attention to concepts like test-driven development (TDD) which encourages writing tests *before* implementation, and the principles of writing isolated, robust unit tests. Lastly, a deeper investigation into the differences between testing techniques such as integration and e2e, alongside unit testing, can help round out a more complete knowledge base on this subject.
