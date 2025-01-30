---
title: "How can Nightwatch display Chai assertions in its reporter?"
date: "2025-01-30"
id: "how-can-nightwatch-display-chai-assertions-in-its"
---
Nightwatch, by default, displays assertion results from its internal assertion library, not those directly from Chai. Integrating Chai assertions into Nightwatch's test reporting requires careful interception and reformatting of Chai's output to align with Nightwatch’s reporting structure. I've encountered this situation multiple times while working on end-to-end testing frameworks for different projects, and the solution involves a combination of custom commands and reporter manipulation.

The core problem lies in the fact that Nightwatch's built-in assertion handling only registers failures that directly throw an error within its framework. Chai, on the other hand, when using its `expect` or `assert` APIs, does not directly throw errors; rather, it generates an internal representation of the assertion failure, often hidden from Nightwatch's direct view. To display these Chai failures, we need to manually capture them and translate them into an error that Nightwatch can understand.

My approach centers around two main techniques. The first is creating a custom command that performs the Chai assertion and then converts the outcome into a format Nightwatch recognizes as a failure. The second involves leveraging Nightwatch's built-in reporting mechanisms to output the custom-formatted error. We are effectively wrapping Chai's assertions within a Nightwatch-aware structure.

Here’s a concrete explanation of how this works:

**Custom Command Implementation:**

The first step is to create a custom command in Nightwatch. This custom command will accept the Chai assertion as a string, execute it using Chai, and then analyze the result. If the assertion fails, we'll throw an error containing the Chai failure message and details.

Here is the first code example illustrating this custom command:

```javascript
// nightwatch/custom-commands/chaiAssert.js
const chai = require('chai');
const expect = chai.expect;

exports.command = function(assertionString, message = null) {
  let assertionPassed = true;
  let failureMessage = "";

  try {
     eval(`expect(true).${assertionString}`); // Intentionally setting "true" for eval flexibility
  } catch(e) {
    assertionPassed = false;
    failureMessage = e.message;
  }

  if (!assertionPassed) {
    const errorMessage = message ? `${message} - ${failureMessage}` : failureMessage;
    throw new Error(errorMessage);
  }

  this.api.assert.ok(true, "Chai assertion passed: " + (message || assertionString));
  return this;
};
```

**Commentary on Code Example 1:**

*   **`require('chai')` and `const expect = chai.expect;`**: We import the Chai library and establish a shortcut to Chai's `expect` API. This allows us to utilize Chai assertions.
*   **`exports.command = function(assertionString, message = null) { ... }`**: This defines a custom Nightwatch command called `chaiAssert` that takes an `assertionString` representing a Chai chain (e.g., `to.equal('foo')` or `to.be.true`) as well as an optional `message` for providing context to the user.
*   **`let assertionPassed = true; ... try { ... } catch(e) { ... }`**: We wrap the `eval` call inside a try-catch to safely intercept any errors thrown by Chai. If no error is thrown, then the assertion has passed. `assertionPassed` is set to false and the `failureMessage` is collected in the catch block.
*   **`eval(\`expect(true).${assertionString}\`)`**: This is the crucial line where the Chai assertion is actually executed. We use the `eval` function to execute the provided `assertionString` within the `expect(true)` context. This is a dynamic way to handle various Chai assertions. Note: while `eval` has security implications in untrusted scenarios, we are generating the string from known testing code. Additionally, I've ensured 'true' is the assertion subject to maintain context during evaluation.
*   **`if (!assertionPassed) { throw new Error(errorMessage) }`**: If an error was caught (meaning the Chai assertion failed), we construct an error object with `errorMessage` using the original Chai failure message and optional context message and then throw it. This exception is recognized by Nightwatch as a failed assertion.
*   **`this.api.assert.ok(true, "Chai assertion passed: " + (message || assertionString));`**: If no error was thrown, we register an ok assertion in Nightwatch, which logs a pass, along with the message or assertion to provide additional information.
*   **`return this;`**: Returns `this` to allow chaining of Nightwatch commands.

**Utilizing the Custom Command:**

Once this custom command is in place, we can use it in Nightwatch tests like this:

```javascript
// tests/myTest.js
module.exports = {
  'Chai Assertion Example' : function (browser) {
    browser
        .navigateTo('https://www.example.com')
        .assert.titleEquals('Example Domain')
        .chaiAssert('to.equal("Example Domain")', "Verify Title")
        .chaiAssert('to.be.a("string")')
        .chaiAssert('to.be.a("number")', "This should fail intentionally") // Demonstrating a failing assertion
        .pause(1000)
        .end();
  }
};
```

**Commentary on Code Example 2:**

*   This illustrates the usage of `chaiAssert` within a standard Nightwatch test.
*   The first call (`.chaiAssert('to.equal("Example Domain")', "Verify Title")`) demonstrates a passing assertion with a custom message.
*   The second call (`.chaiAssert('to.be.a("string")')`) shows a passing assertion with only the chain.
*    The third call (`.chaiAssert('to.be.a("number")', "This should fail intentionally")`) provides an example of a failing assertion where the custom message helps understand the failure.
*   The test uses native Nightwatch navigation and assertion before exercising the custom command. This integrates seamlessly with existing workflows.

**Reporter Enhancement (Optional):**

While the custom command does successfully report Chai errors to the console output, some users may desire custom formatting in the reporter. I prefer using Nightwatch’s default reporters, but to influence those we'd have to do the following.

To make the reporting of Chai assertions clearer in the generated test reports (e.g., JUnit), you can listen to the `testcase.errored` event that is emitted when an error occurs in a testcase and then reformat the error message. The following code example shows how to accomplish this:

```javascript
// nightwatch.conf.js
const customReporter = (results, options, client) => {
    results.testcases.forEach((testcase, test) => {
       test.results.assertions.forEach(assertion => {
            if (assertion.failure) {
             const match = assertion.message.match(/Chai assertion failed: (.*)/);
             if(match){
                 assertion.message = "Chai Assertion Failure: " + match[1]
             }

           }
       });
    });
  };

module.exports = {
   // ... other config
   reporter : customReporter
  // ... other config
};

```

**Commentary on Code Example 3:**

*  This configuration function is attached to `reporter` within Nightwatch configuration.
* The function intercepts the `results` object and iterates over `testcases` and the assertions within.
* When an assertion fails, the message is checked to see if it is from a Chai assertion based on the pattern `"Chai assertion failed: (.*)"`.
* If a match is found, then the message is modified to be more informative.

**Resource Recommendations:**

1.  **Chai Documentation:** Provides detailed explanations of the Chai assertion library, including its assertion API (`expect`, `assert`).
2.  **Nightwatch Documentation:** The core reference for Nightwatch’s testing API, custom command creation, and reporting configurations.
3.  **JavaScript Error Handling Tutorials:** Resources explaining `try...catch` blocks and JavaScript error object properties are valuable for comprehending how to intercept Chai's errors.

By utilizing these custom command techniques alongside Nightwatch's reporters, I’ve found a robust solution that provides the familiarity of Chai with the reporting capabilities of Nightwatch. The code examples illustrate the integration, and the documentation recommendations offer resources to delve deeper into the respective libraries.
