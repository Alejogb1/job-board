---
title: "How can I use Chai's `expect` in Nightwatch.js tests?"
date: "2025-01-30"
id: "how-can-i-use-chais-expect-in-nightwatchjs"
---
Chai, with its expressive assertion library, integrates smoothly into Nightwatch.js tests, enhancing test readability and developer productivity. I've personally found it particularly useful in complex asynchronous scenarios where detailed error messages are crucial. While Nightwatch provides its own assertions, Chai's fluent interface, specifically `expect`, offers more flexible and nuanced validations. This response will outline how to incorporate and utilize Chai's `expect` assertions within a Nightwatch.js test suite.

**Integration Strategy**

Nightwatch.js provides a `client` object, an instance of the Selenium WebDriver, which can be enhanced with custom commands and assertions. Chai is not inherently part of Nightwatch.js and requires manual integration. The key lies in accessing the `expect` function provided by Chai, usually imported directly or made available through a global setup. The standard practice is to make `expect` accessible throughout tests by including Chai in a global setup file loaded prior to any tests executing.

Within your Nightwatch tests, you can use `expect` to perform assertions on any data available within the scope of a test, including the result of browser interactions, JSON responses from APIs, or computed values within your tests. It's worth noting that Chai's `expect` differs from Nightwatch's built-in assertions, which operate directly on the WebDriver's state and elements. The `expect` assertions focus on the evaluated value within your code.

**Implementation**

Let's consider a scenario where I have a web application with a login form.  I want to verify a successful login. First, I need to ensure Chai is integrated properly with Nightwatch.js. Assume I have a `nightwatch.conf.js` file that handles the configuration, and it includes a global setup hook like this:

```javascript
// nightwatch.conf.js

module.exports = {
  //... other configuration
  test_settings: {
    default: {
      globals: {
        before: function (browser) {
            const chai = require('chai');
            global.expect = chai.expect;
        }
       },
    },
  }
  //... other configurations
};
```

This setup ensures that the `expect` function is accessible to all tests. Now let’s proceed with a test implementation.

**Code Example 1: Simple Element Text Verification**

This example demonstrates how I might use `expect` to assert that the text of a specific element matches the expected value after a user interaction:

```javascript
//login_test.js

module.exports = {
  'Login Success Test': function(browser) {
    browser
      .url('https://example.com/login')
      .setValue('#username', 'testuser')
      .setValue('#password', 'testpass')
      .click('#loginButton')
      .waitForElementVisible('#welcomeMessage', 5000, 'Welcome message is visible.')
      .getText('#welcomeMessage', function(result) {
            expect(result.value).to.equal('Welcome, testuser!');
      })
      .end();
  }
};
```

**Commentary:**

In this test, I first navigate to the login page, enter credentials, and click the login button. After waiting for the welcome message to appear, I then use `browser.getText` to extract the text content of the `#welcomeMessage` element. The callback function for `getText` receives a `result` object containing the retrieved text within the `result.value` property. I then use Chai's `expect` to assert that this value precisely matches 'Welcome, testuser!'.  The  `.to.equal()` method is the Chai's method which is used for equality comparisons.  The crucial aspect is the asynchronous nature of `browser.getText`, highlighting why utilizing a callback with `expect` is essential within the Nightwatch.js testing flow.  This keeps the test execution synchronous with the Nightwatch chain.

**Code Example 2: Asynchronous API Call Verification**

My experience also includes scenarios where a user action triggers an API call, and I need to assert based on the response data.  This example shows how to use `expect` to check the content of a JSON response:

```javascript
// api_test.js
module.exports = {
  'API Response Check': function(browser) {
    browser
      .url('https://example.com/profile')
      .click('#fetchProfileButton')
      .waitForElementVisible('#profileDataContainer', 5000)
      .api('GET', '/api/profile', function(response){
            expect(response.status).to.equal(200);
            expect(response.data).to.be.an('object');
            expect(response.data).to.have.property('username').that.is.a('string');
             expect(response.data.username).to.equal("testuser");
       })
      .end();
  }
};

```

**Commentary:**

Here, after triggering an API call, I'm using Nightwatch’s `api` custom command to perform a GET request to the specified `/api/profile` endpoint. The callback receives the response, which includes HTTP status code and the response body. With `expect`, I can chain assertions to verify: first that the request returned a status code of 200, then that the `response.data` is an object, and specifically that the `response.data` contains a string valued property called 'username'. Finally, I assert that the value of the username property is 'testuser'. This demonstrates `expect`'s capability to perform intricate verifications on complex data structures obtained via API calls, which is crucial for testing integrations. `expect`'s method chaining capabilities allow for a compact but readable assertion style.

**Code Example 3: Checking for Array Elements**

This test illustrates how to use `expect` to verify that an array includes expected elements. This is useful for checking the results of dynamic UI lists or data collections:

```javascript
//list_test.js

module.exports = {
  'List Elements Check': function(browser) {
     browser
       .url('https://example.com/items')
       .elements('css selector', '#itemList li', function(result) {
           const listItems = result.value;
           const texts = [];
           let completedCount = 0;
           function getTextForElement(index){
             if (index < listItems.length) {
               browser.elementIdText(listItems[index].ELEMENT, (textResult)=>{
                   texts.push(textResult.value)
                   completedCount++;
                   if (completedCount === listItems.length) {
                     expect(texts).to.include.members(["Item 1","Item 2","Item 3"]);
                     expect(texts).to.have.lengthOf(3)
                   }
                   getTextForElement(index + 1);
               });
             }

           }

           getTextForElement(0);


       })

     .end();
   }
 };

```

**Commentary:**

In this test case, I'm fetching a list of list items using `browser.elements`. Due to the asynchronous nature of extracting text for each element, I've implemented a recursive function called `getTextForElement` to handle this. The element IDs retrieved in the first step are then passed to `browser.elementIdText` to extract the text. Once all the items' text has been extracted, I use `expect` to verify that this array of text values contains "Item 1", "Item 2", and "Item 3", and also that the number of items is exactly 3. This approach highlights the ability of `expect` to operate on collections and complex data processing, all while still adhering to the asynchronous execution flow of Nightwatch.js. The test code also demonstrates that you can create custom functions to manage asynchronous processing of the extracted data before making assertions using `expect`.

**Resource Recommendations**

To improve your proficiency with using Chai's `expect` within Nightwatch.js tests, I recommend consulting the official documentation for both Chai and Nightwatch.js. These documents will provide in-depth information about available assertion styles, syntax, and commands. Exploring examples within the Nightwatch.js GitHub repository and community forums can offer valuable insights. There are also many good articles that explain the usage of Chai assertions with Javascript unit testing.
