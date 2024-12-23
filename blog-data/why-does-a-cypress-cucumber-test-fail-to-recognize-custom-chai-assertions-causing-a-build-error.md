---
title: "Why does a Cypress cucumber test fail to recognize custom Chai assertions, causing a build error?"
date: "2024-12-23"
id: "why-does-a-cypress-cucumber-test-fail-to-recognize-custom-chai-assertions-causing-a-build-error"
---

Okay, let's tackle this one. It’s a classic gotcha that’s tripped up many a seasoned developer, myself included. I remember a particularly grueling project a few years back where we were heavily invested in Cypress with cucumber, and this exact issue nearly derailed our entire test suite. The root of the problem, and what's causing your build error, lies in how Cypress handles assertion extensions and the way cucumber integrates within that ecosystem. It's not a straightforward "Cypress versus cucumber" conflict, but rather a nuance in how their lifecycles interact and how Chai, the underlying assertion library, is used.

The core problem centers around timing and scope. Cypress, by default, utilizes a single instance of Chai throughout the test run. When you register a custom Chai assertion, you're typically doing this within a Cypress support file or similar initialization scope, ensuring that the assertion is available to all Cypress tests. However, when cucumber enters the mix, the execution context can become less straightforward, and it’s quite possible that these custom assertions get loaded, or rather, fail to be properly initialized, within the context where cucumber steps are executed. This is the primary reason why you're seeing the "property 'yourCustomAssertion' of undefined" or a similar error message during your build, signifying that the assertion simply isn't available to the steps when the test code executes.

Let's break this down with a few scenarios to make it more concrete. First, consider the following simplified Cypress support file, where we add a custom chai assertion:

```javascript
// cypress/support/commands.js

chai.Assertion.addMethod('isVisibleAndContains', function(text) {
    new chai.Assertion(this._obj).to.be.visible;
    new chai.Assertion(this._obj).to.contain(text);
});

```

In a typical Cypress spec file (without cucumber), this custom assertion would work flawlessly:

```javascript
// cypress/e2e/example.cy.js

describe('Basic test', () => {
  it('should test element visibility and content', () => {
    cy.visit('/some/page');
    cy.get('#element-id').should('isVisibleAndContains', 'Expected text');
  });
});

```

Here, `isVisibleAndContains` is accessible because it is loaded within the Cypress test lifecycle. Now, the problem materializes when we try to utilize it in a cucumber scenario within a step definition file:

```javascript
// cypress/e2e/features/step_definitions/my_steps.js

import { Given, When, Then } from '@badeball/cypress-cucumber-preprocessor';

Then('I should see element with text {string}', (text) => {
  cy.get('#element-id').should('isVisibleAndContains', text); // This is likely to fail
});
```

Here, because the `chai.Assertion.addMethod` might not have been fully executed within the correct scope for the cucumber step definitions, you get the build error. Cypress runs the `supportFile`, but the cucumber steps can be executed out-of-sync. Think of it this way, Cypress initializes Chai in a way that makes it globally available for its own commands, however cucumber steps need to be aware of those updates too. Cucumber uses a separate runtime that needs to understand those custom assertions.

The solution often involves ensuring your custom assertions are initialized not just within the Cypress support file, but are consistently accessible wherever cucumber steps are executing. There isn't a single ‘silver bullet’ approach, but typically I find success with the following methods:

**1. Explicitly requiring/importing support files in step definitions:**

While somewhat of a workaround, you can explicitly require the `commands.js` file within your step definition file. This forces a re-evaluation and, ideally, registration of the custom assertions. Be aware this might cause the support file to be executed multiple times, which can be problematic if you have side effects during the setup of your custom assertions:

```javascript
// cypress/e2e/features/step_definitions/my_steps.js

import { Given, When, Then } from '@badeball/cypress-cucumber-preprocessor';

require('../../../support/commands') // Added this line

Then('I should see element with text {string}', (text) => {
  cy.get('#element-id').should('isVisibleAndContains', text);
});
```

**2. Initializing Assertions Globally using `beforeEach` hook:**

A more robust approach that avoids multiple executions of the support file is to register the assertion within a `beforeEach` hook at the top level of the cucumber step definition file. This ensures that the assertions are available before each scenario:

```javascript
// cypress/e2e/features/step_definitions/my_steps.js

import { Given, When, Then, Before } from '@badeball/cypress-cucumber-preprocessor';

Before(() => {
    chai.Assertion.addMethod('isVisibleAndContains', function(text) {
        new chai.Assertion(this._obj).to.be.visible;
        new chai.Assertion(this._obj).to.contain(text);
    });
});


Then('I should see element with text {string}', (text) => {
  cy.get('#element-id').should('isVisibleAndContains', text);
});

```

**3. Centralized Assertion Definition:**

For larger projects, where custom assertions might be reused in multiple step definitions, I recommend a separate file dedicated solely to defining and initializing all custom Chai assertions, this file is then explicitly loaded in both cypress support and step definitions. This ensures consistency across the entire test suite and prevents duplication.

```javascript
// cypress/support/assertions.js

chai.Assertion.addMethod('isVisibleAndContains', function(text) {
    new chai.Assertion(this._obj).to.be.visible;
    new chai.Assertion(this._obj).to.contain(text);
});

```

Then, within your Cypress support and cucumber step definition files:

```javascript
// cypress/support/commands.js
import './assertions';
```

```javascript
// cypress/e2e/features/step_definitions/my_steps.js
import { Given, When, Then } from '@badeball/cypress-cucumber-preprocessor';
import '../../../support/assertions';

Then('I should see element with text {string}', (text) => {
  cy.get('#element-id').should('isVisibleAndContains', text);
});
```

In my experience, option 3 is generally the most manageable and avoids common pitfalls. You’ll want to choose the method that best suits your project's complexity. Also, remember that the timing can also be affected by other plugins if those are installed. A thorough understanding of plugin dependencies and lifecycles can help you identify other underlying issues.

For further reading, I highly recommend diving deep into the official Cypress documentation, especially the sections on 'custom commands and assertions' as well as exploring the internals of the `cypress-cucumber-preprocessor` library, understanding how it hooks into Cypress lifecycle events. In particular the following resources are invaluable:
 *  Cypress documentation: specifically about custom commands and assertions.
 *  Chai documentation to understand its API and method extension mechanics.
 *  The source code of the `cypress-cucumber-preprocessor` to grasp how it integrates into Cypress execution lifecycle.

Ultimately, this situation isn't a flaw in either Cypress or cucumber but rather a subtle interaction detail. By understanding the underlying mechanics, you can avoid the frustration and build robust, maintainable tests. It takes a bit of digging, but getting it right early on will save you considerable time down the line.
