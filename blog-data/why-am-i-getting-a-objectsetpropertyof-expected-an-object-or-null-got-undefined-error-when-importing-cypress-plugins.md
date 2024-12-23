---
title: "Why am I getting a 'Object.setPropertyOf: expected an object or null, got undefined' error when importing Cypress plugins?"
date: "2024-12-23"
id: "why-am-i-getting-a-objectsetpropertyof-expected-an-object-or-null-got-undefined-error-when-importing-cypress-plugins"
---

Let’s address this peculiar "Object.setPrototypeOf: expected an object or null, got undefined" error that sometimes crops up when importing Cypress plugins; I’ve definitely spent a few late nights chasing this one myself. It’s frustrating, but the root cause often lies in a subtle interplay of how Cypress, javascript's prototype system, and module loading interact. It's less about any single incorrect line of code in the plugin itself and more about the environment and process around the plugin's setup.

In my experience, this error usually points to a problem with the order of module loading or with a plugin incorrectly attempting to extend a prototype before its target class is fully initialized, resulting in ‘undefined’ being passed to `Object.setPrototypeOf` where an object or null is expected. Let's break down the typical scenarios and how I've resolved them.

The error, as stated, stems from Javascript's prototype inheritance mechanism; `Object.setPrototypeOf()` is a low-level function that attempts to change the prototype of an object. When you see that the argument was ‘undefined,’ it means the code attempted to set an object’s prototype to nothing. This typically happens during initialisation or early import stages, suggesting an ordering problem. Specifically within cypress, it's often triggered by the plugin's execution occurring too early, before the cypress environment or core objects are ready.

There are mainly three patterns that I've found most frequent in my projects over the years:
1.  **Incorrect Plugin Registration**: The most straightforward cause is that your plugin might be incorrectly attempting to set a prototype on an object *before* that object has been defined or initialized by Cypress itself. This could be because the `module.exports` of your plugin configuration is executing before Cypress calls `pluginsFile()` or some other initialization function, meaning your code runs too soon.
2. **Circular Dependencies**: More subtle are circular dependencies. If plugin A tries to import plugin B, and plugin B in turn requires elements from cypress through plugin A's scope, a deadlock is created with modules being evaluated out of order and resulting in a partial initialization of classes and potentially undefined values being passed to `setPrototypeOf`.
3. **Transpilation Issues**: Although less common now with modern bundlers, incorrect or missing transpilations in your plugin's source code could result in JavaScript syntax that older engines, within Cypress's browser environment, cannot process correctly, leading to these errors. This usually manifests as a failure to properly construct a class hierarchy with prototypes.

Now, let's dive into practical examples and how I've solved them:

**Example 1: Incorrect Plugin Registration Order**

Imagine a scenario where your plugin attempts to extend the `Cypress.Commands` object directly during module evaluation:

```javascript
// plugins/my-plugin.js (bad example)
import { addCommand } from "./my-custom-commands";

addCommand(); // Calling the command creation here will trigger errors

module.exports = (on, config) => {
    // on and config have values now
  // The rest of the plugin setup can go here
};

```

```javascript
// plugins/my-custom-commands.js
export function addCommand() {
    Cypress.Commands.add('myCustomCommand', () => {
      // some custom command functionality
      cy.log("Custom Command run")
    });
}
```

The problem here is that `Cypress.Commands` is not fully initialized until *inside* the `module.exports` function, when the `on` and `config` parameters have values. When the code is run, it attempts to add a command to an 'undefined' prototype of the cypress object.

**Solution 1: Modify Plugin Registration**

The correct approach is to move the command registration into the `module.exports` function, or into a function called by it:

```javascript
// plugins/my-plugin.js (good example)
import { addCommand } from "./my-custom-commands";

module.exports = (on, config) => {
  addCommand();
  // The rest of the plugin setup can go here
};
```

```javascript
// plugins/my-custom-commands.js
export function addCommand() {
    Cypress.Commands.add('myCustomCommand', () => {
      // some custom command functionality
      cy.log("Custom Command run")
    });
}
```

By doing this, we ensure the `Cypress.Commands` object is available before trying to extend it. This solved several issues I've encountered with plugin configuration inconsistencies.

**Example 2: Circular Dependency Problems**

Let’s consider a scenario where two custom plugins depend on each other in a circular fashion:

```javascript
// plugins/plugin-a.js (bad example)
import { pluginBFunction } from "./plugin-b";

module.exports = (on, config) => {
  pluginBFunction();

  Cypress.Commands.add('pluginACommand', () => {
      cy.log("plugin A command has run");
  });
};
```

```javascript
// plugins/plugin-b.js (bad example)
import { pluginACommand } from './plugin-a';

export function pluginBFunction() {
    Cypress.Commands.add("pluginBCommand", () => {
        cy.log("plugin B command has run");
    });
};
```

Here, `plugin-a` imports from `plugin-b`, and `plugin-b` attempts to call `Cypress.Commands`, potentially resulting in it not being ready depending on how the module loader schedules execution, leading to the `Object.setPrototypeOf` error. This kind of interaction was a persistent pain point until I identified the pattern.

**Solution 2: Refactor and Decouple**

The solution is to decouple the plugins by extracting shared functionality into a separate module, breaking the circular dependency. The plugins should interact solely via the Cypress API rather than directly importing functionality from each other's plugins. A simple approach to this is removing circular calls, the code should now become:

```javascript
// plugins/plugin-a.js (good example)

module.exports = (on, config) => {

    Cypress.Commands.add('pluginACommand', () => {
        cy.log("plugin A command has run");
    });
  // The rest of the plugin setup can go here
};
```

```javascript
// plugins/plugin-b.js (good example)
export function pluginBFunction() {
    Cypress.Commands.add("pluginBCommand", () => {
        cy.log("plugin B command has run");
    });
};

module.exports = (on, config) => {
    pluginBFunction();
}

```

Now the two plugins no longer depend directly on each other. `pluginB` no longer imports from `pluginA`, preventing the circular dependency and the resulting errors related to initialisation order.

**Example 3: Transpilation or Bundling Issues**

Finally, if you are using custom builds or bundling systems for your plugins, it’s worth double-checking the transpilation of class extensions:

```javascript
// plugins/my-extended-command.js (bad example - assumed incorrect setup)

class BaseClass {
    constructor() {
    }
}

class ExtendedClass extends BaseClass {
    constructor() {
        super();
    }
}

module.exports = (on, config) => {
  Cypress.Commands.add('extendedClass', () => {
      const instance = new ExtendedClass();
  });
};
```

If the plugin's code is not correctly transpiled or bundled for older Javascript engines supported by the browsers cypress executes in, it can sometimes fail to correctly set up the prototypes or classes resulting in an error during class instantiation.

**Solution 3: Correct Transpilation Setup**

Ensure that your build process uses a proper Babel configuration or similar that targets a set of browsers compatible with Cypress's testing environment. Refer to the Cypress documentation for supported browsers and adjust the target compilation settings accordingly.

To understand these deeper concepts, I’d recommend resources like:

*   **"You Don't Know JS" by Kyle Simpson**: It’s an excellent deep dive into Javascript, especially the *this* keyword and prototypes, which are crucial for understanding these errors. Specifically, look into his explanations on prototypal inheritance.
*   **"Effective JavaScript" by David Herman**: This book offers practical ways to write better Javascript, with a dedicated section on the best practices of using prototypes.
*   The **EcmaScript specification (ECMA-262)**: If you need *the* reference material, the spec explains the detailed behavior of every Javascript concept.
*   **The official Cypress documentation itself**: It's surprisingly in-depth in explaining the plugin lifecycle and how the core objects are made available to plugins. Pay attention to the specifics of `pluginsFile` function execution order.

In conclusion, the "Object.setPrototypeOf: expected an object or null, got undefined" error while working with Cypress plugins is typically related to improper plugin initialization sequence, circular dependencies between plugins, or a fault in transpilation of plugin code during bundling. Always remember to register Cypress commands within `module.exports` to ensure the Cypress core API is initialized and available, keep your plugin dependencies clean, and double check your build setup if it is not a simple JS file. By systematically addressing these common pitfalls, you’ll avoid those frustrating ‘undefined’ errors and keep your Cypress test suites running smoothly. These steps, borne from frustrating debugging sessions, have consistently led me to resolution, and hopefully will aid you as well.
