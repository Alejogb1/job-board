---
title: "How can I import JavaScript files using esbuild and jsbundling-rails?"
date: "2024-12-23"
id: "how-can-i-import-javascript-files-using-esbuild-and-jsbundling-rails"
---

Alright, let's unpack this. I’ve spent considerable time navigating the intricacies of asset pipelines, and integrating esbuild with rails’ jsbundling-rails gem is a particular area I've repeatedly encountered. It's not always as straightforward as the initial setup guides suggest, especially when your project grows and you start modularizing your javascript. You're asking how to import javascript files, which sounds simple but depends heavily on your project structure and how you're choosing to leverage esbuild's capabilities.

Let's start with the assumption that you've successfully installed the `jsbundling-rails` gem, have esbuild configured within your rails environment, and your basic application.js is working. The key to effective import is understanding esbuild’s module resolution and how it interacts with rails’ asset paths. One common misconception, especially coming from other asset pipelines like webpacker, is that you can directly reference files based on rails asset paths. Esbuild doesn’t operate directly on the rails asset pipeline as such; instead, it treats your javascript source directory as the root for module resolution unless you tell it otherwise.

Think of it this way: with esbuild, the entry point you define in your `esbuild.config.js` (often `app/javascript/application.js` by default) is where it starts looking for imports. If you’re trying to import a file, esbuild will, by default, expect to find it relative to this entry point or its subdirectories, or through the node module resolution system if you are importing a library installed with npm or yarn.

Now, let's delve into some specific scenarios and how to handle imports using esbuild and jsbundling-rails:

**Scenario 1: Importing files within your `app/javascript` directory.**

This is the most basic case. Let's say you have this structure:

```
app/javascript
├── application.js
└── components
    └── button.js
```

And your `button.js` file contains:

```javascript
export function createButton(text) {
  const button = document.createElement('button');
  button.textContent = text;
  return button;
}
```

To import `createButton` into your `application.js`, you would simply use a relative import like this:

```javascript
import { createButton } from './components/button';

document.addEventListener('DOMContentLoaded', function() {
  const myButton = createButton('Click Me!');
  document.body.appendChild(myButton);
});
```

This is straight-forward and works out-of-the-box as the path is relative to the location of `application.js`. This example demonstrates the most fundamental concept: relative paths within the javascript source directory.

**Scenario 2: Importing files in a directory that's outside of `app/javascript` but still within your app's folder structure.**

This is where it can get a little more involved. Assume you have a folder named `lib` in your app's root, and it contains some utilities.

```
app/
├── javascript
│   └── application.js
└── lib
    └── utils
        └── string_helpers.js
```

And your `string_helpers.js` contains:

```javascript
export function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}
```

Trying to import this with a relative path like `../../lib/utils/string_helpers.js` from `application.js` is possible but is not a good idea. Instead, it's much cleaner, and often required, to utilize the `resolve` feature within esbuild's config.

While `jsbundling-rails` provides a good foundation, you often need to further configure your `esbuild.config.js` to manage this effectively. Here's an example:

```javascript
// esbuild.config.js
const path = require('path');

module.exports = {
  entryPoints: ['app/javascript/application.js'],
  bundle: true,
  outdir: 'app/assets/builds',
  resolve: {
    alias: {
      '@lib': path.resolve(__dirname, 'lib'),
    },
  },
};

```

Here, we’re creating an alias called `@lib` that points to the `lib` directory, effectively creating a virtual path shortcut. Now, in your `application.js` you can do this:

```javascript
import { capitalize } from '@lib/utils/string_helpers';

document.addEventListener('DOMContentLoaded', function() {
  const text = "hello world";
  const capitalizedText = capitalize(text);
  console.log(capitalizedText); // Output: Hello world
});
```

This approach is far more maintainable, particularly when the folder structures become more deeply nested. It decouples the location of our `lib` files from the import statements. The key here is the `resolve.alias` property which enables path aliasing.

**Scenario 3: Importing node modules and how they interact with your own code.**

Assuming you've installed something like `lodash` using npm or yarn and intend to use it in your code, you don't have to do anything special beyond installing it. Esbuild's default configuration handles this via its node module resolution algorithm. The imports for node modules, then, are naturally different as they are identified based on the name of the installed module.

For example, in your `application.js`:

```javascript
import _ from 'lodash';
import { createButton } from './components/button';

document.addEventListener('DOMContentLoaded', function() {
  const numbers = [1, 2, 3, 4, 5];
  const squaredNumbers = _.map(numbers, (n) => n * n);
  console.log(squaredNumbers); // output: [1, 4, 9, 16, 25]

  const myButton = createButton('Click Me!');
  document.body.appendChild(myButton);
});
```

Here, we're directly importing the lodash library (`import _ from 'lodash';`). Esbuild, in this context, will look in your node_modules directory, located next to your package.json. This highlights esbuild's capability to seamlessly integrate with standard npm/yarn workflows.

For further reading and a deeper dive into the topics I've covered, I would suggest exploring:

* **esbuild’s official documentation**: It provides comprehensive details on module resolution, configuration options, and other advanced features. (Specifically look at the 'resolve' configuration options.)
* **"Node.js Design Patterns" by Mario Casciaro and Luciano Mammino**: It covers patterns for module structuring, and how different project setups commonly organize file structures. Though not specific to esbuild, it's valuable in structuring your application.
* **"Effective JavaScript" by David Herman**: While it is focused on JavaScript itself, it provides a solid understanding of JavaScript module patterns, which underpin how esbuild interprets and processes your code.
* **The official documentation for jsbundling-rails**: It contains crucial information on how it interacts with esbuild, specifically on the configuration conventions.

In conclusion, while importing in esbuild and jsbundling-rails might seem simple on the surface, mastering it relies on understanding esbuild's module resolution process and configuring it correctly, often via the `resolve` configuration. Pay close attention to the use of relative paths, path aliasing, and the node module resolution system. By systematically approaching this, you will significantly improve the modularity, organization and maintainability of your javascript code.
