---
title: "Why am I seeing repeated '859: unexpected token' exceptions in a Rails 6 application?"
date: "2024-12-23"
id: "why-am-i-seeing-repeated-859-unexpected-token-exceptions-in-a-rails-6-application"
---

Let's unpack this '859: unexpected token' error you're encountering in your Rails 6 application. It's a frustrating one, I know. I’ve spent my fair share of late nights staring at similar error logs, especially back in the early days of Rails 6 adoption. This error, specifically, usually isn't about an outright syntax error in your code, but rather a parsing issue arising from how webpacker is processing your javascript assets. And it's almost always about the interaction between webpacker's configuration and how you are trying to incorporate javascript code that webpacker finds unsuitable as an input module.

Essentially, webpacker, which Rails 6 uses by default for asset packaging, expects to work with javascript files that are formatted as valid ES modules or CommonJS modules. The '859: unexpected token' error indicates that the parser, often Babel, encountered something in your javascript file that doesn’t conform to those expectations. This could be due to a variety of reasons, but I’ve found a few specific culprits to be most common.

The most frequent cause I've encountered is the presence of "raw" javascript, often legacy scripts or plugins that aren’t explicitly wrapped in a module. Webpacker needs these files to be, at least, module-like. Let's say you had some inline javascript in a partial that you forgot to move to an asset file, or perhaps a third-party javascript you're using. In the past, I was pulling in a carousel library and completely forgot to import it correctly. Instead, I was trying to load it using a script tag directly in a layout file. That triggered this error, and I had to move the script to `app/javascript/packs`, properly import it into `application.js`, and remove the direct inclusion.

Another fairly common situation is mismatched or misconfigured Babel presets. If your application is trying to leverage newer ES features that aren't properly transpiled by your Babel configuration, or if Babel is entirely missing from your webpack configuration, you can run into this exact issue. It is important to have the `babel-loader` appropriately configured to process your javascript in your `webpack.config.js` file, which often lives in your `config/webpack` directory. Without proper presets, the parser won't know how to process modern javascript syntax.

Finally, and this is rarer but happened to me once, incorrect file extensions or import statements can trigger this. If webpacker is trying to parse a file as javascript but it's not, or you have incorrect paths in your import/require statements, the parser can fail. I had an instance where I renamed a component but forgot to update a related import in another file. That minor oversight threw that precise '859' error, highlighting how fussy webpacker can be about file dependencies.

Now, let's get into some code examples to make this clearer:

**Example 1: The "Raw" Javascript Issue**

Let’s assume you have a snippet of legacy javascript like this somewhere:

```javascript
// this is some legacy script in a partial
(function(){
  var globalVar = "hello";
  function legacyFunc() {
    console.log(globalVar);
  }
  legacyFunc();
})();
```

This, if included directly, perhaps within a `<script>` tag, would confuse webpacker. This is *not* a module, and thus webpacker doesn't know how to handle it directly. You should convert this into a module. Here’s what you might do to integrate it into your `application.js` pack:

First, move this code into `app/javascript/packs/legacy_script.js` and change the code into:
```javascript
// app/javascript/packs/legacy_script.js
const globalVar = "hello";

function legacyFunc() {
  console.log(globalVar);
}

export { legacyFunc };
```

Then import and use it in `app/javascript/packs/application.js`:
```javascript
// app/javascript/packs/application.js
import { legacyFunc } from './legacy_script';

document.addEventListener('DOMContentLoaded', () => {
  legacyFunc();
});
```

This ensures that your javascript code is properly packaged by webpacker and does not trigger the dreaded error.

**Example 2: Babel Configuration Issues**

Suppose your webpack config (`config/webpack/environment.js`) is missing a crucial piece. Specifically, you’re not using the `@babel/preset-env` which is a common preset that specifies which javascript transformations should be applied. A minimal, correctly configured webpack config might look like this (although specific loaders/plugins may be different in your own app):

```javascript
// config/webpack/environment.js
const { environment } = require('@rails/webpacker')

environment.loaders.append('babel', {
  test: /\.(js|jsx|mjs|cjs)?(\.erb)?$/,
  exclude: /node_modules/,
  use: {
    loader: 'babel-loader',
    options: {
      presets: [
        ['@babel/preset-env', {
            targets: { browsers: "last 2 versions, not ie <= 10" },
            useBuiltIns: "usage",
            corejs: 3
        }]
        ]
    }
  }
})

module.exports = environment
```

Without the correct Babel presets, webpacker might incorrectly parse more recent syntax causing errors. In the example, I'm explicitly using `@babel/preset-env` along with `useBuiltIns` and `corejs` to handle polyfills.

**Example 3: Incorrect Import/Require Paths**

Imagine you have a React component in `app/javascript/components/MyComponent.jsx` and it uses other modules. An incorrect import path might look like this:

```javascript
// app/javascript/components/MyComponent.jsx
import { someFunc } from '../utils/someutil'; // Note the lowercase path

function MyComponent() {
  // ...component logic
}

export default MyComponent;
```

And then perhaps `someutil.js` lives under `app/javascript/utils/SomeUtil.js` with a capital ‘S’ in the filename. The miscase in the import, while sometimes ok on case-insensitive filesystems locally, can cause issues in deployment. This issue could also be due to forgetting the `.js` extension which may be implicitly added locally, but must be specified on most deployed environments.

The correct import should be:

```javascript
// app/javascript/components/MyComponent.jsx
import { someFunc } from '../utils/SomeUtil.js'; // Corrected path

function MyComponent() {
  // ...component logic
}

export default MyComponent;
```

Ensuring paths are precise and extensions are present is vital for avoiding this parsing issue.

So, what to do? I'd suggest first checking for directly embedded javascript within your views or layouts, then closely examine your webpack and babel configurations. Review the exact file referenced in the error message, especially focusing on any legacy or third-party javascript. Verify that `babel-loader` is properly configured and is utilizing the appropriate presets. Finally, check for any misnamed files or incorrect paths within import/require statements.

For further reading, I highly recommend *Webpack: Concepts and Techniques* by James Nelson, as it gives a comprehensive explanation of Webpack's internals and its configuration. Another invaluable resource is the official *Babel Handbook*, which provides a deep understanding of Babel's capabilities and how to configure it effectively. Additionally, the webpacker documentation, especially the parts regarding javascript, loaders, and the default configuration, should provide more specific information for your Rails 6 setup.

Debugging these types of issues often requires a methodical approach, checking each point of potential failure one by one. Don’t hesitate to use `console.log` or debugger statements liberally, as these tools are still your friends, even when dealing with complex asset processing pipelines. You'll get to the bottom of it.
