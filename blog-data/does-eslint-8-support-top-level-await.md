---
title: "Does ESLint 8 support top-level await?"
date: "2024-12-23"
id: "does-eslint-8-support-top-level-await"
---

Let's tackle this directly. Whether eslint version 8 supports top-level await is not a simple yes or no. It’s more nuanced and depends significantly on how you’re configuring your eslint setup, particularly with the parser options. I've seen this trip up many teams, including one where we spent a good part of an afternoon troubleshooting build failures that boiled down to this very issue.

Here's the breakdown: eslint itself doesn't directly interpret or execute JavaScript code. It relies on a parser to first transform your code into an Abstract Syntax Tree (AST), which it then analyzes. The parser most commonly used for JavaScript projects is `@babel/eslint-parser` or `@typescript-eslint/parser` (if you are working on a TypeScript project). These parsers are the components that need to be aware of the specific syntax, including top-level await.

ESLint 8, by itself, is agnostic to this. Think of eslint as the engine of your linting process. It only knows what its parser tells it. To illustrate:

* **Default behavior:** If you just install eslint 8 and run it without specifying a parser, it likely won't recognize top-level await. You'll typically get a syntax error because the default parser assumes a more restricted set of JavaScript features.
* **With an appropriate parser:** When you configure eslint to use a parser that understands top-level await, it will function correctly without error. The parser does the heavy lifting of understanding the feature.

Top-level await, as you likely know, was introduced into javascript modules to allow module-level code to use promises. This is incredibly handy for initializing modules that depend on asynchronous operations, such as loading configuration, making API calls etc. To avoid unnecessary confusion, ensure you are using nodejs environments that support top-level await features.

Now, let's delve into practical implementations and configuration to see how this manifests. I'll show you three working examples, one each for javascript without any transpilation needed, javascript using babel, and for a typescript project, outlining parser configuration for each.

**Example 1: Vanilla Javascript with ESM Modules and no transpilation required**

Let's say we have `index.js` that requires an async operation on start up:

```javascript
// index.js
import { fetchData } from './dataFetcher.js';

console.log("Starting");

const data = await fetchData();

console.log("Data:", data);

```
and `dataFetcher.js` containing the async operation

```javascript
// dataFetcher.js
export async function fetchData() {
  await new Promise(resolve => setTimeout(resolve, 500));
  return {message: "Data fetched successfully"};
}

```

Here’s the essential part: if you don’t have a `.eslintrc.cjs` configuration, or if it doesn’t define a parser, and you try to lint `index.js` using basic eslint 8, it will likely fail.

Here’s a basic `eslint.config.js` setup that *will* allow this to pass (for eslint 8 and upwards, using the new flat config file):

```javascript
// eslint.config.js
import js from '@eslint/js';

export default [
  js.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module', // Required for modules
    },
    rules: {
      // Additional rules can go here
    }
  },
];
```
In this simple case, we're relying on the built-in parser from eslint. Using `ecmaVersion: 'latest'` effectively tells it we're using the most recent javascript standard, which should enable top-level await to be parsed correctly without error. Because we're writing ESM modules, we have to include `sourceType: 'module'` to get the correct behavior.

**Example 2: Javascript using Babel (Common Setup)**

Most of the time, especially with modern javascript development, you’ll likely be using babel for transpilation. Babel itself can handle top-level await, but it also plays a crucial role in the parsing context with eslint.

Suppose we are now using babel and need to use babel's parser. Our code remains similar to example 1, with the same `index.js` and `dataFetcher.js`. In this situation, we need to add `@babel/eslint-parser` as an eslint dependency, and include this in our configuration file.

Here’s how your `eslint.config.js` might look:

```javascript
// eslint.config.js
import js from '@eslint/js';
import babelParser from '@babel/eslint-parser';

export default [
  js.configs.recommended,
  {
    languageOptions: {
      parser: babelParser,
      sourceType: 'module', // Required for modules
      parserOptions: {
          requireConfigFile: false, // Disable looking for babel config as we are using it through eslint
          babelOptions: {
             plugins: [ ], // Enable any babel plugins as needed here
             presets: [  '@babel/preset-env' ],
            },
      }
    },
    rules: {
        // rules can go here
    }
  },
];

```
Notice we have added the babel parser in `languageOptions.parser`. Additionally, we need to tell babel what settings to use under the `parserOptions.babelOptions`.

**Example 3: TypeScript and @typescript-eslint/parser**

In a TypeScript project, `@typescript-eslint/parser` is essential.  Let’s assume we have a similar `index.ts` file:

```typescript
// index.ts
import { fetchData } from './dataFetcher';

console.log("Starting");

const data = await fetchData();

console.log("Data:", data);
```
and `dataFetcher.ts`
```typescript
// dataFetcher.ts
export async function fetchData(): Promise<{message: string}> {
  await new Promise(resolve => setTimeout(resolve, 500));
  return {message: "Data fetched successfully"};
}
```
Our `eslint.config.js` file needs to explicitly use the typescript parser. Here’s a possible configuration:

```javascript
// eslint.config.js
import js from '@eslint/js';
import typescriptParser from '@typescript-eslint/parser';
import tsRecommended from '@typescript-eslint/eslint-plugin/dist/configs/recommended';

export default [
  js.configs.recommended,
  ...tsRecommended,
  {
    languageOptions: {
        parser: typescriptParser,
        parserOptions: {
            project: true,
            tsconfigRootDir: './',
            sourceType: 'module',
      },
      ecmaVersion: 'latest', // optional as the parser understands ts syntax
    },
    rules: {
     //rules can go here
    }
  },
];
```

Here we are adding the `typescriptParser` and then specifying options to configure the parser, including referencing `tsconfigRootDir` so the parser can use your project's typescript compiler options. In this example we use the `recommended` typescript settings from `@typescript-eslint/eslint-plugin`.

**Key takeaway:** eslint 8 supports top-level await *through its parser*, not directly. The key is choosing the appropriate parser and configuring it correctly for your project type (vanilla js with no transpilation, javascript with babel or typescript) and setting the correct options in the eslint config file.

**Further Reading:**

To gain a more comprehensive understanding of these tools, I recommend these resources:

*   **The official eslint documentation:** It's detailed and the best place to get all the info on configurations: [eslint.org](https://eslint.org/docs/latest/)
*   **Babel’s documentation:** Specifically, familiarize yourself with the `@babel/eslint-parser` section: [babeljs.io](https://babeljs.io/docs/)
*   **Typescript eslint plugin:**  Refer to the documentation for `@typescript-eslint/parser` and the related plugins: [typescript-eslint.io](https://typescript-eslint.io/docs/)

By making sure you configure the parser correctly, you'll be able to take full advantage of top-level await in your project and avoid unexpected build errors and wasted debugging time. This wasn’t an issue with the team, now it is an established part of the toolchain. Good luck, and happy coding.
