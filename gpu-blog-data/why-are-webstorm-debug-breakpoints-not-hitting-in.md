---
title: "Why are WebStorm debug breakpoints not hitting in my Node.js project?"
date: "2025-01-30"
id: "why-are-webstorm-debug-breakpoints-not-hitting-in"
---
Node.js debugging with WebStorm, while powerful, can be frustrating when breakpoints are ignored. From experience troubleshooting similar issues on projects ranging from simple API endpoints to complex event-driven systems, I've found that breakpoint failures typically stem from a mismatch between the debugger's understanding of the application's execution context and the actual runtime environment. This disconnect can manifest in several ways, demanding careful examination of the configuration, code, and runtime behavior.

The most common cause I've encountered is a discrepancy in the source map handling. When Node.js code is transpiled from languages like TypeScript, or processed with tools that alter the file structure, the debugger relies on source maps to correlate the compiled JavaScript with the original source code. If these source maps are missing, misconfigured, or out of sync, WebStorm will struggle to align breakpoints set in your original source with the actual execution points in the runtime. The debugger needs an accurate map to understand the logical relation between your files and the interpreted JavaScript it’s inspecting.

Another prevalent issue arises from incorrect debug configurations. Specifically, the “Attach to Node.js/Chrome” and “Node.js” run/debug configurations in WebStorm handle debugging in different ways. “Attach to Node.js/Chrome” is designed to attach to a running Node.js process, typically one started externally to WebStorm, while “Node.js” starts the Node.js process from within the IDE. Choosing the wrong configuration for your setup, for example, trying to attach when you haven’t already started a Node.js process with debugging enabled, will lead to breakpoints being skipped.

Additionally, ensure your project's entry point specified in the run/debug configuration is correct. The entry point, or main module, is where Node.js begins execution. An incorrectly set entry point can mean the debugger is not watching the correct files or even the correct process, leading to breakpoints not being registered in the runtime. This is easily overlooked, especially when working with monorepos or unconventional project structures.

Finally, asynchronous operations in Node.js can also obscure where exactly breakpoints should be hit. While this doesn't necessarily lead to the debugger *skipping* breakpoints entirely, they may be encountered at unexpected times due to the nature of the event loop. If you're setting breakpoints inside async functions or callbacks, understanding the asynchronous flow is critical for placing them effectively. Misinterpreting asynchronous timing can lead to the impression of skipped breakpoints when the code execution is merely delayed.

Let's explore some specific code scenarios with examples:

**Example 1: Source Map Issue**

Imagine a simple TypeScript project with a `src/index.ts` file that is transpiled to `dist/index.js`.

```typescript
// src/index.ts
export function add(a: number, b: number): number {
  console.log("Adding numbers");
  return a + b;
}

const result = add(5, 3);
console.log("Result:", result);

```

The accompanying `tsconfig.json` might include the following output configuration:

```json
{
    "compilerOptions": {
        "target": "es6",
        "module": "commonjs",
        "outDir": "./dist",
        "sourceMap": true,
        "rootDir": "./src",
        "strict": true
    },
    "include": ["src/**/*"],
    "exclude": ["node_modules"]
}
```

Here, crucial for correct debugging is `"sourceMap": true`. Without it, the debugger would have no knowledge of the correspondence between `src/index.ts` and `dist/index.js`. If you were to set a breakpoint inside the `add` function in `src/index.ts`, and source maps were absent or improperly configured, WebStorm would fail to map this breakpoint to the corresponding line in the compiled JavaScript and, therefore, the breakpoint would not be hit. You’d need to ensure the `outDir` property matches where you are placing the compiled JavaScript and that the `"rootDir"` correctly specifies the root of your source files. After transpilation, the generated Javascript with associated map might look like:

```javascript
// dist/index.js
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.add = void 0;
function add(a, b) {
    console.log("Adding numbers");
    return a + b;
}
exports.add = add;
var result = add(5, 3);
console.log("Result:", result);
//# sourceMappingURL=index.js.map
```

The `index.js.map` file contains the crucial mapping for debugging.

**Example 2: Incorrect Debug Configuration**

Let's examine how an incorrect debug configuration can lead to problems. Consider a scenario where you've started your Node.js application via `npm start` in a terminal outside of WebStorm, which internally executes something like `node dist/index.js`.

If you set up a "Node.js" debug configuration in WebStorm and attempt to debug using it, and without specifying that you want to attach to an existing process, breakpoints will not be hit. This configuration tells WebStorm to start the app, but you've already initiated it externally, leading to a mismatch. Instead, for this scenario, an “Attach to Node.js/Chrome” configuration is essential. You will need to ensure that the Node.js process you're attaching to has been launched with the `--inspect` or `--inspect-brk` flag. For instance, `node --inspect=9229 dist/index.js` would launch the process that WebStorm needs to attach to and inspect. The debugger listens on a specific port and attempts to connect to this running process.  Failure to specify or to actually launch with the inspect flag will cause the attach debugger to fail silently or the breakpoints to be skipped.

**Example 3: Asynchronous Execution**

Consider a simple example with asynchronous code:

```javascript
// index.js
async function fetchData() {
  console.log("Fetching data...");
  return new Promise(resolve => {
    setTimeout(() => {
      console.log("Data fetched.");
      resolve("Some Data");
    }, 1000);
  });
}

async function processData() {
  const data = await fetchData();
  console.log("Processing data:", data); // Setting a breakpoint here
}

processData();

```

If a breakpoint is placed at the line `console.log("Processing data:", data);` and you execute it without understanding the asynchronous flow, it might appear as though the breakpoint is being skipped or delayed. The code is not skipping the breakpoint, but instead the debugger is pausing execution *after* `await fetchData()` has completed and returned a value, which includes the one second pause from the `setTimeout` function. Understanding that `fetchData` will execute asynchronously, and the `await` operator delays subsequent code, is critical for correctly reasoning about the execution. Breakpoints should be placed, not only in the synchronous body, but also inside the resolve of the Promise.  Using the step over button (F8) can be more appropriate than step in (F7) when stepping through async functions to avoid diving into implementation details.

For further troubleshooting, I recommend consulting the official WebStorm documentation on debugging Node.js applications. This resource provides detailed information about setting up various debug configurations, handling source maps, and dealing with asynchronous code execution. Furthermore, the Node.js documentation on the built-in debugger offers insight into how the inspection protocol functions, which can be very beneficial in comprehending underlying debugger behaviors. Finally, exploring Stack Overflow and other developer forums can uncover similar issues and a wealth of insights from other experienced developers. Focusing on these resources, along with careful attention to code and configurations, usually resolves the issues I've encountered while debugging Node.js applications in WebStorm.
