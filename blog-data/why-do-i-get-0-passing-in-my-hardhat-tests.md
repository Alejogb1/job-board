---
title: "Why do I get `0 passing` in my hardhat tests?"
date: "2024-12-16"
id: "why-do-i-get-0-passing-in-my-hardhat-tests"
---

Okay, let's get into it. It seems you're encountering the dreaded `0 passing` result in your hardhat tests, and I've certainly been there myself. In my experience, debugging this kind of issue can be a bit like peeling an onion – multiple layers to get through. More often than not, it's not some deeply hidden compiler bug, but rather a misunderstanding in how Hardhat discovers and executes tests.

From my time spent troubleshooting these scenarios, the first place I typically investigate is the test file discovery mechanism. Hardhat relies on pattern matching to identify your test files, and a simple misconfiguration can easily lead to it overlooking them entirely. By default, Hardhat looks for files with the `.test.js`, `.test.ts`, `.spec.js`, or `.spec.ts` suffixes within the `test/` directory. If your test files don't adhere to this convention, that's an immediate red flag.

Furthermore, double-check your `hardhat.config.js` (or `.ts`) file. There is a section where you can specify a custom test directory. If you have accidentally set this to a non-existent folder, or to a directory that doesn’t contain your test files, then, quite naturally, Hardhat won’t find any tests.

Let's assume your file naming and directory structure are correct. The next most probable cause, from my experience, is that you haven't defined any actual tests within those files. It's remarkably common to set up a test file, import all the dependencies, then forget to write any `it(...)` blocks, which are essential to structuring the tests. Here’s a simplified illustration with an example that initially results in the `0 passing` output:

```javascript
// test/example_missing_tests.test.js

const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ExampleContract", function () {
  // No 'it' blocks here!
});
```

If you run this, you will observe that Hardhat reports no passing tests. That's because while we have a `describe` block, the actual individual tests themselves (via `it` blocks) are missing. So, let’s add some basic tests, which would then be correctly interpreted by Hardhat:

```javascript
// test/example_working_tests.test.js

const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ExampleContract", function () {
  it("Should be able to deploy the contract", async function () {
    const ExampleContract = await ethers.getContractFactory("Example");
    const exampleContract = await ExampleContract.deploy();
    await exampleContract.deployed();
    expect(exampleContract.address).to.not.equal(ethers.constants.AddressZero);
  });

  it("Should retrieve a value", async function() {
      const ExampleContract = await ethers.getContractFactory("Example");
      const exampleContract = await ExampleContract.deploy();
      await exampleContract.deployed();
      await exampleContract.setValue(42);
      const retrievedValue = await exampleContract.getValue();
      expect(retrievedValue).to.equal(42);
  });

});
```

This code snippet showcases a basic contract deployment test and a retrieval test. You would now have a `2 passing` output. This is because we've actually added the crucial `it()` blocks that hardhat uses to recognize individual tests. I've found over the years that a forgotten `it` block is a quite common oversight.

Another, somewhat less obvious, source of this error stems from asynchronous operations within a test suite that aren't properly awaited. Hardhat tests rely on the async/await pattern to manage asynchronous interactions with the blockchain. If an async function inside a `beforeEach` or `it` block isn't awaited properly, the tests might conclude before the necessary setup or assertions take place. Thus, Hardhat has no test to "pass." The following example demonstrates the problem:

```javascript
// test/example_broken_async.test.js

const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Async Setup Issue", function () {
  let exampleContract;

  beforeEach(function () {
   // Note: Missing "await"
    const ExampleContract = ethers.getContractFactory("Example");
    ExampleContract.deploy().then(function(contract){
      exampleContract = contract;
    });
  });

  it("Should perform an action with contract", async function () {
      // At this point, exampleContract may not be set yet
     if(exampleContract){
       await exampleContract.deployed();
       expect(exampleContract.address).to.not.equal(ethers.constants.AddressZero);
    } else {
     expect(false).to.be.true; // Will trigger failure
   }

  });
});

```

The issue here is that the deployment inside `beforeEach` uses `then` and does not `await`. The `it` block may run before the contract has actually been deployed, leading to unpredictable or failed tests. Correcting this is straightforward by using `async/await`:

```javascript
// test/example_fixed_async.test.js

const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Async Setup Issue", function () {
  let exampleContract;

  beforeEach(async function () { // Added 'async' and 'await'
    const ExampleContract = await ethers.getContractFactory("Example");
    exampleContract = await ExampleContract.deploy();
    await exampleContract.deployed();

  });

  it("Should perform an action with contract", async function () {
       expect(exampleContract.address).to.not.equal(ethers.constants.AddressZero);
  });
});
```

By introducing the await keyword both inside the `beforeEach` setup and also making the function `async`, we ensure the code will wait for the contract to deploy before moving to the next step of the tests. This resolves the race condition and the test now passes.

To further investigate, I often use the Hardhat verbose output. Running your test command with the `--verbose` flag provides a lot of additional information, which will, in many cases, help point you to the root cause of the issue if the above suggestions don't help.

In terms of further reading, I'd strongly recommend delving into the official Hardhat documentation; it provides excellent explanations of test setups and common issues. Also, the "Testing Smart Contracts" section in the book "Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood is a valuable resource. Although it uses truffle as a tool, the testing principles are transferable, providing a broader understanding of the best practices. Furthermore, exploring the Chai assertion library documentation, which Hardhat uses, can also help you to gain a better understanding of how you can express your tests.

These are a few of the most common scenarios I've encountered, and while it's impossible to pinpoint the exact cause without more specific details about your setup, going through this systematic checklist has usually helped me solve the "0 passing" problem. Remember to double-check, step by step, your file structure, test definitions, and asynchronous operations. This is where most issues tend to lie.
