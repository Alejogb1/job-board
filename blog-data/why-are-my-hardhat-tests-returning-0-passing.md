---
title: "Why are my Hardhat tests returning '0 passing'?"
date: "2024-12-23"
id: "why-are-my-hardhat-tests-returning-0-passing"
---

Alright, let's tackle this "0 passing tests" issue with Hardhat. It's a frustrating situation, I've been there myself – staring at that output thinking, "where did I go wrong?". This isn't a bug that magically appears; it's usually a combination of subtle misconfigurations or assumptions about how Hardhat expects your testing environment to be set up. From my experience, especially dealing with complex multi-contract projects, pinpointing the exact cause often requires a systematic approach. It's rarely a singular error, and more commonly an aggregation of details not quite lining up.

Firstly, consider the fundamental structure. Hardhat, unlike some other testing frameworks, doesn't automatically assume where your tests are, or how you’re defining them. It operates on conventions, but also requires explicit configuration. If you've got your test files sitting in, say, a random 'my_tests' directory, but your `hardhat.config.js` is still expecting them in the default 'test' folder, the framework is simply not going to find anything to execute. It’s analogous to telling someone to look for their keys in the kitchen when they're actually on the bedside table. It's just not the location Hardhat knows to check.

Secondly, examine the nature of your test files. Hardhat expects tests to be written using a testing framework that it understands, typically Mocha, with assertions from a library like Chai. If you're writing tests with a different syntax or if you haven't correctly imported the necessary functions, the tests won't be recognized as such. I once spent a good hour banging my head against the wall realizing that I had accidentally imported `assert` from node's standard library rather than from `chai`. The compiler didn't complain, but mocha was understandably confused. The symptom is often exactly what you're seeing – zero passing, with no helpful error messages.

Thirdly, the connection to your smart contracts might be the problem. You must make sure that Hardhat knows how to locate, compile and deploy your contract, as well as access the correct testing artifacts. Problems here usually stem from incorrect path configurations in your `hardhat.config.js` for contracts, or when dependencies (like imported interfaces or libraries) are not available to the testing environment. I distinctly remember debugging a deployment issue where, due to some library pathing changes, the contracts were compiling fine, but tests could not actually find the deployment artifacts to interact with – hence, no test execution.

Let me illustrate these points with a few code examples.

**Example 1: Incorrect Test Directory Configuration**

Imagine you've organized your test files into a `tests/integration` directory and the file is named `my_integration_tests.js`. Your project structure looks like this:

```
project/
├── contracts/
│    └── MyContract.sol
├── tests/
│    └── integration/
│        └── my_integration_tests.js
├── hardhat.config.js
└── package.json
```

If your `hardhat.config.js` looks like this, it won't work:

```javascript
module.exports = {
  solidity: "0.8.19",
};
```

Because `hardhat` defaults to the `/test` directory, it won't see the tests nested deeper. To correct this, you need to explicitly specify where your test files are located. Here's the corrected `hardhat.config.js` :

```javascript
module.exports = {
  solidity: "0.8.19",
  mocha: {
    testFiles: ["tests/**/*.js"]
  }
};
```

Here, `testFiles: ["tests/**/*.js"]` tells mocha (and by extension, Hardhat) to look recursively in any subfolders within `tests/` for javascript files and execute them as tests.

**Example 2: Missing or Incorrect Assertion Imports**

This is where a subtle import error can cause zero tests to pass. Let's assume your `my_integration_tests.js` looks something like this:

```javascript
const { expect } = require('chai');

describe("MyContract", function () {
  it("Should test a property of my contract", async function () {
    const MyContract = await ethers.getContractFactory("MyContract");
    const myContract = await MyContract.deploy();
    await myContract.deployed();

    // Wrong assertion method:
    assert.equal(await myContract.getValue(), 10); 
  });
});
```

The problem here is that `assert` is from Node.js built-in module and not the `chai` library that is used by Hardhat. `assert.equal` is not compatible with how chai works with hardhat. Hardhat doesn't understand what this means, and as a result, it treats it as a non-test file resulting in zero passing tests. To fix this, replace `assert.equal` with `expect`:

```javascript
const { expect } = require('chai');

describe("MyContract", function () {
  it("Should test a property of my contract", async function () {
    const MyContract = await ethers.getContractFactory("MyContract");
    const myContract = await MyContract.deploy();
    await myContract.deployed();

    // Correct assertion:
    expect(await myContract.getValue()).to.equal(10); 
  });
});
```
This `expect` usage is what mocha and Hardhat are built to work with and will interpret it as a valid test case.

**Example 3: Deployment Artifact Issues**

Finally, issues with getting contract artifacts are also a common culprit. Suppose you have a dependency on another library contract (`LibraryContract`) that's not available during testing, and thus isn't compiled and available. Your main contract `MyContract` uses `LibraryContract`

```
project/
├── contracts/
│    ├── MyContract.sol
│    └── LibraryContract.sol
├── test/
│    └── my_contract_test.js
├── hardhat.config.js
└── package.json
```

If we have a deployment setup for our contract, but the required dependency isn't getting picked up by the testing enviroment, you'll see the tests not finding the right artifact. Let's say the test looks like this:

```javascript
const { expect } = require('chai');
const { ethers } = require("hardhat");

describe("MyContract", function () {
    it("Should use deployed contracts", async function () {
    const LibraryContract = await ethers.getContractFactory("LibraryContract");
    const libraryContract = await LibraryContract.deploy();
    await libraryContract.deployed();

    const MyContract = await ethers.getContractFactory("MyContract");
    const myContract = await MyContract.deploy();
    await myContract.deployed();

        expect(await myContract.someFunctionThatUsesLibrary()).to.equal(true);
    });
});

```
If the path to `LibraryContract` is not properly configured, it might not be deployed. To ensure dependencies work properly, you can modify `hardhat.config.js`:

```javascript
module.exports = {
  solidity: "0.8.19",
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  }
};
```

This isn't a magic fix, however ensuring that your paths are correct can help in situations where the testing environment doesn't know where to look for files.

To go deeper on this, I’d recommend referring to the official Hardhat documentation, which is generally well-structured. Additionally, the Mocha documentation, and the Chai documentation will be particularly useful. For more conceptual understanding about testing methodologies in solidity, "Mastering Ethereum" by Andreas Antonopoulos and Gavin Wood also offers good information on the broader contexts of smart contract testing, though it's not specific to hardhat it will help you with overall testing approaches.

In conclusion, the "0 passing tests" problem in Hardhat is seldom due to a single issue, but more often a convergence of configurations and assumptions. Systematically reviewing your file organization, test syntax, and artifact dependencies, as I've outlined above, will significantly improve your ability to debug and write robust tests. Remember that testing takes iteration and patience, and it's a core skill to master when developing smart contracts.
