---
title: "How to validate functions in a hardhat test?"
date: "2024-12-15"
id: "how-to-validate-functions-in-a-hardhat-test"
---

alright, so you're looking at how to validate functions within your hardhat tests, right? i've been there, and it's a pretty common spot to stumble. let me walk you through how i've handled this in the past. it's not about just calling the function and hoping it works; it's about making sure it does *exactly* what it's supposed to do, under various scenarios. think of it as creating little contracts within your tests to guarantee function behavior.

first off, let's talk about the basic setup. assume you've got a hardhat project up and running. you have your smart contracts in the `contracts` folder, and your tests live in the `test` folder, typically using mocha and chai for your testing framework. if you are starting out, read 'testing javascript applications' by luciano ramalho or 'effective javascript' by david herman both have good explanations on creating robust javascript tests.

the core of validating a function isn't all that difficult, it's about defining clear expectations. you're essentially setting up a situation, calling your function, and then asserting that the state of the contract or the returned value is what you'd expect. that "expectation" is what i call 'the assertion part'.

let's look at some scenarios. suppose we have a simple contract, maybe something that handles token transfers. you might have a function called `transfer`, and you’d want to test it. here's a basic example of how that would look in a test file:

```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("TokenContract", function () {
  let TokenContract;
  let tokenContract;
  let owner;
  let addr1;
  let addr2;

  beforeEach(async function () {
    [owner, addr1, addr2] = await ethers.getSigners();
    TokenContract = await ethers.getContractFactory("TokenContract");
    tokenContract = await TokenContract.deploy(1000);
    await tokenContract.deployed();
  });

  it("should transfer tokens between accounts", async function () {
      const initialBalanceOwner = await tokenContract.balanceOf(owner.address);
      const initialBalanceAddr1 = await tokenContract.balanceOf(addr1.address);

      await tokenContract.transfer(addr1.address, 100);

      const finalBalanceOwner = await tokenContract.balanceOf(owner.address);
      const finalBalanceAddr1 = await tokenContract.balanceOf(addr1.address);

      expect(finalBalanceOwner).to.equal(initialBalanceOwner - 100);
      expect(finalBalanceAddr1).to.equal(initialBalanceAddr1 + 100);

  });
});
```

in this first example, we're setting up a `TokenContract`, deploying it with some initial supply, and then testing the `transfer` function. we grab the initial balances, make the transfer, then check the balances again to ensure the transfer happened as expected. notice the use of `expect` from chai: it's what we use to make our assertions. we are using the `equal` matcher, but there are many more options for complex verifications.

now, a frequent spot for issues to arise is when you're dealing with functions that should `revert` under certain conditions. for instance, if a user tries to transfer more tokens than they own, it should error. hardhat's test setup makes handling reverts rather easy. let me illustrate with another test example.

```javascript
it("should revert if sender has insufficient balance", async function () {
    await expect(
      tokenContract.connect(addr1).transfer(addr2.address, 100)
    ).to.be.revertedWith("ERC20: transfer amount exceeds balance");
  });
```

here, we're specifically testing that the transfer function `reverts` when the sender (`addr1`) doesn't have the funds, as it is supposed to according to the erc20 standard.  the `.connect(addr1)` is crucial: we're telling hardhat to send the transaction as the `addr1` address, not as the contract owner.  the `revertedWith` matcher checks that the error thrown by the smart contract contains the specific error message. using such explicit and detailed checks helps you pinpoint the source of problems faster, that has been my experience from working with complex contracts.

moving on, functions might not always be just transferring data. a function could be calculating something, like a complicated interest formula or some cryptographic operation. when the return value is complex it can be hard to validate. you might need a specific precision comparison or check against a known value. here's an example of how to handle a function with a more complex return value:

```javascript
  it("should correctly calculate interest", async function () {
    const principal = ethers.utils.parseEther("100");
    const rate = 5;
    const duration = 12;
    const expectedInterest = principal.mul(rate).mul(duration).div(100 * 12);

    const calculatedInterest = await tokenContract.calculateInterest(principal, rate, duration);

    expect(calculatedInterest).to.equal(expectedInterest);
  });
```

in this case, we have a `calculateInterest` function that returns the calculated interest. we compute the expected result on the test side, and then assert that the contract’s returned value matches our calculation. here we did not use a 'closeTo' matcher as we are not working with a real numbers context, but if you use floats or integers with a high range of values you should consider using the 'closeTo' matcher in chai for handling floating point or integer errors when dealing with very large values to avoid false negatives in your testing.

now, i've run into cases where my contract uses libraries and sometimes it's hard to debug. when debugging it can be useful to print some data for comparison, although this is not a best practice, for debugging a contract you should try hardhat's console functionality. it's also important to be testing every path of your function (and code). if you have if/else statements in your function, you want a test for every if and every else branch. that guarantees you've touched all the code, and are not making assumptions about what it's doing. that's why it's so important to design your tests before implementation.

also, i've found myself multiple times debugging a transaction that was succeeding but giving me incorrect results. you can't always rely on just the returned value from a transaction: sometimes you need to also check for emitted events. many smart contract functions emit events upon completion, and validating those is also part of creating tests that are robust. you can inspect those events with hardhat.

let me be a bit more pragmatic: i remember once i had a bug where the smart contract would do weird things during high gas usage scenarios. i ended up using hardhat's gas reporting tools to simulate these, which is a good way of catching potential bugs that only show up in rare conditions. these tools help to better understand how functions perform under heavy load and different gas prices, it is something that you must use when you feel the tests are not enough and more edge-case scenarios are needed.

by the way, i hope you've been keeping your tests short and concise: long, complex tests are very hard to debug and maintain and you'll find out you spend more time testing than programming if you don't follow that rule. the way i see it, a test should only focus on one thing. if you're testing something that needs data from a lot of other parts of the system, consider mocking/stubbing those parts to isolate the unit you are validating.

regarding resources, for general testing best practices and patterns i would recommend 'xunit test patterns' by gerard meszaros, it has a comprehensive guide to unit testing concepts, and 'the art of unit testing' by roy osherove for more implementation specific advice.

one last bit of advice, and it’s a joke (and maybe it is not funny at all): testing is like flossing your smart contracts: you might not like it, but it will save you from a big problem in the future.

well, hopefully these few examples and tips help you along the way in your smart contracts journey. remember, testing is a journey, not a destination. the more you test and validate, the more secure and reliable your smart contract will become. you got this.
