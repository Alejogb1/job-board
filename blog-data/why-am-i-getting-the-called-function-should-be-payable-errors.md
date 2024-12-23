---
title: "Why am I getting 'The called function should be payable' errors?"
date: "2024-12-23"
id: "why-am-i-getting-the-called-function-should-be-payable-errors"
---

Okay, let's address this "payable function" error. I’ve seen this crop up plenty of times, and it usually boils down to a mismatch between how you’re calling a function and how that function is declared within a smart contract, specifically when interacting with the Ethereum Virtual Machine (evm). It’s a foundational concept, but the error message isn't always immediately clear, so let's unpack it.

The core issue revolves around the `payable` modifier. In Solidity, and by extension other languages targeting the evm, functions can be designated as `payable` or non-`payable`. A `payable` function explicitly indicates that the function is designed to receive ether (or any other native cryptocurrency of an evm-compatible chain) during its execution. If a function isn't marked as `payable`, the evm rejects any transaction that attempts to send ether along with the function call. This is a security measure to prevent unintended fund transfers and ensures contracts handle funds responsibly. The error message "The called function should be payable" simply means you’re attempting to send ether to a function that hasn't explicitly declared it’s equipped to receive it. It’s like trying to deposit cash into a vending machine that only accepts cards.

I recall one particularly frustrating case back when I was building a decentralized exchange prototype. The core `deposit()` function was throwing this error intermittently. I had initially declared it as non-`payable`, then seemingly changed it to `payable` but the issue persisted. Turns out, I was only updating the function signature in my local copy of the contract, but the bytecode on the network was still reflecting the old, non-`payable` state. I ended up wasting a good chunk of time troubleshooting because I didn't double-check the deployed contract version on the blockchain. Always verify your deployment.

Let's get down to some code to illustrate the concepts.

**Example 1: Incorrect non-payable function call**

```solidity
pragma solidity ^0.8.0;

contract ExampleContract {
    uint256 public value;

    function setValue(uint256 _newValue) public {
        value = _newValue;
    }
}
```
This simple contract defines a function `setValue` which is non-`payable`. Now, let's see how this behaves when called incorrectly. Assume this contract has been deployed and its address is known. In a JavaScript-based environment interacting with the contract (like web3.js or ethers.js), we might mistakenly attempt to send ether:

```javascript
// Assume web3 is initialized and contract is loaded as 'contractInstance'
contractInstance.methods.setValue(123).send({from: userAddress, value: web3.utils.toWei('0.1', 'ether')})
  .then((receipt) => {
    console.log("Transaction Receipt: ", receipt);
  })
  .catch((error) => {
    console.error("Transaction failed:", error); // You'd see the payable function error here
  });

```

This transaction would fail with the described "payable function" error because we are attempting to send `0.1` ether to a function (`setValue`) that is not explicitly declared `payable` in the Solidity code. The evm rejects it to maintain data integrity.

**Example 2: Correct payable function call**

Let’s now modify the contract to correctly handle ether transfers.

```solidity
pragma solidity ^0.8.0;

contract ExampleContract {
    uint256 public balance;

    function deposit() public payable {
       balance += msg.value;
    }
}
```
Now, `deposit` is declared `payable`. This allows it to receive ether, which is accessible via `msg.value`. Now, let's execute a proper call:

```javascript
// Assume web3 is initialized and contract is loaded as 'contractInstance'
contractInstance.methods.deposit().send({from: userAddress, value: web3.utils.toWei('0.1', 'ether')})
  .then((receipt) => {
    console.log("Transaction Receipt: ", receipt); // Success this time
  })
  .catch((error) => {
    console.error("Transaction failed:", error);
  });

```
This code would now execute successfully, the ether would be received by the contract, and the contract’s `balance` would be updated.

**Example 3: Payable fallback function (edge case)**

There's another scenario where this can pop up and that’s the fallback function. The fallback function is a special function that's called when a contract receives ether without a specific function selector or when no other functions match the incoming call. It's often `payable`, but it's worth mentioning because misconfigurations are common.

```solidity
pragma solidity ^0.8.0;

contract FallbackExample {
    uint256 public balance;
    receive() external payable {
        balance += msg.value;
    }
    fallback() external payable {
       //optional, but here for illustrative purposes
    }
}

```
Both `receive` and `fallback` functions are marked as `payable` here. Any transaction that sends ether to this contract without matching a specific function signature will trigger the `receive` function, if present or the fallback function if there is not `receive` function and there is a `fallback`. If these functions are not marked `payable`, you'll encounter the same "payable function" error.

To delve further into this, I highly recommend reading the official Solidity documentation (available online) and focusing on the sections discussing modifiers and the `msg` object (especially `msg.value`). For a deeper dive into evm mechanics and smart contract development best practices, consider reading "Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood. This book gives you not just the practical aspects but also the reasoning behind these design choices. It’s good to understand *why* these are enforced, not just how to write the code. Also, keep an eye on resources related to specific frameworks like web3.js or ethers.js for more context on how these interact with deployed contracts from the client-side. Specifically understanding the nuances of `send()` function and how to handle sending value is crucial.

In summary, the "payable function" error stems from a mismatch between what your code is trying to do (send ether) and how the smart contract function is declared to accept that ether. Always ensure the function signature is correct and that the deployed bytecode matches the code you expect. Triple check those deployments and always, always use the appropriate tooling to debug. Getting hands-on with these errors is how you learn best, so keep practicing and you'll get the hang of it.
