---
title: "Why does a Solidity bool variable not change to false after being assigned 'true'?"
date: "2024-12-23"
id: "why-does-a-solidity-bool-variable-not-change-to-false-after-being-assigned-true"
---

Let’s unpack this one – it’s a common point of confusion for those new to Solidity, and frankly, I’ve seen even experienced devs stumble over this, having once spent a rather frustrating afternoon debugging a smart contract that exhibited precisely this behavior. The short answer is: a Solidity `bool` variable, once set to `true`, absolutely *can* be changed to `false`. The problem likely lies not with the variable’s mutability itself but with the context in which the assignment is occurring. Let's explore what causes this misconception.

Firstly, understand that Solidity functions have varying visibility and mutability characteristics. Functions can be declared as `view`, `pure`, or they can modify the contract’s state. Crucially, `view` and `pure` functions cannot alter the state of the blockchain. If you're attempting to change the boolean value within a function that's marked as `view` or `pure`, your changes will be purely local to that function call. They won't persist to the blockchain state itself, which might lead you to think the boolean isn't changing, when in fact, your attempt is just ignored by the underlying EVM. This is particularly important for those coming from more procedural programming environments, where changes to local variables are often preserved.

Second, the issue may stem from incorrect transaction execution. In Ethereum, state-changing operations (like modifying a boolean) must be done via transactions that are then submitted to the network. If you're attempting to modify a boolean through a call (not a transaction) or from a read-only function, the change will not persist. The subtle difference here is between a function call which retrieves data from the blockchain versus a transaction which modifies its state.

Finally, and often overlooked, improper variable scoping and usage within complex contracts can also lead to this impression. For example, shadowing of variables or unintended reassignment can result in behavior that appears as though the boolean is not changing.

To illustrate, let’s consider a few examples.

**Example 1: The 'View' function Pitfall**

In this scenario, we have a function intended to flip the boolean value, but it’s marked as `view`.

```solidity
pragma solidity ^0.8.0;

contract BooleanTest {
    bool public myBool = true;

    function tryFlipBool() public view {
        myBool = false; // This change will not persist
    }

    function getBool() public view returns (bool) {
        return myBool;
    }
}
```

In this contract, when you call `tryFlipBool()` and then call `getBool()`, `getBool()` will still return `true`. Why? Because `tryFlipBool()` is a `view` function, and modifications within it are not saved to the blockchain’s state. The `myBool` variable *does* get locally set to `false` within `tryFlipBool()`, but that change is not written back to storage.

**Example 2: The Correct Approach: State-Modifying Function**

Now, let’s modify `tryFlipBool()` to be a state-changing function using no modifiers.

```solidity
pragma solidity ^0.8.0;

contract BooleanTest {
    bool public myBool = true;

    function flipBool() public {
        myBool = false; // This change persists
    }

    function getBool() public view returns (bool) {
        return myBool;
    }
}
```

In this improved scenario, calling `flipBool()` via a transaction will modify the state, causing `myBool` to now evaluate to `false`. Calls to `getBool()` after the transaction confirms will now return the modified value. The absence of `view` or `pure` indicates to the compiler (and the EVM) that the function *intends* to alter the blockchain state and therefore the modification of `myBool` is preserved.

**Example 3: Scoping and Accidental Reassignment**

Lastly, here’s an example that demonstrates a more subtle error: accidental reassignment due to scope.

```solidity
pragma solidity ^0.8.0;

contract BooleanTest {
    bool public myBool = true;

    function checkAndMaybeFlip(bool inputBool) public {
        bool myBool = inputBool; //shadowing here!
        if (myBool == true) {
             myBool = false; // this change is local to this function scope and doesn't affect the contract's myBool.
        }
    }
    function getBool() public view returns (bool) {
      return myBool;
    }
}
```

Here, within `checkAndMaybeFlip`, we declare a *new* local variable called `myBool`, effectively shadowing the contract-level `myBool` variable. Any changes to this local variable will not affect the contract’s `myBool`. When the function terminates, the changes to the local `myBool` are discarded, leaving the public variable unaffected. This highlights the importance of paying careful attention to variable scoping and avoiding accidental shadowing.

So, to summarize, while it might initially *appear* that a `bool` variable stubbornly refuses to become `false` after being set to `true`, the root cause is invariably one of: incorrect function modifiers such as `view` or `pure`, making changes via calls not transactions, or, occasionally, issues of variable scoping or accidental reassignment within complex contracts.

To learn more about these nuances in detail, I highly recommend exploring the official Solidity documentation and thoroughly reviewing books like “Mastering Ethereum” by Andreas M. Antonopoulos and Gavin Wood, which provides an in-depth look at Ethereum’s inner workings, and “Programming Blockchain with Solidity” by Chris Dannen, which is a more programming-focused view, but covers essential topics such as function modifiers and storage in a smart contract context very well. Furthermore, academic papers on the Ethereum Virtual Machine (EVM) architecture and its state management can give you a much deeper understanding of why certain design choices are implemented as they are and help avoid these types of errors in the future. Careful reading of these resources, coupled with persistent experimentation and debugging, will solidify your understanding of Solidity’s behavior and allow you to write more robust smart contracts. Remember, attention to detail, particularly regarding the subtleties of Solidity, is paramount.
