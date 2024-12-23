---
title: "Why isn't the referrer function sending msg.value to the correct address?"
date: "2024-12-23"
id: "why-isnt-the-referrer-function-sending-msgvalue-to-the-correct-address"
---

Okay, let's tackle this. I've bumped into this particular issue countless times in my years developing smart contracts, and it's almost always a subtle misunderstanding of how `msg.value` and transaction contexts operate within the ethereum virtual machine (evm). Let’s unpack it. The short answer is this: the referrer function, as you’re experiencing, probably isn’t explicitly *forwarding* the `msg.value`, it’s merely being called, typically within the context of another transaction.

Imagine this scenario, back in my early days working on a decentralized exchange prototype. We had a core smart contract handling trades, and we wanted to implement a referral system. The initial approach was simplistic: a user would call the main trade function, and if a referrer address was provided, we'd trigger another function on the same contract. We expected the full `msg.value` from the user's initial trade transaction to automatically pass along to this referral function, thinking the internal call would propagate that context. We were, to put it mildly, incorrect.

The key concept to understand here is that `msg.value` represents the amount of ether *directly* sent to the *current* function being executed. When you call a function *internally*, from within another function of the same contract, it doesn’t inherently inherit the original `msg.value`. Instead, the internally called function sees `msg.value` as zero, unless explicitly included when making the internal call. Let’s examine this in more detail.

Think of it like a cascade. The initial transaction starts from an externally owned account (eoa), with a specific amount of ether as `msg.value`. The first function the contract executes *does* have access to this initial `msg.value`. However, when that function then calls a second function within the same contract, the second function is considered an *internal* call. It doesn't automatically pick up the initial `msg.value`; rather, it starts with a clean slate where `msg.value` is zero, unless explicitly specified.

The issue arises because the referrer function isn’t on the receiving end of a direct, external transaction. It’s a function being executed internally, triggered by the main transaction function. It's not being ‘sent’ the `msg.value`; it’s being executed within the execution scope set up by the initial transaction’s call.

Here’s a simplified code example to illustrate:

```solidity
pragma solidity ^0.8.0;

contract ReferralExample {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function mainTransaction(address referrer) payable public {
        // This function receives msg.value from the external transaction
        // Logic of the main transaction (e.g., a trade)

        if (referrer != address(0)) {
             _handleReferral(referrer); // Internal call, DOES NOT forward msg.value by default
        }

    }


    function _handleReferral(address referrer) internal {
        // Inside this function, msg.value will be 0 unless passed explicitly.
        (bool success, ) = referrer.call{value: msg.value}("");
        require(success, "Referral transfer failed");

    }

    receive() external payable {}

}

```

In the code snippet above, `mainTransaction` receives the `msg.value` correctly. However, the internally called function, `_handleReferral`, would see a `msg.value` of zero by default if it wasn't set explicitly during the call. It’s crucial to understand, therefore, that transferring value to another address in your referral function needs to be an *explicit* value transfer using `.call{value: amount}(...)`. This is not automatic.

Here's a second example, showcasing a common pattern to fix this:

```solidity
pragma solidity ^0.8.0;

contract CorrectReferralExample {
    address public owner;

     constructor() {
         owner = msg.sender;
     }
    function mainTransaction(address referrer) payable public {
        // This function receives msg.value from the external transaction
        // Logic of the main transaction (e.g., a trade)

        if (referrer != address(0)) {
            _handleReferral(referrer, msg.value); // Explicitly pass msg.value
        }
    }

      function _handleReferral(address referrer, uint256 amount) internal {
        // Now msg.value is not used, it receives the passed-in 'amount'
           (bool success, ) = referrer.call{value: amount}("");
           require(success, "Referral transfer failed");

     }
    receive() external payable {}
}
```

In this corrected example, we're explicitly passing the `msg.value` from `mainTransaction` as an argument to the `_handleReferral` function. This allows the `_handleReferral` function to use this value when sending ether to the referrer address.

Lastly, here's a snippet showing how to handle partial referrals:

```solidity
pragma solidity ^0.8.0;

contract PartialReferralExample {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

      uint256 public referralPercentage = 10; // 10% referral fee


    function mainTransaction(address referrer) payable public {
         // This function receives msg.value from the external transaction
         uint256 referralAmount = (msg.value * referralPercentage) / 100;

        if (referrer != address(0) && referralAmount > 0) {
            _handleReferral(referrer, referralAmount);
            // optionally process the remainder of msg.value in the main transaction here
        }

    }

    function _handleReferral(address referrer, uint256 amount) internal {
      (bool success, ) = referrer.call{value: amount}("");
        require(success, "Referral transfer failed");

    }
    receive() external payable {}
}
```

Here, I'm extracting a portion of the initial `msg.value` as the referral amount, explicitly passing that and the referrer's address to the `_handleReferral` function for a value transfer.

In summary, the referrer function doesn't automatically get `msg.value` because it’s called internally, within the context of another function in the same contract. The original `msg.value` is only accessible to the first function in the call stack within the contract. To send ether during an internal call, you must *explicitly* pass the intended value to the internal function and make the transfer using `.call{value: amount}(...)`.

For further depth, I'd recommend taking a look at the official Solidity documentation, specifically the sections related to function visibility, call context, and the low-level `.call()` operation. Additionally, “Mastering Ethereum” by Andreas Antonopoulos and Gavin Wood offers comprehensive coverage on the technical aspects of smart contracts and the evm, which is invaluable for developing a solid understanding of these nuances. You'll also find insightful information in "Ethereum Yellow Paper" by Gavin Wood, which details the technical specification of the evm. A strong grasp of these concepts will save you significant headaches down the road. This is a core understanding crucial for any serious smart contract development.
