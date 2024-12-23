---
title: "Why does tx.origin not match address1, even when they are the same address in Solidity?"
date: "2024-12-23"
id: "why-does-txorigin-not-match-address1-even-when-they-are-the-same-address-in-solidity"
---

Alright, let’s tackle this interesting quirk in Solidity. It's a scenario that has tripped up many developers, and frankly, I recall a project a few years back where it caused me a bit of a headache too. We were building a decentralized voting system, and the inconsistent behavior of `tx.origin` versus `msg.sender` created some unexpected authorization issues. So, let's delve into the why.

Essentially, `tx.origin` and `msg.sender` behave differently due to their underlying function within the Ethereum transaction context. When you see that `tx.origin` doesn't match an address like `address1`, even when you *think* they should be the same, it's not a bug; it's by design. Let’s break down the critical distinction.

`msg.sender` represents the address that directly called the current contract function. Think of it as the immediate caller in a series of contract interactions. If contract A calls a function in contract B, within B's execution context, `msg.sender` will be the address of contract A.

On the other hand, `tx.origin` represents the *external address* that initiated the entire transaction chain. This is the address of the user who started the transaction from an externally owned account (EOA), like your MetaMask address, regardless of how many contract calls were involved in that chain. It's essentially the root of the entire call stack.

The reason they diverge in your scenario, and this is important, is because your code is most likely indirectly invoking the second contract through an intermediary. So, let’s say `address1` is your external account. If your account, `address1`, calls function *A* in contract *X*, and function *A* in turn calls function *B* in contract *Y*, then within the context of function *B*, `msg.sender` would be the contract address of *X*, *not* your address (`address1`), but `tx.origin` *will* remain your address (`address1`).

This difference is crucial for security, especially concerning reentrancy attacks. If contracts relied solely on `tx.origin` for authorization, a malicious contract could disguise its origin by acting as an intermediary, fooling the recipient contract into thinking the call originated from the user rather than the attacker. `msg.sender`, being context-specific, provides that crucial layer of protection.

Let’s illustrate with some code examples. We will create three contracts: `User`, `ContractA`, and `ContractB`.

```solidity
// User.sol
pragma solidity ^0.8.0;

contract User {
    address public userAddress;

    constructor() {
        userAddress = msg.sender;
    }

    function getOrigin() public view returns (address) {
        return tx.origin;
    }

    function getSender() public view returns (address) {
        return msg.sender;
    }
}
```

The `User` contract simply stores the user's address upon deployment and allows us to view `tx.origin` and `msg.sender`. Now, let's add the next contract:

```solidity
// ContractA.sol
pragma solidity ^0.8.0;

import "./ContractB.sol";

contract ContractA {
    ContractB public contractB;

    constructor(address _contractBAddress) {
        contractB = ContractB(_contractBAddress);
    }

   function callContractB() public {
       contractB.viewOrigin();
       contractB.viewSender();
    }
    function getOrigin() public view returns (address) {
        return tx.origin;
    }

    function getSender() public view returns (address) {
        return msg.sender;
    }

}
```

`ContractA` initiates the interaction with `ContractB`. Here's the code for ContractB:

```solidity
// ContractB.sol
pragma solidity ^0.8.0;

contract ContractB {
    address public origin;
    address public sender;
   function viewOrigin() public {
      origin = tx.origin;
   }

    function viewSender() public {
       sender = msg.sender;
    }
}
```
Now, if you deploy `User`, `ContractB`, and `ContractA` (passing the address of `ContractB` to `ContractA`'s constructor), then deploy the user contract and then call the `callContractB` in Contract A, then you can observe that even though the user’s address is `tx.origin`, the `msg.sender` is different for `ContractB`.

Now, if we deployed the User contract as contract C and deployed and interact with ContractA using that instance, then the user's address is not going to match `tx.origin`. This is important when doing authorization.

In this example, if `address1` (your MetaMask address or whichever EOA you use) initiated the transaction chain, and that transaction first called `callContractB` via contract `ContractA` in our example, then within `ContractB`, `tx.origin` would still be `address1`, but `msg.sender` would be the address of `ContractA`. This discrepancy exists precisely because the call didn’t directly originate from the EOA to `ContractB`.

A crucial distinction to highlight is that if `address1` directly calls a function in `ContractB`, then in the execution of that function `tx.origin` *and* `msg.sender` would be equal to `address1`.

So, why did I face issues with this when building the decentralized voting system? We were using `tx.origin` to authorize certain actions, assuming the direct user was always interacting with the smart contract. However, we added another layer through a proxy contract for security. This proxy contract introduced that layer of indirection, and our authorizations, which were based on `tx.origin` were no longer valid in the specific contract, so we had to migrate to `msg.sender` based authorizations.

For further understanding, I'd highly recommend reading the Ethereum Yellow Paper, specifically section 4 regarding transaction and execution context. Also, “Mastering Ethereum” by Andreas M. Antonopoulos and Gavin Wood is a good text. Section 4 of the Solidity documentation offers a very explicit account of the variables available in the execution context. I also advise looking up discussions on reentrancy attacks and the use of `msg.sender` for mitigation, such as those found in the Ethernaut challenges on OpenZeppelin (although they do not always contain direct documentation, the practical aspect of them is helpful). These resources will provide a much more comprehensive grasp on this topic.

The key takeaway here is that understanding the difference between `tx.origin` and `msg.sender` is paramount for writing secure and robust smart contracts. Avoid using `tx.origin` for authorization. Use `msg.sender`, and always account for contracts acting as intermediaries. It is a lesson I learned the hard way, and hopefully, this explanation helps others avoid similar pitfalls.
