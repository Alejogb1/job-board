---
title: "Why am I getting the error `The called function should be payable` when sending a value with my Solidity contract?"
date: "2024-12-23"
id: "why-am-i-getting-the-error-the-called-function-should-be-payable-when-sending-a-value-with-my-solidity-contract"
---

, let's unpack this. I've certainly seen my share of `The called function should be payable` errors over the years, and it usually boils down to a mismatch between how you're attempting to send ether to a function and how that function is actually defined in your smart contract. It's a common stumbling block for those new to Solidity, and even experienced developers can get tripped up by it. So, let’s delve into what's happening under the hood.

The core issue revolves around the concept of *payable* functions in Solidity. By default, functions in Solidity contracts are designed to reject any incoming ether. This is a security feature; it prevents unintended or malicious transfers of funds to functions that aren't designed to handle them. When you try to send ether to a non-payable function, the EVM (Ethereum Virtual Machine) throws this error, letting you know that the function's signature doesn't allow it to receive value. A *payable* modifier explicitly tells the compiler, and more importantly, the EVM, that this specific function is designed to accept ether during its execution.

I recall a situation years ago where I was building a marketplace contract. I had implemented a function for buyers to finalize a purchase, which involved sending the agreed-upon ether to the seller’s account. I forgot to add the `payable` modifier to that `finalizePurchase` function, and spent a frustrating couple of hours scratching my head. The transaction would revert with the very error you're experiencing. It wasn't a complicated issue in the end, just a missed keyword.

Let’s look at some examples to clarify this further. Imagine a basic contract:

```solidity
pragma solidity ^0.8.0;

contract BasicExample {

    uint256 public balance;

    function deposit() public {
        balance += msg.value;
    }

    function withdraw(uint256 amount) public {
      require(amount <= balance, "Insufficient funds");
      balance -= amount;
      payable(msg.sender).transfer(amount);
    }
}
```

Here, if you attempt to send ether along with a call to the `withdraw` function, you will get the error because `withdraw` is *not* declared as payable even though it transfers funds. It expects that its only input is the amount, and will fail if the transaction value is non-zero. Conversely, if you were to interact with `deposit` and pass a non-zero amount of ether, the contract will successfully process the deposit since `msg.value` is non-zero, and the function is declared *public*, which is not equal to being *payable*.

Now, let's modify `withdraw` to accept ether, effectively turning it into a function where it might make sense to send value along:

```solidity
pragma solidity ^0.8.0;

contract PayableExample {

    uint256 public balance;

    function deposit() public payable {
        balance += msg.value;
    }

    function withdraw(uint256 amount) public payable {
        require(amount <= balance, "Insufficient funds");
        balance -= amount;
        payable(msg.sender).transfer(amount);
    }
}
```

With this change, both `deposit` and `withdraw` are marked with the `payable` keyword, and can accept ether transfers as part of the function call. In the case of `withdraw`, you might use it to withdraw excess funds that were sent along with a transaction that over-paid a function requiring payment.

The `msg.value` is a special global variable that holds the amount of wei sent along with a transaction calling the smart contract function. Inside a *payable* function, you can access `msg.value` to determine how much ether was sent, as seen in `deposit`. Without the `payable` modifier, this variable would be zero. If you want to use the sent value, this modifier is indispensable.

Let's take another example. Imagine a more complex scenario where you might have different payment processing requirements:

```solidity
pragma solidity ^0.8.0;

contract PaymentExample {

    address payable public recipient;

    constructor(address payable _recipient) {
        recipient = _recipient;
    }


    function payRecipient() public payable {
        recipient.transfer(msg.value);
    }

    function payRecipientSpecificAmount(uint256 amount) public payable {
      require(msg.value >= amount, "Value sent insufficient to cover amount.");
      recipient.transfer(amount);
    }
}
```

In `payRecipient`, all ether sent along with the transaction is forwarded to the address stored in the `recipient` variable. `payRecipientSpecificAmount` on the other hand, makes sure the amount sent is equal to or greater than the specified amount. Without the *payable* modifier on these two functions, it would be impossible to send funds along with the transaction. This is a fundamental piece of how solidity enables ether transfers between addresses via functions.

When you encounter the `The called function should be payable` error, always start by checking the function signature. Make sure that the function you are calling is explicitly declared with the `payable` keyword if you intend to send it some ether. Pay attention to the use cases - a simple data-updating function won't need this modifier, but functions designed to handle ether certainly will. It is essential to understand when and why a function needs to receive ether as part of its execution.

To further your understanding, I'd recommend digging into *Mastering Ethereum* by Andreas M. Antonopoulos and Gavin Wood. This book gives a thorough treatment of the EVM and Ethereum's mechanics. Another excellent resource is the official Solidity documentation, always kept up to date, and should be your first port of call. For a deeper technical dive, I suggest exploring the Ethereum Yellow Paper, a specification of the protocol, though it is rather dense. Additionally, studying example projects on platforms like GitHub can offer practical insights into how developers structure their contracts and manage value transfers.

In summary, the `The called function should be payable` error is there to help you, not hinder you. It’s a safeguard preventing unintended behavior and prompting you to consider the intended purpose and design of your contract functions carefully. It forces you to be explicit about your design decisions with regard to payment and value transfer and allows you to reason more clearly about how value is being handled within your contracts. By thoroughly understanding the payable modifier, you'll be better equipped to construct robust and secure Solidity contracts.
