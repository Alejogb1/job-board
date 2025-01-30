---
title: "Why is my Solidity function reporting 'The called function should be payable'?"
date: "2025-01-30"
id: "why-is-my-solidity-function-reporting-the-called"
---
Solidity requires explicit designation of functions as `payable` when they are designed to receive ether. This requirement stems from the Ethereum Virtual Machine's (EVM) handling of ether transfers and protects against accidental loss of funds. Failure to declare a function `payable` when attempting to send it ether will result in the "The called function should be payable" error, signaling a mismatch between the function's expected behavior and the action attempted.

The EVM executes operations based on strict opcode definitions. When ether is sent to a contract, the EVM checks if the function called along with the transaction is explicitly marked as capable of receiving ether. Without the `payable` modifier, a function's compiled bytecode does not include logic to properly handle incoming ether. Attempting to send ether to a non-payable function therefore triggers the error, forcing developers to clearly define when a function should accept value. This approach also maintains the integrity of contract state, ensuring ether is processed as intended within the defined logic. It is not sufficient for a function to simply manipulate balances; it must also be prepared, by virtue of the `payable` designation, to actively receive ether as part of a transaction.

The issue arises commonly when calling functions through methods such as `transfer()` or `send()`, or specifying an ether value when triggering a function call directly using `call()`. If these methods are employed with the expectation that ether will be processed without the corresponding `payable` modifier on the recipient function, the transaction will fail and revert, thereby preventing unintentional loss of user funds.

Consider the following contract, representing an initial and flawed attempt at handling payments. The function `purchaseProduct()` aims to allow users to pay for a product, but lacks the `payable` modifier:

```solidity
pragma solidity ^0.8.0;

contract ProductStore {
    uint256 public productPrice = 1 ether;
    address payable public owner;

    constructor() {
        owner = payable(msg.sender);
    }

    function purchaseProduct() public {
        require(msg.value == productPrice, "Incorrect amount sent.");
        // Some logic here to manage the product
        payable(owner).transfer(msg.value);
    }
}
```

This code will produce the error if someone sends ether when invoking the `purchaseProduct()` function, because it's not marked payable. Note that while `payable(owner).transfer(msg.value)` *itself* correctly handles transferring ether, the called function, namely `purchaseProduct()`, must *also* explicitly state it can receive this initial transfer. The absence of `payable` prevents the initial ether transfer from reaching `purchaseProduct()` within the `msg.value` context.

To resolve this, I need to add the `payable` modifier to the function definition, indicating that it is designed to handle incoming ether:

```solidity
pragma solidity ^0.8.0;

contract ProductStore {
    uint256 public productPrice = 1 ether;
    address payable public owner;

    constructor() {
        owner = payable(msg.sender);
    }

    function purchaseProduct() public payable {
        require(msg.value == productPrice, "Incorrect amount sent.");
        // Some logic here to manage the product
        payable(owner).transfer(msg.value);
    }
}
```

The inclusion of `payable` in `function purchaseProduct() public payable` enables the function to correctly process the received ether from the initial transaction and complete the transaction without reversion. It is now correctly configured to receive ether and then forward it to the contract owner.

Lastly, consider a scenario where a contract has a fallback function. If that function should receive ether, it too must be declared `payable`. The fallback function is called by the EVM when a message is received that does not match any function signatures in the contract, and must also be prepared to handle situations where ether is sent along with the message. Failing to do so will trigger the same "The called function should be payable" error.

```solidity
pragma solidity ^0.8.0;

contract FallbackExample {
    address payable public owner;

    constructor() {
        owner = payable(msg.sender);
    }

    // Incorrect example: fallback function without payable
    fallback() external {
        payable(owner).transfer(msg.value);
    }
}
```

In this example, because the `fallback()` function is not declared `payable`, any attempt to send ether directly to this contract will result in a transaction failure. To allow ether to be deposited using the fallback mechanism, you must use `payable` keyword.

```solidity
pragma solidity ^0.8.0;

contract FallbackExample {
    address payable public owner;

    constructor() {
        owner = payable(msg.sender);
    }

    // Correct example: payable fallback function
    fallback() external payable {
       if (msg.value > 0) {
           payable(owner).transfer(msg.value);
       }
    }

    receive() external payable {
       if (msg.value > 0) {
           payable(owner).transfer(msg.value);
       }
    }
}
```

Here, both the `fallback()` and the `receive()` functions are now `payable`, which allows the contract to handle incoming ether when no specific function is called. The conditional `if (msg.value > 0)` clause within each function ensures that the transfer operation is executed only if an amount of ether has been attached to the transaction. The `receive` function is a more recent and preferred method for handling ether sent directly to a contract address. However, a contract cannot have *both* `fallback()` and `receive()` without one specifically dispatching to the other.

In summary, the `payable` modifier in Solidity is not merely a suggestion; it's a critical safety mechanism enforced by the EVM. I've seen this error multiple times, and it almost always stems from overlooking this specific requirement when designing functions meant to receive ether. Without it, the EVM will treat any attempt to send ether to that function as an error. Careful application of `payable`, understanding of the underlying EVM, and diligent testing can effectively avoid this recurring issue.

For further exploration into Solidity's features related to ether transfer, I recommend consulting the official Solidity documentation, various Ethereum developer forums, and the book "Mastering Ethereum". These resources provide extensive details on contract security, best practices, and the subtle nuances of Solidity, which can greatly improve one's comprehension of the underlying mechanics and prevent common pitfalls.
