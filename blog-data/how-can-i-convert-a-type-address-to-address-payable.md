---
title: "How can I convert a type address to address payable?"
date: "2024-12-23"
id: "how-can-i-convert-a-type-address-to-address-payable"
---

Alright, let's tackle this address conversion challenge. It's a common hiccup, especially when dealing with older smart contracts or bridging across different contract versions, and I've certainly been in the trenches with this one a few times. You’re working with Solidity, most likely, and the issue boils down to the nuanced distinction between `address` and `address payable`. In short, `address` is just that, an address, a location on the Ethereum blockchain, while `address payable` is specifically an address that *can* receive ether. The compiler enforces this distinction to maintain type safety and prevent accidental transfers to addresses not meant to receive funds. So, how do we gracefully move between these two types?

The straightforward conversion isn’t a direct cast. You can’t just write `address payable(my_address)` and expect the compiler to be happy. Instead, we use a type conversion process. Solidity doesn’t allow implicit conversions from `address` to `address payable` because that would break the strong typing system. However, there is a way.

Essentially, we must bypass the compiler's implicit conversion limitations by using a slightly lower-level approach which involves creating a new `address payable` instance through a type conversion from the generic `address`.

Here’s how it breaks down logically. Imagine you’ve retrieved an address using a query, maybe from a mapping or an event log, and it’s currently of type `address`. You need to transfer some ether to it, which requires it to be `address payable`. The solution involves taking the 'address' value and using it as the value to initialize a new `address payable` variable. This creates the `address payable` type while maintaining the numerical identifier which makes it a proper address to send funds to.

Let's look at some code examples.

**Example 1: Basic Conversion**

This first example demonstrates the most common scenario – taking an `address` variable and converting it to `address payable` for a transfer.

```solidity
pragma solidity ^0.8.0;

contract AddressConverter {

    address public myAddress;

    constructor() {
        myAddress = address(0x5B38Da6a701c568545dCfcB03FcB875f56beddC4);
    }

    function convertAndTransfer(uint256 amount) public payable {
        // Here's the critical conversion.
        address payable recipient = payable(myAddress);
        (bool success,) = recipient.call{value: amount}("");
        require(success, "Transfer failed");
    }

    function getMyAddress() public view returns (address) {
      return myAddress;
    }
}
```

In this example, `myAddress` is initially stored as just a regular `address`. Inside `convertAndTransfer`, we convert it to `address payable` using `payable(myAddress)`. This allows us to use the `.call` method to send ether, which requires an `address payable`. The `require(success, "Transfer failed");` line is standard practice, ensuring the transfer went through.

**Example 2: Function Argument Conversion**

Sometimes, you receive an address as an argument to a function. Here's how you would handle that:

```solidity
pragma solidity ^0.8.0;

contract AddressConverterArg {

    function transferTo(address _recipient, uint256 amount) public payable {
        address payable payableRecipient = payable(_recipient);
        (bool success,) = payableRecipient.call{value: amount}("");
        require(success, "Transfer failed");
    }
}
```

In this case, `_recipient` is of type `address`. Inside the function, we convert it to `address payable` with `payable(_recipient)`. It’s important to remember you can't change the type of the input variable, `_recipient` here. Instead, a new `address payable` variable `payableRecipient` is created to hold the converted value. This method helps maintain good function design.

**Example 3: Handling Contract Addresses**

This last example shows how to deal with contract addresses, that might need to receive ether, even if they are just of type 'address':

```solidity
pragma solidity ^0.8.0;

contract Receiver {
    event Received(address from, uint256 amount);

    receive() external payable {
        emit Received(msg.sender, msg.value);
    }
}

contract AddressConverterContract {

    Receiver public receiverContract;

    constructor(address _receiverAddress) {
        receiverContract = Receiver(_receiverAddress);
    }

    function sendToContract(uint256 amount) public payable {
        address payable receiverPayable = payable(address(receiverContract));
        (bool success,) = receiverPayable.call{value: amount}("");
        require(success, "Transfer failed");
    }
}
```

Here, we first deploy a contract called `Receiver` which has a receive function that can accept ether. In the contract `AddressConverterContract`, the `receiverContract` is of type `Receiver` which has an address. The contract address is then converted to an `address` type before converting it to an `address payable` type using the `payable()` function. Again, note that `address(receiverContract)` is a temporary type conversion, and then the `payable()` operation does a proper conversion to a new variable `receiverPayable`.

**Key Considerations:**

*   **Security:** While these type conversions are technically straightforward, always verify that the address you are sending ether to is, in fact, an address that can receive it. This is especially crucial when interacting with external or untrusted contracts. A simple mistake in passing the incorrect address could mean you loose your funds with no recourse.
*   **Gas Usage:** The gas cost is generally minimal as we are just doing a simple type conversion.
*   **Error Handling:** The use of `require(success, "Transfer failed");` is essential to handle the case where transfers fail which will revert your transaction. Always use proper error handling and consider adding specific error messages.
*   **Modern Solidity:** These days the concept of `payable` has been refined a bit to make it more specific to where it is required. You should aim to only use `address payable` where you have to.

**Further Learning:**

To delve deeper, I’d recommend focusing on these resources:

1.  **"Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood:** This book is an excellent all-encompassing guide to Ethereum and smart contract development, and covers Solidity type conversions quite well in detail and in context.
2.  **Solidity Documentation:** The official Solidity documentation is always your go-to reference. Check the sections on types and function calls, it provides up-to-date information, and frequently includes code snippets that explain conversions.
3.  **Ethernaut by OpenZeppelin:** Although not a traditional resource, Ethernaut is a fantastic gamified platform for learning about smart contract vulnerabilities, which often includes challenges that force you to work with type conversions and understand the nuances of `address` and `address payable`.
4.  **The official Solidity blog.** This resource will usually have the latest updates on the language, and will often go into more complex detail about language decisions and design patterns that can help improve your understanding.

In conclusion, while converting an address to an `address payable` seems simple, the details matter. The key is understanding that it's not an implicit cast but a type conversion to a new variable. Ensure you have a solid grasp on these concepts and best practices when handling ether transfers.
