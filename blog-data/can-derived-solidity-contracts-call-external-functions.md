---
title: "Can derived Solidity contracts call external functions?"
date: "2024-12-23"
id: "can-derived-solidity-contracts-call-external-functions"
---

Okay, let’s tackle this one. It's a question that often pops up, especially when working with complex inheritance structures in Solidity. I remember a particularly challenging project a few years back – we were building a multi-sig wallet system that involved a chain of abstract contracts and implementations. The precise question of whether a derived contract can call external functions from other contracts was absolutely critical for the architecture we envisioned. The answer, in short, is yes, a derived Solidity contract absolutely *can* call external functions of other contracts, provided the necessary access mechanisms are in place. Let me elaborate on the ‘how’ and some of the nuances involved.

The first important thing to understand is that solidity's inheritance model allows derived contracts (also called child contracts) to inherit both state variables and functions from their base contracts (or parent contracts). However, when we talk about calling *external* functions, we’re specifically referring to functions that are declared as `external` in a *different* contract. Calling an external function means interacting with that contract’s *public* interface, essentially sending a transaction to it.

Inheritance doesn't magically grant a derived contract access to external functions defined *outside* of its inheritance hierarchy. A derived contract doesn’t treat inherited contracts the same way it interacts with a deployed, distinct contract. We must establish the access by creating a contract instance and then call its method through this instance. This distinction is key to grasping the mechanics at play. You must either have an address of a previously deployed contract or the means to create a new instance.

Let's illustrate with some code. Consider the following `BaseContract` which provides an `external` function, and `DerivedContract` which will interact with it:

```solidity
pragma solidity ^0.8.0;

contract BaseContract {
    uint public baseData;

    function setBaseData(uint _data) public {
        baseData = _data;
    }

    function getBaseData() external view returns (uint) {
        return baseData;
    }
}
```

Here we have a simple contract. `setBaseData` allows changing internal state, and `getBaseData` is the method we’d like to call from a different contract. Note that `getBaseData` is marked `external`, meaning it is called as a contract-to-contract interaction.

Now, here's the `DerivedContract`, showcasing a few different ways a child can access the base function:

```solidity
pragma solidity ^0.8.0;

import "./BaseContract.sol";


contract DerivedContract {

    BaseContract public baseContract;


    constructor(address _baseContractAddress) {
        baseContract = BaseContract(_baseContractAddress);
    }

    function callBaseContractFunction() public view returns (uint) {
        // This is a direct call to an external function of an instanced contract.
         return baseContract.getBaseData();
    }

     function setAndCall(uint _data) public {
        baseContract.setBaseData(_data);
       
    }
}
```

In this snippet, the `DerivedContract` has a contract member `baseContract` which is an instance of `BaseContract`. The constructor expects an address of the deployed `BaseContract` and assigns it to the member. The `callBaseContractFunction` shows how to execute the desired `external` method.  The `setAndCall` method, while not directly calling an external method, demonstrates how you can interact with the instantiated contract through its non external functions, which ultimately has a flow-on effect when you call the external function.

Let's go through a final and perhaps more complex example. Suppose the `BaseContract` also has an `external` function that accepts complex types, for example, a struct.

```solidity
pragma solidity ^0.8.0;

contract BaseContractAdvanced {
   
    struct Data {
        uint a;
        string b;
    }
    Data public storedData;

    function setData(uint _a, string memory _b) public {
        storedData = Data(_a, _b);
    }


    function getData() external view returns(Data memory){
         return storedData;
    }

    function processData(Data memory _data) external view returns(uint) {
        //Some processing logic.
        return _data.a + uint(bytes(_data.b).length);
    }

}
```

And here is an example implementation in the derived contract:

```solidity
pragma solidity ^0.8.0;

import "./BaseContractAdvanced.sol";

contract DerivedContractAdvanced {

    BaseContractAdvanced public baseContract;


    constructor(address _baseContractAddress) {
       baseContract = BaseContractAdvanced(_baseContractAddress);
    }

    function callBaseContractProcessData(uint _a, string memory _b) public view returns (uint) {
       BaseContractAdvanced.Data memory data =  BaseContractAdvanced.Data(_a, _b);
      return baseContract.processData(data);
    }

}
```

Here, we see a derived contract instantiating the `BaseContractAdvanced` and calling the external function `processData` which requires a struct to be provided as an argument.

So, to re-iterate, derived contracts can definitely call external functions, however you must have the address of the instantiated contract and you must create a variable of that contract's type. You can’t simply call methods as if they were local ones.

A few key points to remember when working with external function calls:

1. **Gas Costs:** Calling external functions is generally more expensive than calling internal or public functions within the same contract. This is because external calls are essentially transactions to a different contract and hence incur additional gas costs.
2. **Address Management:** The derived contract needs to know the address of the contract it’s trying to call. This address can be passed during deployment (as demonstrated in our examples) or obtained through other mechanisms. In my experience, passing the contract address in the constructor is a clean and reliable pattern.
3.  **Interface Requirements:** When interacting with external contracts, the types of your calls must match the types of the target function. Solidity's type system provides helpful compile-time checks in many instances.

For a deeper dive into contract interactions, I would suggest reviewing the official Solidity documentation, particularly the sections on contract deployment, calling other contracts, and inheritance patterns. Also, "Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood is an excellent, in-depth resource that includes substantial content regarding contract interactions. Finally, the paper "A Formal Model for Ethereum Transactions" by Sergey A. Stepanov provides a rigorous look at the underlying principles of EVM operations, including transaction costs.

Hopefully, this helps clear up how derived contracts can leverage external function calls. As with many things in software development, the devil is in the detail. Understanding how contracts interact is foundational to building more complex and robust applications on the blockchain. And always test your contract deployments thoroughly!
