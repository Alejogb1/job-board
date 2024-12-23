---
title: "Why are only reference types stored in memory in Solidity?"
date: "2024-12-23"
id: "why-are-only-reference-types-stored-in-memory-in-solidity"
---

,  Memory management in Solidity, particularly its handling of reference types versus value types, has been a recurring point of discussion – and something I’ve had to troubleshoot a few times in past projects involving complex smart contracts. The core of the matter lies in how the Ethereum Virtual Machine (EVM) operates and how Solidity's type system is built upon it. Let me unpack why only reference types find themselves residing in memory.

First, it’s crucial to differentiate between *value types* and *reference types*. Value types in Solidity (like `uint`, `bool`, `address`, `enum`) are stored directly within a variable’s storage space. When you assign a value type to another variable, you are essentially creating a copy of that value. Changes to one variable do not affect the other. Reference types, on the other hand, including `arrays`, `structs`, and `mappings`, are inherently more complex. These do not store the actual data in the variable location. Instead, they store a *pointer* (a memory address) referencing where the data is held.

Now, why does Solidity opt to only store reference types in memory? The short answer is efficiency and the EVM's limitations with its stack-based architecture. In Solidity, there are three main storage locations: `storage`, `memory`, and `calldata`. `Storage` is persistent storage on the blockchain. It's expensive in terms of gas cost for modification but persists across transactions. `Calldata` is a read-only location used to pass function arguments and is cheaper than `storage`. `Memory` is a temporary, volatile space used for storing data during the execution of a function; it gets cleared at the end of each transaction.

When you’re working with complex data structures (reference types), copying the entire structure every time you perform an assignment or pass it as a function argument would be incredibly inefficient in terms of both gas and memory. Imagine copying a large array multiple times – it would quickly make transactions prohibitively costly. This is where memory's address-based storage for reference types becomes invaluable. By only storing a memory address, assignments and function argument passing can be achieved by merely copying the memory address itself, which is a significantly cheaper operation.

Furthermore, the EVM has a limited stack size. Value types are small and fit onto the stack readily. However, if complex reference types were allocated directly on the stack (similar to how value types are handled), it could easily exceed its capacity, leading to a stack overflow error. Storing the reference to a location in memory avoids this limitation. Memory allocation in the EVM, while also constrained, tends to be more generous in that it expands as needed during function execution. Thus, it makes logical sense to use `memory` as the allocated storage space for these reference types.

To illustrate these concepts, let's look at some code snippets.

**Example 1: Value type vs Reference type assignment.**

```solidity
pragma solidity ^0.8.0;

contract TypeComparison {

    uint public valueType;
    uint[] public referenceType;

    constructor() {
        valueType = 10;
        referenceType = [1, 2, 3];
    }

    function modifyTypes() public {
        uint copiedValueType = valueType;
        uint[] memory copiedReferenceType = referenceType;

        copiedValueType = 20;
        copiedReferenceType[0] = 10;

        // valueType remains 10
        // referenceType[0] becomes 10

        // Notice here we specifically need to define
        // copiedReferenceType as memory, it does not 
        // take storage by default like valueType.
        // This is because a copy of the array must
        // be made in memory if you intend to modify it
        // independently of the storage location
    }
}

```
In this first example, we have two variables: a `uint` (value type) and a `uint[]` (reference type). When we copy the `valueType` to `copiedValueType`, it results in a copy. Modifications to `copiedValueType` do not affect the original `valueType`. On the other hand, when copying `referenceType` into `copiedReferenceType` the copy is only a pointer to the same data; so modifications to the copy *do* modify the original data as they refer to the same location in memory. Note the necessary `memory` keyword in the second copy to achieve a true copy of a reference type’s pointer in memory; otherwise, a read reference is made, making changes in copiedReferenceType also affect the original.

**Example 2: Function arguments and memory.**

```solidity
pragma solidity ^0.8.0;

contract MemoryUsage {

    uint[] public myArray;

    constructor() {
        myArray = [1, 2, 3];
    }

    function modifyArray(uint[] memory _array) public {
        _array[0] = 100;
    }

    function callModifyArray() public {
        uint[] memory tempArray = myArray;
        modifyArray(tempArray);

        // myArray[0] becomes 100
        // tempArray's pointer value now points to
        // the same data as myArray
    }
}
```
Here, the `modifyArray` function takes a `uint[] memory` as an argument, specifying that it expects the array to be passed in memory. Because the function argument is also a pointer, it modifies the original data in storage. The `callModifyArray` function creates a `memory` variable `tempArray`, which references the `myArray` data. Passing `tempArray` to `modifyArray` changes the original `myArray`. This further emphasizes how reference types work with memory addresses rather than direct value copies.

**Example 3: Returning a modified array.**

```solidity
pragma solidity ^0.8.0;

contract ReturnModifiedArray {

    uint[] public myArray;

    constructor() {
        myArray = [1, 2, 3];
    }
    
    function modifyAndReturn(uint[] memory _array) public pure returns (uint[] memory) {
       
        uint[] memory newArray = _array;
        newArray[0] = 100;

       return newArray;
    }

    function checkArray() public {
       uint[] memory returnedArray = modifyAndReturn(myArray);
       //returnedArray[0] will equal 100, the data of
       // myArray will remain unchanged.
       
       uint first = myArray[0];
       //first will equal 1
    }
}
```

In this case, the function `modifyAndReturn` is marked as pure, indicating it does not change contract state. Inside the function, a new pointer `newArray` is created referencing the `_array` data, and then that data is modified, then returned. However, this data is entirely isolated within the scope of the function's memory, and no changes to the original `myArray` are ever made. It's a new pointer, which is then returned and assigned to `returnedArray` in the `checkArray` function.

In conclusion, only reference types are stored in memory in Solidity primarily due to gas cost considerations and the EVM’s architectural limitations. The address-based storage approach in memory allows for efficient management of complex data structures and prevents issues related to the limited stack size. While initially this might appear counterintuitive, it’s a fundamental design aspect to optimize gas consumption and ensure the feasibility of complex smart contracts on the Ethereum platform. Understanding this fundamental distinction is crucial when writing and debugging Solidity code.

For a deeper dive, I’d recommend consulting "Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood; it covers these concepts extensively. Also, the official Solidity documentation is an invaluable resource and should be considered a primary point of reference. Finally, I would suggest delving into the Yellow Paper which details the low-level specifics of the EVM itself, providing a fundamental understanding of its mechanics. These resources should offer you a broader perspective and a more nuanced understanding of how memory management operates within the context of smart contract development.
