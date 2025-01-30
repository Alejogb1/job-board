---
title: "What causes the 'invalid blockhash' error during smart contract deployment?"
date: "2025-01-30"
id: "what-causes-the-invalid-blockhash-error-during-smart"
---
The "invalid blockhash" error during smart contract deployment, typically encountered when utilizing blockhash-dependent logic within a contract's constructor, stems from the fundamental fact that a contract's constructor code executes *before* the transaction is included within a block. At the moment the constructor runs, no block has yet finalized containing the deployment transaction. This results in the `blockhash(block.number)` operation returning a zero value (often represented as a string of null bytes), or raising an error depending on the specific blockchain VM implementation and gas estimation processes.

The Solidity programming language, primarily used for Ethereum and compatible blockchains, provides `blockhash(uint blockNumber)` as an intrinsic function allowing smart contracts to access the hash of previous blocks. This function is often leveraged for pseudo-random number generation, or for verifying the integrity of data from past blocks. However, its inappropriate use during the contract's construction phase frequently leads to the described "invalid blockhash" error.

When a contract is deployed, the EVM (Ethereum Virtual Machine) executes the constructor code within a special context before actually including the deployment transaction in a block. During this pre-block execution phase, the `block.number` will either be zero or a highly unpredictable value based on the node’s current internal state, preventing any legitimate access to the hash of a finalized block. When the `blockhash` opcode is then called during this phase with an invalid (or non-finalized) `block.number`, the EVM is unable to retrieve a valid hash, typically reverting the deployment transaction. The precise error message and its nature vary slightly based on the specific tooling used (e.g., Truffle, Hardhat) and blockchain implementation details, but generally translate to an "invalid blockhash" or "block not found".

To elaborate further, the constructor's execution is part of a larger process: a user broadcasts a transaction that includes the compiled bytecode of the smart contract. Upon receiving the transaction, the EVM within a node initially simulates the transaction execution to estimate gas costs and ensure its validity. This gas estimation process includes running the constructor code; however, since this is happening before the transaction is included in a block, the block-related intrinsics provide unpredictable (or zeroed) values. Once the miner (or validator) finally includes the transaction in a block, the constructor execution is not repeated. Instead, only the resulting contract creation transaction is persisted in the block chain, which involves storing the contract's code at the designated address, with the initial storage values as defined during constructor execution.

The implications are that we must avoid any reliance on finalized block context during the contract's creation phase. Any logic requiring data from a specific block hash must be deferred to a subsequent function call after the contract's creation has been completed. This delay can be triggered by user interaction or through other contract functionality that operates post-deployment. It is important to separate initialisation logic from logic requiring finalized chain information.

To illustrate, consider the following problematic code example:

```solidity
pragma solidity ^0.8.0;

contract BadExample {
    bytes32 public randomValue;

    constructor() {
        uint256 currentBlockNumber = block.number;
        randomValue = blockhash(currentBlockNumber);  // Will revert!
    }
}
```

In this snippet, during deployment, `block.number` will likely be zero, resulting in `blockhash(0)`, or an internally generated, non-finalized `block.number` which leads to the "invalid blockhash" error. The constructor attempts to obtain the hash of the current block during its execution which does not exist because the execution is occurring before the transaction is finalized in a block.

Here is a modified version to avoid the error:

```solidity
pragma solidity ^0.8.0;

contract GoodExample {
    bytes32 public randomValue;
    bool public initialized = false;


    function initialize() public {
        require(!initialized, "Already initialized");
        uint256 currentBlockNumber = block.number;
         randomValue = blockhash(currentBlockNumber); // Works fine now
        initialized = true;
    }
}
```

In `GoodExample`, the `blockhash` call has been moved out of the constructor and into a new function `initialize`, which is called after deployment. At the time of the call, `block.number` will be a valid block number that has been finalized, and a valid block hash can be retrieved. We use a boolean flag `initialized` to prevent reinitialization. Note this function could also be triggered by other logic within the contract, not only an external call.

Finally, consider this additional example, using the constructor to set some default state and initializing the random value to some dummy value, and a function to retrieve the true value later.

```solidity
pragma solidity ^0.8.0;

contract AnotherExample {
    bytes32 public randomValue;
    bool public initialized = false;

    constructor(){
        randomValue = 0x0; // Some default value
    }

    function setRandomValue() public {
      require(!initialized, "Already initialized");
      uint256 currentBlockNumber = block.number;
      randomValue = blockhash(currentBlockNumber); // Now valid
      initialized = true;
    }

    function getRandomValue() public view returns (bytes32) {
       return randomValue;
    }

}
```

Here, the constructor performs the task of setting an initial placeholder value, and the call to obtain the blockhash is deferred to the `setRandomValue` function. This pattern demonstrates a clear separation of concern between contract initialization and blockhash-dependent actions. The `getRandomValue()` demonstrates a use case for calling this value later when it is set.

In conclusion, the root cause of the "invalid blockhash" error during smart contract deployment stems from the fact that the constructor executes prior to the transaction's inclusion in a block, thereby making it impossible to obtain a finalized block hash. The solution lies in avoiding blockhash-dependent code within the constructor and instead moving such logic to functions invoked after the contract’s successful deployment. This separation is crucial to correctly initialize and handle on-chain data. For further understanding on this topic, consulting the official Solidity documentation regarding block and transaction properties is recommended. Additionally, exploring resources which discuss smart contract deployment best practices and patterns will provide a more complete view. Reviewing practical examples in reputable open-source smart contract projects will also be advantageous.
