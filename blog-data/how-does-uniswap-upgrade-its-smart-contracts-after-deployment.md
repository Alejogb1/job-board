---
title: "How does Uniswap upgrade its smart contracts after deployment?"
date: "2024-12-16"
id: "how-does-uniswap-upgrade-its-smart-contracts-after-deployment"
---

Okay, let’s tackle this. It’s a question I've personally navigated several times, particularly back when we were scaling a DeFi platform that leveraged heavily on AMM technology—similar challenges, similar solutions. Upgrading smart contracts, especially after deployment, is a complex issue in the blockchain world, primarily because of the immutable nature of these contracts once they're on the chain. Uniswap, like many decentralized protocols, doesn't directly *change* the deployed code, but rather, it deploys *new* contracts and establishes a migration strategy to move users and liquidity over. It's not a simple replacement. This needs a structured, secure, and very deliberate approach.

The core concept is built around the idea of *proxy contracts*. The proxy contract serves as a sort of intermediary, a front-facing contract that users interact with. This proxy contract doesn't house the core logic itself. Instead, it delegates all function calls to an *implementation* contract, often referred to as the "logic" contract. The beauty of this setup is that you can swap the implementation contract while the proxy contract remains at the same address. This preserves the persistent address that users rely on, without breaking existing integrations. Think of it as changing the engine of a car while keeping the chassis and its known registration unchanged.

The upgrade process typically involves these key steps:

1. **Development and Testing of New Implementation Contract:** The first, and arguably most crucial step, is developing the enhanced or modified logic contract. This process involves meticulous planning, development, and testing. We're talking thorough unit tests, integration tests, and even formal verification to check that the contract behaves as expected and doesn’t introduce unforeseen vulnerabilities. When working on our DeFi project, this stage routinely went through multiple rounds of peer review and audits to minimise the risk. We used tools like Mythril and Slither for static analysis, and the Brownie framework alongside Ganache for setting up local test networks.

2. **Deployment of the New Implementation Contract:** Once the new implementation contract is deemed secure and functional, it’s deployed on the blockchain. This is where having a sound deployment pipeline becomes paramount. We would use CI/CD tools like GitHub Actions to ensure each deployment was reproducible and auditable. However, at this point, it's still inactive because the proxy contract is still pointing to the old logic. This careful, two-step deployment is key for managing a clean transition.

3. **Proxy Contract Update:** The proxy contract’s administrative functions—usually accessible only to the contract’s owner or designated multisig—are used to point to the *new* deployed implementation contract. This step is pivotal and carefully executed. The proxy contract stores the address of the implementation contract, and updating it with the new address routes function calls to the latest version.

4. **User Migration (If Needed):** In some instances, such as when significant data structures or storage layouts change, users may need to initiate an action to transition their data or liquidity to the new contract. This might involve a specific migration function to allow users to withdraw from the older contract and deposit to the new one. This phase requires clear communication with the user base to facilitate a smooth and understandable migration. The whole purpose is to make this transition as seamless and straightforward as possible.

Let’s break this down further with some hypothetical code examples to illustrate the core components. Note that these are simplified for clarity, and real-world contracts are far more complex, but these examples give you the essential idea:

**Example 1: Proxy Contract**

```solidity
pragma solidity ^0.8.0;

contract Proxy {
    address public implementationAddress;
    address public owner;

    constructor(address _implementationAddress) {
        implementationAddress = _implementationAddress;
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner.");
        _;
    }

    function setImplementationAddress(address _newImplementationAddress) external onlyOwner {
        implementationAddress = _newImplementationAddress;
    }

    fallback() external payable {
        address _implementationAddress = implementationAddress;
        assembly {
            calldatacopy(0, 0, calldatasize())
            let result := delegatecall(gas(), _implementationAddress, mload(0), calldatasize(), 0, 0)
            let size := returndatasize()
            returndatacopy(0, 0, size)
            switch result
            case 0 { revert(0, size) }
            default { return(0, size) }
         }
    }
}
```

This proxy contract holds a single address `implementationAddress`. The `fallback` function uses `delegatecall` to forward any transaction to that address, thus ensuring that all the calls are executed in the context of the proxy’s storage but with the logic defined in the implementation. Crucially, the implementation contract’s methods execute within the storage context of this proxy contract. This is vital for transparent contract upgrading.

**Example 2: Initial Implementation Contract**

```solidity
pragma solidity ^0.8.0;

contract ImplementationV1 {
    uint256 public version = 1;
    uint256 public storedData;

    function setStoredData(uint256 _data) external {
        storedData = _data;
    }

    function getStoredData() external view returns (uint256) {
        return storedData;
    }
}
```

This is a simple initial implementation contract. It holds and lets users update a piece of `storedData`.

**Example 3: New Implementation Contract**

```solidity
pragma solidity ^0.8.0;

contract ImplementationV2 {
    uint256 public version = 2;
    uint256 public storedData;
    string public metadata;

     function setMetadata(string memory _metadata) external {
        metadata = _metadata;
     }


    function setStoredData(uint256 _data) external {
        storedData = _data;
    }

    function getStoredData() external view returns (uint256) {
        return storedData;
    }

      function getMetadata() external view returns(string memory){
        return metadata;
    }
}
```

This new implementation contract (`ImplementationV2`) adds a new `metadata` variable and a function `setMetadata`. Crucially, it preserves the old functions. This demonstrates a basic upgrade where a functionality is added to the implementation logic without breaking existing user interactions.

The sequence of a typical upgrade is thus:

1. Deploy `Proxy`, `ImplementationV1`, and use the `Proxy` constructor to point to `ImplementationV1`.
2. Users now interact with `Proxy`’s address, but the methods called are those from `ImplementationV1`.
3. Later, `ImplementationV2` is deployed.
4. The owner of the `Proxy` calls `setImplementationAddress` using the address of `ImplementationV2`.
5. From this moment, when users call functions via the proxy, they interact with the methods from `ImplementationV2`. The storage remains with the `Proxy`, hence, even though the code is upgraded, all the existing state is preserved.

For further reading, I highly recommend diving into the following resources:

*   **"Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood**: A comprehensive book that provides an in-depth understanding of Ethereum, including detailed explanations of smart contracts, proxy patterns, and upgrade strategies.
*   **"Solidity Programming Essentials" by Ritesh Modi**: This is a great resource to get into solidity and the best practices when coding with it.
*   **EIP-1967:** This is a great place to read more about the standard proxy storage slot.
*   **The ZeppelinOS project**: It contains lots of open-source code that shows how upgradable contracts are implemented in practice.

It's important to understand that there’s no one-size-fits-all approach to upgrades. The complexity depends heavily on the protocol, the degree of changes, and risk appetite. You have to carefully weigh different design patterns. I hope this breakdown offers some clarity. It's a challenging area, but understanding the principles behind it is key to building scalable and robust decentralized applications.
