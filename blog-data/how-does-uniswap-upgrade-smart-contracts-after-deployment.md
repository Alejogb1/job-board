---
title: "How does Uniswap upgrade smart contracts after deployment?"
date: "2024-12-23"
id: "how-does-uniswap-upgrade-smart-contracts-after-deployment"
---

Okay, let's talk about Uniswap's upgrade mechanisms. It's a frequently asked question, and for good reason; smart contract upgrades are inherently tricky, especially when dealing with decentralized, immutable ledgers like Ethereum. From the trenches, I've personally navigated the complexities of upgrade strategies several times on various blockchain projects, and Uniswap's approach is certainly one worth analyzing.

One key thing to understand is that smart contracts, once deployed, are technically immutable. We can’t just directly change the code running at a specific address. So, how do you "upgrade" something that can’t be altered? The solution relies on a clever combination of proxy contracts and a delegate call mechanism.

Imagine you've got two buildings: a public facade, easily accessible, and a private office hidden inside. The public facade is your proxy contract. It's the address users interact with. The private office is your implementation contract, which holds the actual business logic. When you want to upgrade, you don't tear down the facade; instead, you change the door in the back to point to a *new* private office, one containing updated code. This is roughly the architecture Uniswap leverages.

The proxy contract stores data, like token balances and pool information, and it delegates all function calls to the implementation contract using the `delegatecall` opcode. `delegatecall` is absolutely crucial here. Unlike a normal function call, `delegatecall` executes the code *within the context of the calling contract*. This means the code from the implementation contract uses the storage of the proxy contract. When an upgrade is needed, a new implementation contract is deployed, and the proxy contract’s internal pointer is updated to point to the new address.

Let me walk you through it with some simplified code snippets. First, let's consider our *proxy contract* (simplified for illustration):

```solidity
pragma solidity ^0.8.0;

contract Proxy {
    address public implementationAddress;
    address public admin;

    event ImplementationUpdated(address newImplementation);

    constructor(address _implementationAddress) {
        implementationAddress = _implementationAddress;
        admin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "Not authorized");
        _;
    }

    function upgradeImplementation(address _newImplementation) public onlyAdmin {
        implementationAddress = _newImplementation;
        emit ImplementationUpdated(_newImplementation);
    }

    fallback() external payable {
        address _impl = implementationAddress;
        assembly {
            let ptr := mload(0x40)
            calldatacopy(ptr, 0, calldatasize())
            let result := delegatecall(gas(), _impl, ptr, calldatasize(), 0, 0)
            let size := returndatasize()
            returndatacopy(ptr, 0, size)
            switch result
            case 0 { revert(ptr, size) }
            default { return(ptr, size) }
        }
    }
}
```

This `Proxy` contract stores the current implementation address and the admin. It has a function, `upgradeImplementation`, that allows the admin to change the implementation address. The critical part is the `fallback()` function. This assembly code is using `delegatecall` to forward all calls to the address stored in `implementationAddress`, using the same storage.

Next, here’s a very simple *implementation contract* (again, simplified):

```solidity
pragma solidity ^0.8.0;

contract ImplementationV1 {
    uint256 public value;

    function setValue(uint256 _newValue) public {
        value = _newValue;
    }

    function getValue() public view returns (uint256) {
        return value;
    }
}
```

This `ImplementationV1` contract has a simple value and some associated functions. Initially, the proxy would point to this contract. Now, imagine we want to update our logic—we’ll create `ImplementationV2`:

```solidity
pragma solidity ^0.8.0;

contract ImplementationV2 {
   uint256 public value;
    string public message;

    function setValue(uint256 _newValue, string memory _message) public {
        value = _newValue;
        message = _message;
    }

    function getValue() public view returns (uint256) {
        return value;
    }

     function getMessage() public view returns(string memory) {
        return message;
    }

}
```
`ImplementationV2` introduces a string `message`, and adds that to the `setValue` function. Crucially, the `value` field, because it’s a storage variable, lives in the storage of the proxy contract. Because we are calling using `delegatecall` in the proxy contract, any updates we make to the `value` variable in either `ImplementationV1` or `ImplementationV2` will be modifying the same storage location inside the `Proxy` contract, thereby preserving state during the upgrade.

After deploying `ImplementationV2`, an admin would then call the `upgradeImplementation` function on the `Proxy` contract, updating its `implementationAddress` variable to point to `ImplementationV2`. No user interaction is needed and their interactions seamlessly transition to using the new implementation logic.

This is, of course, a vastly simplified version of Uniswap's upgrade pattern. Uniswap uses a sophisticated pattern of proxies and implementation contracts, with multiple layers of abstraction, and employs custom upgrade mechanisms, such as a time-lock on proposed updates. Further, different Uniswap versions adopt distinct patterns. For example, Uniswap V3 uses a "diamond" pattern for its core contracts, where functionality is divided into multiple modules, which can be upgraded independently.

This approach of using a proxy contract and `delegatecall` is incredibly powerful because it allows for zero-downtime upgrades. Users continue to interact with the same contract address while the underlying implementation is changed. It requires careful consideration during development, though. Storage layouts must be compatible between implementation contracts, and proper admin control is essential to avoid security vulnerabilities. If the storage layout changes in an incompatible manner, existing data can be corrupted.

For anyone looking to delve deeper into this topic, I highly recommend these resources:

*   **"Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood**: This book provides a comprehensive understanding of Ethereum, including details on smart contract development and upgrade patterns. Specifically, look into the sections discussing proxy patterns and `delegatecall`.
*   **Ethereum Improvement Proposals (EIPs):** Examining EIP-1967 ("Standard Proxy Storage Slots"), EIP-1822 ("Upgradeable Proxy Standard"), and EIP-2535 ("Diamond Standard") provides excellent technical insight.
*   **"Solidity Programming Essentials" by Ritesh Modi:** This book includes detailed examples of proxy contracts and `delegatecall`, as well as general smart contract coding best practices. The section on upgradeability and deployment patterns is particularly useful.
*   **Uniswap's Official Documentation and Source Code:** Reviewing the Uniswap whitepaper and exploring the actual Solidity code on GitHub offers firsthand insight into their specific architectural choices and upgrade patterns used. The V3 contracts are particularly interesting in this regard.

These resources should be more than enough to give any aspiring solidity developer the foundations needed to navigate the complex subject of smart contract upgrades. The upgrade mechanism behind Uniswap is not simple, and understanding how it works is absolutely essential for anyone seriously involved in decentralized finance. It’s a testament to the ingenuity of the Ethereum ecosystem that such seemingly intractable problems can be solved with careful engineering and design.
