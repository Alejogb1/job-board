---
title: "How does uniswap upgrade its smart contract given that smart contracts are immutable once deployed?"
date: "2024-12-23"
id: "how-does-uniswap-upgrade-its-smart-contract-given-that-smart-contracts-are-immutable-once-deployed"
---

Alright, let's tackle this one. It's a common misconception that smart contracts are *entirely* immutable in a practical sense. While it's true that the code itself, once deployed, cannot be directly modified on the blockchain, there are well-established patterns to enable upgrades. I’ve certainly navigated this situation more than once in my time, particularly when working on decentralized finance protocols, which often need to evolve rapidly.

The core challenge lies in that foundational principle of immutability – how do you change the rules of the game when the rules are etched in stone? The solution, broadly speaking, involves a layer of indirection and careful planning during the contract's initial deployment. It's not magic; it's good engineering. Uniswap, like many other projects facing this issue, leverages what we often refer to as proxy patterns for this purpose.

Instead of interacting directly with a contract that contains all the logic, users interface with a *proxy contract*. This proxy contract has a single, crucial job: to delegate calls to a different contract, the *implementation contract*, which contains the actual business logic. Crucially, the address of the implementation contract is stored in the proxy and can be changed by the contract owner (usually through a multi-sig wallet, or a governance protocol).

When an upgrade is needed, you don't modify the deployed implementation contract; you deploy a *new* version. Then, the proxy contract's stored implementation address is updated to point to this new version. This seamlessly switches users to the new logic. This also preserves the storage, which remains with the proxy contract and is still accessed by the new logic. This separation of storage and logic is crucial.

Now, let's unpack how this process unfolds with some more detail and examples.

Firstly, there are a few common proxy patterns we’d consider. The transparent proxy pattern, often used in projects of this scale, uses fallback functions to delegate calls. This approach maintains compatibility with existing interfaces. I've found it’s effective when maintaining interoperability between contract versions is paramount. Another option is the Universal Upgradeable Proxy Standard (UUPS), where the proxy contract includes upgrade logic directly. The choice often comes down to the desired level of gas efficiency, security implications, and how centralized the upgrade process should be.

Let me illustrate this with code snippets. Consider a simplified version focusing on the proxy pattern (these are for illustrative purposes only, and should not be deployed to mainnet without thorough review).

**Snippet 1: Proxy Contract (Simplified Transparent Proxy)**

```solidity
pragma solidity ^0.8.0;

contract Proxy {
    address public implementationAddress;
    address public owner;

    constructor(address _implementation) {
        implementationAddress = _implementation;
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    function upgradeTo(address _newImplementation) public onlyOwner {
        implementationAddress = _newImplementation;
    }


    fallback() external payable {
        address impl = implementationAddress;
        assembly {
            calldatacopy(0, 0, calldatasize())
            let result := delegatecall(gas(), impl, 0, calldatasize(), 0, 0)
            let size := returndatasize()
            returndatacopy(0, 0, size)
            switch result
            case 0 { revert(0, size) }
            default { return(0, size) }
        }
    }
    receive() external payable {}
}

```

In this example, `Proxy` stores the address of the `implementationAddress` and uses the fallback function and inline assembly to delegate all function calls, preserving all gas and value. Importantly, there’s an owner controlled `upgradeTo` function to swap in a new implementation contract, allowing the upgrade.

**Snippet 2: Example Implementation Contract (Version 1)**

```solidity
pragma solidity ^0.8.0;


contract ImplementationV1 {

    uint256 public value;

    function setValue(uint256 _value) external {
        value = _value;
    }

    function getValue() external view returns (uint256) {
        return value;
    }

}
```

This is our first version of the implementation contract. It has a simple `value` variable.

**Snippet 3: Example Implementation Contract (Version 2 - Upgrade)**

```solidity
pragma solidity ^0.8.0;

contract ImplementationV2 {
    uint256 public value;
    string public name;

    function setValue(uint256 _value) external {
        value = _value;
    }

     function getValue() external view returns (uint256) {
        return value;
    }


    function setName(string memory _name) external {
        name = _name;
    }

    function getName() external view returns (string memory){
        return name;
    }

}
```

Here’s `ImplementationV2`. We've added a `name` string variable and corresponding getters/setters. After deploying `ImplementationV2`, the owner of the `Proxy` would call `upgradeTo` with the new address. All subsequent calls to the proxy now interact with `ImplementationV2`, yet the underlying storage remains with the proxy contract itself, preserving the `value` variable as intended from the previous version. The new `name` variable is initialized to the default of an empty string when the new implementation is swapped in.

This highlights several core principles that I’ve seen consistently in effective contract upgrades:

1.  **Separation of Concerns:** Keep your storage logic separate from your business logic via proxy contracts.
2.  **Backward Compatibility:** Ensure your updated implementation contracts maintain the same interface, wherever possible, to reduce the potential for disruption or require updates on the user side.
3. **Data Migration:** More sophisticated upgrade scenarios might necessitate a data migration function or procedure that copies data from the old storage structure to the new one during the upgrade. This can be a fairly complex step depending on the degree of change involved.
4.  **Governance:** Establishing a transparent and well-defined governance process for upgrades minimizes the risk of malicious or flawed updates. Multi-signature wallets are very commonly used to control the `Proxy` contract’s upgrade function.

Of course, this is a simplified example. In reality, Uniswap's upgrade procedures are far more nuanced. They often involve extensive testing, audit procedures, and community input to ensure that upgrades are secure and beneficial. These include features such as access control, emergency shutdown mechanisms, and other security considerations.

For a deeper dive, I'd recommend looking into the work done by Consensys on proxy patterns; their documentation on the different upgradeable contract patterns is very helpful. Additionally, the OpenZeppelin library is an excellent resource for building secure and auditable contracts, including proxy implementations, and I have personally used this many times. Specifically, they have very comprehensive documentation and examples that are worth going through in depth. Understanding the details of these patterns will give you a much clearer insight into how projects like Uniswap tackle the challenge of contract upgrades. Finally, studying the actual Uniswap contracts themselves on Etherscan (if you’re comfortable with smart contract code) is a great way to learn how these concepts are put into practical application at scale. They are open source, so it’s a fantastic educational resource. It's a complex topic, and I’ve only scratched the surface, but understanding these fundamentals will give you a solid grounding.
