---
title: "How to reference LINK token on a forked development network without invalid opcode errors?"
date: "2025-01-30"
id: "how-to-reference-link-token-on-a-forked"
---
The specific challenge of referencing a LINK token on a forked development network, particularly one forking from a mainnet where LINK is a frequently used ERC-677 token, stems from the discrepancies in deployment addresses between the mainnet and the forked environment. Specifically, a common error, `invalid opcode`, arises when contracts compiled against the mainnet's LINK address interact with a forked network that either lacks a LINK token deployment at the expected address or has a different implementation entirely. This occurs because the bytecode of the contract, often hardcoded or configured using mainnet addresses, isn't dynamically aware of these address shifts.

The problem is not inherent in forked networks themselves but rather in how deployment addresses are handled in Solidity and the subsequent bytecode. A smart contract, when compiled, directly encodes the specific addresses it expects to find other contracts at. For instance, a contract expecting to find a LINK token at address `0x514910771AF9Ca656af840dff83E8264EcF986CA` (the mainnet address for Chainlink’s LINK) will fail to function on a fork if the forked network does not have a contract deployed at that exact address or does not have an interface compatible with the LINK token. This usually manifests as an `invalid opcode` because the called address contains an unexpected bytecode or does not exist, causing the EVM to halt execution.

To address this, I've employed several successful techniques over the years, most fundamentally involving controlled address substitution. The core concept is to avoid direct, hardcoded references to the mainnet LINK address in the deployment pipeline targeting the forked network. Instead, the address of the LINK token contract to be used within the fork must be explicitly injected, typically through a configurable mechanism. This can be done by passing it in as a constructor argument or by using a setter function within the contract. The key is dynamic assignment rather than static, compile-time binding to the mainnet address.

Consider a simplified example of a contract that requires LINK:

```solidity
//Example 1: Contract with Hardcoded Mainnet Address (Problematic)
pragma solidity ^0.8.0;

interface LinkTokenInterface {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract RequiringLink {
    LinkTokenInterface public linkToken;
    address public linkAddress = 0x514910771AF9Ca656af840dff83E8264EcF986CA;

    constructor() {
        linkToken = LinkTokenInterface(linkAddress);
    }

    function checkBalance() public view returns (uint256) {
        return linkToken.balanceOf(address(this));
    }
    function transferToAddress(address _recipient, uint256 _amount) public returns (bool){
      return linkToken.transfer(_recipient, _amount);
    }
}
```
In the first example, the contract's `linkAddress` variable is explicitly set to the mainnet address during compilation. This approach will always lead to the "invalid opcode" error when running on a forked network. The `constructor` then uses this static address to instantiate the `linkToken` variable. This is the root problem.

The fix, which involves dynamic address assignment, can take two primary forms: constructor injection or a setter function. Let's examine an example that uses constructor injection:

```solidity
//Example 2: Contract with Constructor Injection of Address (Solution)
pragma solidity ^0.8.0;

interface LinkTokenInterface {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract RequiringLink {
    LinkTokenInterface public linkToken;

    constructor(address _linkAddress) {
        linkToken = LinkTokenInterface(_linkAddress);
    }

    function checkBalance() public view returns (uint256) {
        return linkToken.balanceOf(address(this));
    }

     function transferToAddress(address _recipient, uint256 _amount) public returns (bool){
      return linkToken.transfer(_recipient, _amount);
    }
}
```

In the second example, the contract now takes the `_linkAddress` as a constructor argument. When deploying to a forked network, the address of the LINK token deployed on that specific fork, rather than the mainnet address, is provided as an argument to the constructor. This dynamic allocation during deployment resolves the issue. The key insight here is that the bytecode generated from this second example will not contain any specific address embedded in it, relying on runtime address input.

A similar outcome can be achieved through a setter function. This method proves particularly useful if the address needs to be changed post-deployment.

```solidity
// Example 3: Contract with Setter Function for Address (Alternative Solution)
pragma solidity ^0.8.0;

interface LinkTokenInterface {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract RequiringLink {
    LinkTokenInterface public linkToken;
    address public linkAddress;

    function setLinkAddress(address _linkAddress) public {
        linkAddress = _linkAddress;
        linkToken = LinkTokenInterface(_linkAddress);
    }

    function checkBalance() public view returns (uint256) {
        return linkToken.balanceOf(address(this));
    }
    function transferToAddress(address _recipient, uint256 _amount) public returns (bool){
      return linkToken.transfer(_recipient, _amount);
    }
}
```

Here, the `setLinkAddress` function provides a means to dynamically update the `linkToken` variable after the contract has been deployed. It's essential, especially in a more complex system, to implement appropriate access control on setter functions. Generally, this setter method also provides more flexibility for updating references post deployment in cases where they may change on a local fork.

When interacting with a forked network, my preferred development strategy involves using a configurable deployment script. Such scripts generally handle the process of either deploying a mock LINK token to the forked network if one doesn’t exist, or retrieving the address of a previously deployed LINK contract. This is then passed as a constructor parameter, or via the setter, when deploying the dependent contracts. A well-architected deployment script also handles error checking and logging when an address is not found.

In addition to addressing the address resolution problem, understanding the concept of a "mock" or "dummy" ERC-677 token can be beneficial. These mock implementations provide basic ERC-677 token functionality without requiring actual Chainlink LINK, aiding testing within forked environments, or when LINK is unavailable on the fork. Several open-source implementations of ERC-677 token mocks are available; creating a lightweight one for specific needs is also viable. The important thing is to conform to the correct ERC-677 interface so that your contract will interact with your mock token as it would the real token.

For resources, I'd recommend looking into comprehensive guides on deploying smart contracts with Hardhat or Foundry, as well as guides concerning testing strategies with forked Ethereum networks. Understanding the fundamental principles of contract dependency injection is crucial. Specifically, review materials on the use of constructor arguments and setter methods for dynamic contract address configurations. In addition to this, exploration of mock contract implementations for testing purposes will help you to better understand the interface requirements and best practices. Finally, understanding of how contract bytecode is generated by Solidity compilers can often help one with better understanding the root of these issues.
