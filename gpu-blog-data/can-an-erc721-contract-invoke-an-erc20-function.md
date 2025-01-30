---
title: "Can an ERC721 contract invoke an ERC20 function?"
date: "2025-01-30"
id: "can-an-erc721-contract-invoke-an-erc20-function"
---
The core issue hinges on the interaction capabilities, or lack thereof, between distinct smart contract standards.  While an ERC721 contract doesn't inherently possess the functionality to directly call an ERC20 function, the interaction is achievable through intermediary mechanisms. My experience building and auditing decentralized applications (dApps) across several blockchains has shown that this is a common requirement, particularly in scenarios where NFT ownership unlocks access to utility tokens or facilitates other token-based interactions.

**1. Explanation: Indirect Invocation through a Delegate Contract**

The critical understanding here is that ERC721 and ERC20 contracts are independently deployed and do not have inherent knowledge of each other.  Direct invocation is impossible because the ERC721 contract lacks the necessary ABI (Application Binary Interface) information to interact directly with the ERC20 contract's functions. To achieve interaction, we must leverage a trusted intermediary, commonly implemented as a separate smart contract.  This delegate contract acts as a bridge, possessing the ABI information for both ERC721 and ERC20 contracts, enabling controlled interaction.

The process involves the following steps:

1. **User Interaction:** A user initiates a transaction interacting with the ERC721 contract.  This could be, for instance, burning an NFT.

2. **ERC721 Contract Invocation:** The ERC721 contract verifies the user's ownership and performs any necessary internal logic (e.g., burning the NFT).  Critically, it then calls a function within the delegate contract.

3. **Delegate Contract Invocation:**  The delegate contract, having the ABIs for both contracts, verifies the parameters passed from the ERC721 contract. It then directly calls the desired function on the ERC20 contract (e.g., `transfer`).

4. **ERC20 Contract Execution:** The ERC20 contract executes the requested function (e.g., transfers tokens to the specified address).

5. **Result:** The outcome of the ERC20 transaction is passed back through the delegate contract to the ERC721 contract, concluding the process.

This multi-stage approach ensures that the ERC20 interaction is securely managed and bound to the conditions enforced by the ERC721 contract.


**2. Code Examples with Commentary**

For clarity, I’ll present simplified examples using Solidity.  Note that these examples omit error handling and advanced features for brevity. In a production environment, extensive error handling and security audits are absolutely essential.

**Example 1:  Simplified Delegate Contract**

```solidity
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
}

interface IERC721 {
    function ownerOf(uint256 tokenId) external view returns (address);
    function burn(uint256 tokenId) external;
}

contract DelegateContract {
    IERC20 public erc20Token;
    IERC721 public erc721Token;

    constructor(address _erc20, address _erc721) {
        erc20Token = IERC20(_erc20);
        erc721Token = IERC721(_erc721);
    }

    function transferERC20OnBurn(uint256 _tokenId, address _to, uint256 _amount) external {
        address owner = erc721Token.ownerOf(_tokenId);
        require(msg.sender == owner, "Only NFT owner can call this function");
        erc721Token.burn(_tokenId);
        erc20Token.transfer(_to, _amount);
    }
}
```

This contract defines interfaces for ERC20 and ERC721, allowing it to interact with any contract conforming to these standards.  The `transferERC20OnBurn` function demonstrates the core functionality: verifying ownership, burning the NFT, and transferring ERC20 tokens.


**Example 2:  Simplified ERC721 Contract**

```solidity
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol"; //Example using OpenZeppelin
import "./DelegateContract.sol";

contract MyERC721 is ERC721 {
    DelegateContract public delegate;

    constructor(string memory _name, string memory _symbol, address _delegate) ERC721(_name, _symbol) {
        delegate = DelegateContract(_delegate);
    }

    function burnAndTransfer(uint256 _tokenId, address _to, uint256 _amount) external {
        require(_exists(_tokenId), "Token does not exist");
        require(msg.sender == ownerOf(_tokenId), "Only NFT owner can call this function");
        delegate.transferERC20OnBurn(_tokenId, _to, _amount);
    }

}
```

This contract demonstrates the interaction.  `burnAndTransfer` calls the delegate contract, delegating the ERC20 transfer.  The use of OpenZeppelin provides a robust base for the ERC721 implementation.


**Example 3:  Simplified ERC20 Contract (for completeness)**

```solidity
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol"; //Example using OpenZeppelin

contract MyERC20 is ERC20 {
    constructor(string memory name, string memory symbol) ERC20(name, symbol) {}
}
```

This is a basic ERC20 contract; its simplicity highlights that the complexity resides in the interaction mechanism, not the individual contract functionalities.


**3. Resource Recommendations**

For deeper understanding, I would recommend studying the official ERC721 and ERC20 specifications.  In addition, thoroughly reviewing Solidity documentation and exploring well-documented open-source projects employing similar interaction patterns will significantly improve your grasp of this concept.  Furthermore, understanding the OpenZeppelin library and its various components is highly beneficial for developing secure and efficient smart contracts.  Finally, engaging with community forums and seeking peer reviews before deploying any contract to mainnet is absolutely crucial.
