---
title: "Is Chainlink's AggregatorV3Interface compatible with an OpenZeppelin upgradable contract?"
date: "2024-12-23"
id: "is-chainlinks-aggregatorv3interface-compatible-with-an-openzeppelin-upgradable-contract"
---

Okay, let's tackle this. I've spent a considerable amount of time navigating the intricacies of smart contracts, and the question of integrating Chainlink's data feeds with upgradeable contracts is definitely one I've encountered firsthand. It's not a straightforward 'yes' or 'no,' but rather a conditional 'yes, with careful considerations.'

The short answer is: yes, the `AggregatorV3Interface` can be compatible with an OpenZeppelin upgradable contract. However, you can't just blindly import and use it; there are crucial nuances to understand. The primary challenge revolves around how we handle storage and proxies in upgradable contracts. Direct, immutable storage of the Chainlink feed address in a way that's difficult to access after upgrade can quickly become a major pitfall.

Let's unpack this. When you deploy an upgradable contract using OpenZeppelin’s system, you're essentially deploying a proxy contract and an implementation contract. The proxy contract holds the state, and all interactions go through it. The implementation contract holds the actual logic. Upgrades involve deploying a new implementation contract, which the proxy then points to. This is where things can get tricky.

One of the initial issues I ran into on a past project involving a yield aggregation protocol was directly storing the Chainlink feed address within the implementation contract. This worked perfectly on initial deployment, but when we needed to roll out an update to include additional functionality and upgraded the implementation, the proxy still pointed to the old implementation and lost access to the previous feed address. This meant a lot of manual reconfigurations, and the potential for errors was significant.

To overcome this, you should store the Chainlink feed address within the proxy storage space. This ensures that when a new implementation contract is deployed, the proxy maintains access to the feed address. The implementation contract retrieves it through a designated getter, making it accessible across upgrades.

Now, let’s look at some specific code examples that illustrate the typical approach, and some pitfalls to avoid.

**Example 1: Direct Storage (Incorrect Approach)**

This demonstrates the problem: the feed address is stored directly in the implementation contract.

```solidity
// Incorrect Approach (Implementation Contract v1)
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

contract MyImplementationV1 {
    AggregatorV3Interface internal priceFeed;

    constructor(address _feedAddress) {
        priceFeed = AggregatorV3Interface(_feedAddress);
    }

    function getLatestPrice() public view returns (int256) {
        (,int256 price,,,) = priceFeed.latestRoundData();
        return price;
    }
}
```

In this example, the `priceFeed` address is stored directly in the `MyImplementationV1` contract. When this contract is upgraded, the new version will have to be initialized with the price feed address again.

**Example 2: Proxy Storage Approach (Correct)**

This example shows how we properly use storage in the proxy to handle the Chainlink feed address:

```solidity
// Proxy Contract
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/proxy/transparent/TransparentUpgradeableProxy.sol";

contract MyProxy is TransparentUpgradeableProxy{

    constructor(address _implementation, address _admin, bytes memory _data)
        TransparentUpgradeableProxy(_implementation, _admin, _data)
    {}
}
```

```solidity
//Implementation Contract v2
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";


contract MyImplementationV2 is Initializable {
     AggregatorV3Interface public priceFeed;
    
     address public proxyAddress;
     
    function initialize(address _feedAddress) public virtual initializer {
            priceFeed = AggregatorV3Interface(_feedAddress);
    }


    function getLatestPrice() public view returns (int256) {
        (,int256 price,,,) = priceFeed.latestRoundData();
        return price;
    }
    
    function setProxyAddress(address _proxyAddress) public {
        proxyAddress = _proxyAddress;
    }

    function getFeedAddress() public view returns(address) {
        return address(priceFeed);
    }


}
```

In this approach, the proxy holds the Chainlink feed address and we are initializing the implementation by passing the feed address to it. The initialization logic is contained within the implementation, but when we deploy the upgradeable proxy, we initialize the state of the implementation with the feed address during deployment, using the `initialize` function within the implementation contract.

**Example 3:  Fetching Through Proxy Contract (Slight Modification for Flexibility)**
This example illustrates how you can decouple the implementation contract further from the proxy by fetching the feed address through a call into the proxy. This enhances flexibility if you want more dynamic control.

```solidity
// Proxy contract
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/proxy/transparent/TransparentUpgradeableProxy.sol";

contract MyProxy is TransparentUpgradeableProxy{
    address public feedAddress;

    constructor(address _implementation, address _admin, bytes memory _data, address _initialFeedAddress)
    TransparentUpgradeableProxy(_implementation, _admin, _data)
    {
         feedAddress = _initialFeedAddress;
    }

    function setFeedAddress(address _newFeedAddress) external{
        feedAddress = _newFeedAddress;
    }

    function getFeedAddress() external view returns (address) {
      return feedAddress;
    }

}
```

```solidity
// Implementation Contract v3

pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";


contract MyImplementationV3 is Initializable {
    
    address public proxyAddress;
    
    AggregatorV3Interface public priceFeed;

    function initialize() public virtual initializer {
       
    }

   function setProxyAddress(address _proxyAddress) public {
        proxyAddress = _proxyAddress;
         priceFeed = AggregatorV3Interface(MyProxy(proxyAddress).getFeedAddress());
    }
    
    function getLatestPrice() public view returns (int256) {
        (,int256 price,,,) = priceFeed.latestRoundData();
        return price;
    }

}
```
In this version the proxy stores the feed address and the `MyImplementationV3` fetches the address through the proxy by calling the `getFeedAddress` function after it is initialized and it's proxy address has been set. The logic of `setProxyAddress` remains within the implementation.

**Key Considerations and Best Practices**

1.  **Proxy Storage:** The Chainlink feed address should be stored in the proxy contract’s storage, not the implementation’s storage, to survive upgrades.

2.  **Initialization:** Use the `initializer` pattern of OpenZeppelin to set up the feed address after deployment via the proxy.

3.  **Security:**  Ensure the admin for the proxy contract is a secure address, typically a multisig wallet.

4.  **Testing:**  Thoroughly test your upgrade process and contract interactions with local development networks to catch issues early.

5.  **Gas Optimization:** Consider gas optimization strategies, especially when reading from external data feeds like Chainlink.

6.  **Documentation:** Maintain clear documentation of your contract architecture, especially with regards to state variable placement.

7. **Access Control:** Carefully configure access control for any functions which allow changing the feed address. Make sure that only authorized roles are allowed.

**Recommended Resources:**

*   **OpenZeppelin Documentation:** Start with the official documentation for their Contracts library, specifically focusing on the upgradable contract patterns. The documentation covers the intricacies of proxy storage, initialization, and upgrade procedures extensively, and it’s continuously updated.

*   **Chainlink Documentation:** Refer to Chainlink’s official documentation, particularly the section on using their data feeds in smart contracts. It outlines the `AggregatorV3Interface` functionalities, the process for selecting appropriate feeds, and the associated best practices.

*   **"Mastering Ethereum" by Andreas Antonopoulos:** This book provides an in-depth look at Ethereum’s internals and offers valuable insights into how contracts and storage mechanisms function. It can give you a solid theoretical foundation.

*  **"Building Secure Blockchain Applications: Cryptography, Security, and Privacy for Distributed Ledgers" by Thomas Hardjono and Alex Pentland.** Although this book is not specific to Solidity, it is essential for understanding security and access controls with smart contracts.

In my experience, the key to integrating Chainlink and upgradable contracts successfully is a firm grasp of how proxy contracts interact with their implementations. If you pay careful attention to storage and make sure to use proper initialization techniques, the `AggregatorV3Interface` works without significant issues in an upgradeable system. Remember that it is important to implement rigorous testing, especially for contracts that have to persist storage data across upgrades.
