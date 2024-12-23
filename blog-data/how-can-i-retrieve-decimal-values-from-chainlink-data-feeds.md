---
title: "How can I retrieve decimal values from Chainlink data feeds?"
date: "2024-12-23"
id: "how-can-i-retrieve-decimal-values-from-chainlink-data-feeds"
---

Okay, let's dive into this. Retrieving decimal values from Chainlink data feeds isn’t always as straightforward as pulling a simple integer. The issue stems from how Chainlink often delivers price data, which, while fundamentally numeric, is typically represented with a defined level of precision using an integer format. This approach conserves gas costs by avoiding floating-point operations directly in the solidity code, which are notoriously expensive on the ethereum virtual machine (evm). Over my years working with decentralized applications, I've encountered this many times, particularly when building complex derivatives platforms, and learned a few tricks of the trade.

The core challenge is interpreting these integer representations to yield their actual decimal values. Chainlink data feeds commonly express values by multiplying the intended decimal value by a large power of ten. The specific power used is determined by the *decimals* field available within the feed itself. You’re not just grabbing a number; you’re grabbing an *integer representation* of a number. Understanding this is paramount to extracting accurate decimal values.

To achieve this, you need to fetch two pieces of information: the *latestAnswer* (the integer representation of the value) and the *decimals* (the exponent of ten used for scaling) from the chainlink data feed contract. Once you have these, calculating the actual decimal value becomes a relatively simple division by the appropriate power of ten. It's also essential to ensure that you implement checks for valid responses, verifying data is both recent and hasn’t deviated significantly from expectations, which Chainlink also provides methods for. Failing this, you risk using stale or corrupted data.

Now, let's explore how to achieve this using solidity. I've found it useful to create a dedicated library function for this process; It encapsulates the logic and makes its reuse very straightforward.

**Example 1: Solidity Library Function for Retrieval**

Here's a concise example of a solidity library designed to help with this:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

library ChainlinkDecimal {
    function getPrice(AggregatorV3Interface priceFeed) internal view returns (int256 decimalPrice, uint8 decimals) {
        (
            /* uint80 roundID */,
            int256 answer,
            /* uint256 startedAt */,
            /* uint256 updatedAt */,
            /* uint80 answeredInRound */
        ) = priceFeed.latestRoundData();
        
        decimals = priceFeed.decimals();

        decimalPrice = answer;
        return (decimalPrice, decimals);

    }

    function convertToDecimal(int256 answer, uint8 decimals) internal pure returns (int256) {
        int256 factor = 10**uint256(decimals);
        return answer / factor;
    }

}
```

In this example, `getPrice` retrieves the raw integer value and the decimals from a Chainlink data feed contract. The function `convertToDecimal` then performs the division to convert the integer value to its decimal representation. Note the use of `int256`, which is very important as Chainlink's answers can be negative. This library assumes you're using an `AggregatorV3Interface` compliant contract, which is typical of most Chainlink price feeds.

**Example 2: Integrating the Library into a Contract**

Now, let's illustrate how to incorporate this library into a smart contract:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import "./ChainlinkDecimal.sol";

contract MyContract {

    AggregatorV3Interface public priceFeed;

    constructor(address _priceFeedAddress) {
        priceFeed = AggregatorV3Interface(_priceFeedAddress);
    }

    function getEthPriceInUsd() public view returns (int256) {
        (int256 rawPrice, uint8 decimals) = ChainlinkDecimal.getPrice(priceFeed);
        return ChainlinkDecimal.convertToDecimal(rawPrice, decimals);
    }
}
```

Here, the `MyContract` instantiates an `AggregatorV3Interface` using a feed address. It then calls the `getPrice` and `convertToDecimal` functions from the `ChainlinkDecimal` library to fetch and process the latest eth price in usd. This approach keeps your core logic clean and avoids repeating the same retrieval logic across different parts of your project.

**Example 3: Advanced Error Handling and Data Validation**

While the previous examples are a good start, it's crucial to implement error handling, especially for scenarios where data may be invalid. Here is an enhanced version with checks for staleness and other potential issues:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import "./ChainlinkDecimal.sol";

contract MyContract {

    AggregatorV3Interface public priceFeed;
    uint256 public acceptableTimeDelta = 10 minutes; // max age of data

    constructor(address _priceFeedAddress) {
        priceFeed = AggregatorV3Interface(_priceFeedAddress);
    }

    function getEthPriceInUsd() public view returns (int256) {
         (
            /* uint80 roundID */,
            int256 answer,
            /* uint256 startedAt */,
            uint256 updatedAt,
            /* uint80 answeredInRound */
        ) = priceFeed.latestRoundData();


        if (block.timestamp - updatedAt > acceptableTimeDelta) {
            revert("Data feed is too stale"); // Use custom errors for better clarity
        }

         uint8 decimals = priceFeed.decimals();
        return ChainlinkDecimal.convertToDecimal(answer, decimals);
    }


    function setAcceptableTimeDelta(uint256 _timeDelta) public {
        acceptableTimeDelta = _timeDelta;
    }
}
```

In this version, I've added a check to ensure that the timestamp of the data isn't older than the set `acceptableTimeDelta`. If the data is too old, the function reverts. Including these kinds of checks are vital for the integrity of any system relying on external data feeds. It also includes a function `setAcceptableTimeDelta` to change that max data age.

To get a deep understanding, I'd recommend studying the official Chainlink documentation, specifically around using aggregator contracts and data retrieval. Also, looking into the source code of the Chainlink contracts themselves (usually found in the `@chainlink/contracts` npm package) is highly informative. I also found the “Mastering Ethereum” book by Andreas Antonopoulos and Gavin Wood, especially the chapter on oracles, to be an excellent resource on how these interactions typically work. Additionally, the papers on the "Chainlink" protocol's architecture provide a valuable understanding of their underlying mechanism. These resources, particularly the source code, will solidify your comprehension of Chainlink feeds.

In summary, extracting decimal values requires careful handling of the integer representations provided by Chainlink. A library for reuse is generally the most suitable implementation. Remember to handle errors appropriately and implement data validation to maintain the robustness of your smart contracts. And always verify that the data you're using is recent and accurate by considering all of the data points exposed by the aggregator.
