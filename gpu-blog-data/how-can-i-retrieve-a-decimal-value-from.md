---
title: "How can I retrieve a decimal value from an external API using Chainlink in Solidity?"
date: "2025-01-30"
id: "how-can-i-retrieve-a-decimal-value-from"
---
The crucial aspect of retrieving a decimal value from an external API via Chainlink in Solidity lies in understanding that Solidity, by design, does not inherently support floating-point numbers. Therefore, an oracle like Chainlink delivers data as an integer, requiring careful scaling and interpretation on the smart contract side to represent decimal values accurately. My experience building decentralized financial applications has shown this to be a common point of confusion and requires precise handling to avoid miscalculations.

**Understanding the Challenge and Solution**

External APIs typically return numbers in a human-readable decimal format. When using Chainlink, this data needs to be converted into an integer representation before being passed to the smart contract. The typical process involves the API provider multiplying the decimal value by a power of ten, essentially shifting the decimal point to the right. For example, a price like "12.34" might be transmitted as "123400" with an agreed-upon decimal scale of two, indicating that the client needs to divide by 100 (or 10^2) to arrive at the original value.

On the receiving end, the smart contract needs to be aware of this scaling factor to accurately reconstruct the decimal number for its calculations. This commonly involves storing the scale as a constant within the contract itself and then performing division during usage. It's vital that both sides – the API provider and smart contract – agree on the scale; otherwise, severe data interpretation errors will occur. Chainlink’s data feeds typically provide details about this scaling, often referred to as the *decimals* parameter associated with a data feed.

The challenges also involve handling large integers. Solidity has limits on the size of integers it can efficiently process, so extremely large scaled numbers could lead to overflows or truncation. Prudence requires careful assessment of potential ranges and choosing appropriate data types. `uint256` is generally the preferred choice for storing prices and similar values, but it is still critical to anticipate and avoid potential overflows when performing operations like multiplication and addition.

**Code Examples**

Here are three code examples that illustrate retrieving a decimal value from a Chainlink data feed. Each example increases in complexity, adding improvements and addressing potential edge cases.

**Example 1: Basic Retrieval with Fixed Scale**

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

contract PriceConsumerBasic {
    AggregatorV3Interface public priceFeed;
    uint256 public price;
    uint8 private constant DECIMAL_PLACES = 8;

    constructor(address _priceFeed) {
        priceFeed = AggregatorV3Interface(_priceFeed);
    }

    function getLatestPrice() public {
        (,int256 priceInt,,,) = priceFeed.latestRoundData();
        price = uint256(priceInt);
    }

    function getPrice() public view returns (uint256){
      return price / (10 ** DECIMAL_PLACES);
    }
}
```

In this basic example, `PriceConsumerBasic` retrieves the latest price from a Chainlink aggregator contract. The crucial detail here is that the `DECIMAL_PLACES` constant is fixed at 8. This assumes that the Chainlink data feed uses a scaling factor of 10^8. After receiving the scaled integer from `priceFeed.latestRoundData()`, we store it in the `price` variable and when we retrieve price with `getPrice` we divide it by the scale factor to produce the desired decimal representation, albeit still as an integer. This demonstrates the fundamental concept of scaling and division after retrieval.

**Example 2: Dynamic Scale Retrieval from Chainlink**

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

contract PriceConsumerDynamic {
    AggregatorV3Interface public priceFeed;
    uint256 public price;
    uint8 public decimalPlaces;

    constructor(address _priceFeed) {
        priceFeed = AggregatorV3Interface(_priceFeed);
        decimalPlaces = priceFeed.decimals();
    }

    function getLatestPrice() public {
        (,int256 priceInt,,,) = priceFeed.latestRoundData();
        price = uint256(priceInt);
    }

    function getPrice() public view returns (uint256){
        return price / (10 ** decimalPlaces);
    }
}
```

`PriceConsumerDynamic` improves on the first example by dynamically fetching the scaling factor or decimal places from the Chainlink data feed itself via the `decimals()` function, during initialization. This eliminates the assumption that it has a fixed scaling factor. The `getPrice` function continues to utilize the scaled price, dividing by 10 to the power of `decimalPlaces`, effectively re-constructing the decimal value. This offers a more flexible and safer approach, particularly when dealing with different types of data feeds.

**Example 3: Avoiding Division in Calculation**

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

contract PriceConsumerCalculations {
    AggregatorV3Interface public priceFeed;
    uint256 public scaledPrice;
    uint8 public decimalPlaces;
    uint256 private constant SCALE_FACTOR = 100000000;

    constructor(address _priceFeed) {
        priceFeed = AggregatorV3Interface(_priceFeed);
        decimalPlaces = priceFeed.decimals();
    }

    function getLatestPrice() public {
       (,int256 priceInt,,,) = priceFeed.latestRoundData();
        scaledPrice = uint256(priceInt);
    }

    function calculateValue(uint256 _amount) public view returns (uint256) {
        return (_amount * scaledPrice) / SCALE_FACTOR ;
    }

    function getPrice() public view returns (uint256){
      return scaledPrice;
    }
}
```

`PriceConsumerCalculations` emphasizes another critical aspect: efficiency. Instead of constantly dividing to represent the actual value, it maintains the scaled value. When calculation is required, as seen in `calculateValue()`, we multiply the input amount with the scaled price and only then do we divide by a fixed scale factor which here is `100000000`. This avoids unnecessary divisions when the scaled value is used in multiple operations. Also, we avoid recalculating `(10 ** decimalPlaces)` for each `calculateValue` call. Also note that, the `getPrice` function will return the scaled price as is which might be confusing for a developer not familiar with the design.

**Resource Recommendations**

When learning about Chainlink and Solidity, consider studying the official Chainlink documentation. It is exceptionally thorough and provides examples for various use cases. For Solidity fundamentals, focus on understanding data types, especially integer limitations and their behavior. The Solidity documentation provides precise information on these topics. Additionally, exploring community forums and discussions on platforms such as Stack Exchange can uncover practical tips and solutions that are not directly covered in official materials. Further, investigate secure coding practices specifically tailored for smart contract development, as this is critical for maintaining contract integrity and preventing vulnerabilities related to incorrect data handling. Reviewing other open source contracts that use Chainlink would be another effective learning tool. Careful study of how other developers structure their contracts, handle scaling and perform calculations offers valuable learning opportunities.
