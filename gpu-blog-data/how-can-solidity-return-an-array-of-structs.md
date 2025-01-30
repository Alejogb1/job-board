---
title: "How can Solidity return an array of structs?"
date: "2025-01-30"
id: "how-can-solidity-return-an-array-of-structs"
---
Returning arrays of structs in Solidity presents a challenge primarily due to the limitations imposed by the EVM's gas cost model and the inherent limitations on stack-based computation.  My experience optimizing smart contracts for high-throughput decentralized exchanges (DEXs) heavily involved addressing this, particularly when handling large datasets of order book entries represented as structs.  The key lies in understanding that directly returning large arrays of structs is often impractical; instead, efficient strategies rely on manipulating data within the contract and employing techniques that minimize gas consumption.

**1. Explanation: Gas Optimization Strategies**

Solidity's gas consumption model penalizes large data structures passed between functions.  The cost of returning an array is directly proportional to its size.  Attempting to return a large array of structs directly will likely result in an "out of gas" error, especially within computationally intensive operations.  To overcome this, we employ one of three core strategies:

* **Returning a single struct containing an array:** This involves embedding the array within a single struct. While this doesn't directly address the size issue, it can improve readability and organization of data passed within the contract.

* **Returning a mapping's keys or indices:** Instead of returning the array itself, the function can return a list of keys or indices referencing the data within a contract-level mapping.  This significantly reduces the returned data size, as the keys are typically smaller than the structs themselves.  The calling contract can then fetch individual struct data as needed.

* **Using events to emit data:** For situations where immediate return values are not critical, emitting the array data via events is an effective strategy.  This allows off-chain applications to access the data without incurring the high gas cost of returning it directly within the transaction.

Choosing the optimal approach depends entirely on the specific use case and the anticipated size of the data.  Smaller arrays may tolerate direct return, but larger datasets necessitate a more sophisticated gas-optimized strategy, frequently employing mappings and events.  Throughout my work with DEXs,  I've consistently observed that using events for large dataset updates coupled with efficient index-based retrieval via mappings minimizes gas consumption dramatically.


**2. Code Examples and Commentary**

**Example 1: Returning a Single Struct with an Array**

```solidity
pragma solidity ^0.8.0;

struct Order {
    uint256 id;
    uint256 price;
    uint256 quantity;
}

struct OrderBook {
    Order[] orders;
}

contract OrderBookContract {
    function getOrders() public view returns (OrderBook memory) {
        Order[] memory orders = new Order[](2); // Example, replace with actual data
        orders[0] = Order(1, 100, 10);
        orders[1] = Order(2, 200, 20);
        return OrderBook(orders);
    }
}
```

This example demonstrates embedding an array of `Order` structs within a `OrderBook` struct.  The function `getOrders()` returns an instance of `OrderBook`, containing the array. Note that this approach still suffers from gas cost limitations if the array grows large. This method is most suitable for situations where the size of the array is relatively small and predictable.  In my experience, this was useful for returning summaries or subsets of large datasets.

**Example 2: Returning Indices and Using a Mapping**

```solidity
pragma solidity ^0.8.0;

struct Order {
    uint256 id;
    uint256 price;
    uint256 quantity;
}

contract OrderBookContract {
    mapping(uint256 => Order) public orders;
    uint256[] public orderIds;

    function addOrder(uint256 id, uint256 price, uint256 quantity) public {
        orders[id] = Order(id, price, quantity);
        orderIds.push(id);
    }

    function getOrderIds() public view returns (uint256[] memory) {
        return orderIds;
    }

    function getOrder(uint256 id) public view returns (Order memory) {
        return orders[id];
    }
}
```

Here, we utilize a mapping to store the `Order` structs and an array to track their IDs. The function `getOrderIds()` returns only the array of IDs, significantly reducing the gas cost compared to returning the entire array of structs.  The `getOrder()` function allows retrieval of individual orders using their IDs. This approach is significantly more gas efficient for large datasets.  I employed this extensively within my DEX contracts for managing order books efficiently.

**Example 3: Emitting Data via Events**

```solidity
pragma solidity ^0.8.0;

struct Order {
    uint256 id;
    uint256 price;
    uint256 quantity;
}

contract OrderBookContract {
    event OrdersUpdated(Order[] orders);

    function updateOrders(Order[] memory newOrders) public {
        emit OrdersUpdated(newOrders);
    }
}
```

This showcases the use of events to communicate changes in the `Order` array. The `OrdersUpdated` event emits the entire array.  Off-chain components can listen for this event and process the data.  This is especially beneficial for situations where immediate on-chain access to the entire array isn't critical.  For large, infrequent updates in my DEX, this method proved extremely efficient, offloading data processing to external systems.


**3. Resource Recommendations**

For a deeper understanding of Solidity's gas optimization techniques, I recommend exploring the official Solidity documentation, focusing on sections detailing gas costs and best practices.  Additionally, consult advanced Solidity tutorials and books concentrating on smart contract optimization for decentralized applications.  Finally, reviewing the source code of well-established and audited smart contracts can provide valuable insights into practical implementation strategies.  These resources offer a comprehensive approach to understanding the nuances of handling large data structures within the constraints of the Ethereum Virtual Machine.
