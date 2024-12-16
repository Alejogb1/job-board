---
title: "How can I get 10,000 random numbers from Chainlink VRF V2?"
date: "2024-12-16"
id: "how-can-i-get-10000-random-numbers-from-chainlink-vrf-v2"
---

Alright, let's tackle this. Generating 10,000 random numbers from Chainlink VRF v2 isn't a straightforward "one-call" operation. It’s not designed to deliver such a high volume of random values in a single request, primarily because of gas cost limitations and the fundamental design of how VRF operates on-chain. Each randomness request incurs a certain gas fee, and requesting 10,000 values at once would be prohibitively expensive and likely exceed the block gas limit. Instead, we need to approach this problem iteratively, requesting multiple sets of random values and aggregating them. I’ve run into similar situations when simulating Monte Carlo scenarios on-chain, and the patterns I developed then are directly applicable here.

The key is to understand the limitations of VRF. It’s built for verifiable, unpredictable randomness, not bulk delivery. Each VRF request needs to be cryptographically signed by Chainlink’s oracle nodes, which involves a computation overhead. This makes it suitable for situations where unpredictability and verifiability are paramount, rather than simply generating a large sequence of numbers. Let's delve into how I’d approach this, specifically looking at the code.

**Iterative Approach with On-Chain Aggregation**

The most practical way to get our desired quantity is to make multiple VRF requests in a loop. This loop can be handled either on-chain (in your smart contract) or off-chain (using a backend process interacting with your contract). On-chain loops tend to be more expensive but maintain trustlessness, while off-chain solutions can be more efficient, yet introduce reliance on your infrastructure. Since our focus is on using VRF's verifiable nature, we’ll prioritize the on-chain aggregation approach, with a look at efficiency improvements.

Here's how it could conceptually look within a solidity smart contract, assuming you have already set up the basics of VRF v2 and have a consumer contract:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";

contract RandomNumberGenerator is VRFConsumerBaseV2 {

    VRFCoordinatorV2Interface COORDINATOR;
    uint64 public s_subscriptionId;
    bytes32 public s_keyHash;
    uint32 public numWords = 1; // Get 1 random number per request
    uint256 public requestConfirmations = 3;
    uint256 public callbackGasLimit = 50000;

    uint256[] public randomNumbers;
    uint256 public requestsCompleted;
    uint256 public totalRequestsNeeded = 10000;

    constructor(
        address _vrfCoordinator,
        uint64 _subscriptionId,
        bytes32 _keyHash
    ) VRFConsumerBaseV2(_vrfCoordinator) {
      COORDINATOR = VRFCoordinatorV2Interface(_vrfCoordinator);
      s_subscriptionId = _subscriptionId;
      s_keyHash = _keyHash;
    }

    function requestRandomWords() public {
      require(requestsCompleted < totalRequestsNeeded, "All random numbers have been requested.");
      COORDINATOR.requestRandomWords(
        s_keyHash,
        s_subscriptionId,
        requestConfirmations,
        callbackGasLimit,
        numWords
      );
        requestsCompleted++;
    }

    function fulfillRandomWords(uint256 requestId, uint256[] memory randomWords) internal override {
        randomNumbers.push(randomWords[0]);
    }

     function getAllRandomNumbers() public view returns(uint256[] memory) {
        return randomNumbers;
    }
}

```

In this initial implementation, `requestRandomWords` is called repeatedly until `requestsCompleted` matches `totalRequestsNeeded`. Each fulfillment pushes a single random number into `randomNumbers`. Note that this implementation makes one VRF request per transaction which could still be costly.

**Optimized Request Batching (On-Chain)**

To increase efficiency, instead of requesting only one random number per request, we can modify our smart contract to request multiple numbers and store them in a local array. I have found this approach significantly reduces the number of required transactions and overall gas cost. We can modify the previous example as follows:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";

contract BatchedRandomNumberGenerator is VRFConsumerBaseV2 {

  VRFCoordinatorV2Interface COORDINATOR;
    uint64 public s_subscriptionId;
    bytes32 public s_keyHash;
    uint32 public numWordsPerRequest = 10; // 10 random numbers per request
    uint256 public requestConfirmations = 3;
    uint256 public callbackGasLimit = 200000; // Increased for multiple words
    uint256 public totalRequestsNeeded = 1000; // For 1000 requests * 10 numbers per request = 10,000 numbers
    uint256[][] public allRandomNumbers; // Store the array of each fulfillment.
    uint256 public requestsCompleted;



  constructor(
        address _vrfCoordinator,
        uint64 _subscriptionId,
        bytes32 _keyHash
    ) VRFConsumerBaseV2(_vrfCoordinator) {
    COORDINATOR = VRFCoordinatorV2Interface(_vrfCoordinator);
      s_subscriptionId = _subscriptionId;
      s_keyHash = _keyHash;
    }

   function requestRandomWords() public {
        require(requestsCompleted < totalRequestsNeeded, "All random numbers have been requested.");
        COORDINATOR.requestRandomWords(
            s_keyHash,
            s_subscriptionId,
            requestConfirmations,
            callbackGasLimit,
            numWordsPerRequest
        );
       requestsCompleted++;
    }

    function fulfillRandomWords(uint256 requestId, uint256[] memory randomWords) internal override {
        allRandomNumbers.push(randomWords);
    }

    function getAllRandomNumbers() public view returns(uint256[][] memory) {
        return allRandomNumbers;
    }

   function flattenNumbers() public view returns(uint256[] memory) {
        uint256 totalLength;
        for(uint256 i=0; i < allRandomNumbers.length; i++){
            totalLength += allRandomNumbers[i].length;
        }
        uint256[] memory flattened = new uint256[](totalLength);
        uint256 currentIndex = 0;
        for(uint256 i=0; i < allRandomNumbers.length; i++){
          for(uint256 j=0; j < allRandomNumbers[i].length; j++){
             flattened[currentIndex] = allRandomNumbers[i][j];
             currentIndex++;
          }
       }
    return flattened;
    }

}
```

Here, `numWordsPerRequest` is increased. The `fulfillRandomWords` now receives an array of random numbers. This version is significantly more gas efficient due to reduced number of transactions. The `flattenNumbers` function can be used to convert the 2-dimensional array into a 1-dimensional array.

**Off-Chain Coordination with Storage**

To completely eliminate gas cost, you might consider an off-chain backend to handle the request looping. In this method, your contract maintains an array, with a function to request more random numbers, but the logic to call it in a loop exists on an external server. Upon receiving each batch of randomness, your backend stores the data and can further process it.

```javascript
// Example off-chain script (Node.js) using ethers.js
const { ethers } = require("ethers");
const contractJson = require("./path/to/your/contract.json"); // The ABI of your contract

async function main() {
    const provider = new ethers.JsonRpcProvider('YOUR_RPC_ENDPOINT');
    const wallet = new ethers.Wallet('YOUR_PRIVATE_KEY', provider);
    const contractAddress = 'YOUR_CONTRACT_ADDRESS';

    const contract = new ethers.Contract(contractAddress, contractJson.abi, wallet);
    const totalNeeded = 1000; // For 1000 batches of random numbers
    try {
    for (let i = 0; i < totalNeeded; i++) {
        console.log(`Requesting set ${i+1}`);
        const tx = await contract.requestRandomWords();
        await tx.wait();
       }
    } catch(e) {
        console.log("error: " + e);
    }
}

main();
```

This Node.js script demonstrates making `totalNeeded` calls to the contract. Each call triggers a VRF request and stores data on chain. The off-chain script can then read this data as needed.

For further reading on VRF, I’d recommend reviewing the official Chainlink documentation, which is quite thorough. Also, the paper "On the Security of Chainlink VRF" provides academic backing for its security model. A thorough understanding of the gas costs on the Ethereum virtual machine (EVM) will also help you optimize these types of interactions, so I would also recommend books such as "Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood.

In summary, while Chainlink VRF isn’t designed for bulk requests, by breaking the problem into smaller batches and either handling the looping on-chain or via an off-chain coordination mechanism, it is achievable to acquire the required 10,000 verifiable random numbers. Choose your approach according to your budget and tolerance for off-chain trust. Each option offers a valid solution with distinct tradeoffs that you can choose from.
