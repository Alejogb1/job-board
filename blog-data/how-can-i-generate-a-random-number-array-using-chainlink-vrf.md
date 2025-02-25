---
title: "How can I generate a random number array using Chainlink VRF?"
date: "2024-12-23"
id: "how-can-i-generate-a-random-number-array-using-chainlink-vrf"
---

, let's tackle this one. I remember back in my early days developing decentralized gaming applications, we constantly bumped up against the need for verifiable randomness, especially when dealing with in-game loot boxes or procedural generation. Initially, we tried a few naive approaches, like relying on block hashes or using a simple `Math.random()` on the client, and quickly realized these were either easily manipulated or utterly deterministic. That's when we started exploring Chainlink VRF, and it became an indispensable tool for us. So, let me walk you through how to generate a random number array using VRF, sharing some of the lessons I've picked up along the way.

The core concept behind Chainlink VRF is that it provides cryptographically secure and verifiable randomness on-chain. Unlike typical pseudo-random number generators, which are predictable, VRF utilizes a secure oracle network to generate a random value that's provable. This is crucial in applications where the integrity of randomness is paramount. We're not just talking about generating a single random number; we want an array, which adds a bit of complexity, but it's certainly achievable.

Fundamentally, you're going to make a request to a VRF Coordinator smart contract, which will in turn initiate a process of retrieving a random number. This number, a `uint256`, is what you'll then use as a seed for generating multiple numbers. We often used a simple linear congruential generator (LCG) or a similar approach on-chain to expand the original random value into our array. Now, keep in mind that gas costs are a concern; doing intensive computation directly on-chain isn’t cost-effective. However, there are ways to optimize for that, such as limiting the size of the random array or using carefully constructed seed generation techniques.

Here’s the typical workflow, broken down with some code examples in Solidity:

**1. Setting up the VRF Request:**

First, you need to initiate the request for randomness. This usually involves inheriting the `VRFConsumerBase` contract provided by Chainlink and configuring it with your VRF Coordinator address, key hash, and a subscription id (which funds your requests). This part is fairly standard. You’ll call a function like `requestRandomWords` and specify the number of random words you want (we'll use '1' initially, as we’ll expand it into our array). This process also requires a callback function, that'll be invoked when the randomness has been generated by the VRF Coordinator.

```solidity
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBase.sol";

contract RandomArrayGenerator is VRFConsumerBase {
    bytes32 public keyHash;
    uint64 public subscriptionId;
    uint256 public requestID;
    uint256 public randomness;
    uint256[] public randomArray;
    uint256 public constant MAX_ARRAY_SIZE = 50;


    constructor(address _vrfCoordinator, bytes32 _keyHash, uint64 _subscriptionId)
        VRFConsumerBase(_vrfCoordinator)
    {
        keyHash = _keyHash;
        subscriptionId = _subscriptionId;
    }

   function requestRandomArray() public {
      requestID = requestRandomWords(keyHash, subscriptionId, 1);
    }


   function fulfillRandomWords(uint256 _requestID, uint256[] memory _randomWords) internal override {
        require(_requestID == requestID, "Incorrect requestID received.");
        randomness = _randomWords[0];
        generateRandomArray(randomness);

   }

  function generateRandomArray(uint256 seed) internal {
        randomArray = new uint256[](MAX_ARRAY_SIZE);
        uint256 current = seed;

        for (uint i=0; i < MAX_ARRAY_SIZE; i++) {
            current = uint256(keccak256(abi.encode(current, i)));
            randomArray[i] = current;

        }

    }

}
```

**2. The Callback `fulfillRandomWords`:**

The `fulfillRandomWords` function is where the magic happens. Chainlink will call this once it has generated the random number based on your request. Notice that it accepts an array, `_randomWords`. Although we only requested one word in our `requestRandomWords` call, it’s still given as an array. This architecture is meant to enable efficient batch requests, which is something you should consider in more complex applications that require multiple sources of random values at once. In our example we’re taking the first (and only) value of that array. We use the received `uint256` value from Chainlink as our `seed` to our `generateRandomArray` function.

**3. Generating the Array from the Seed:**

The `generateRandomArray` function is where we expand this single random seed into an array. In this specific case we’re using a simple hash-based method to expand our seed into an array of pseudo-random numbers. The function will iteratively generate values, using the `current` value with the index and hashing those together, resulting in the next `current` value that is added to the array. This method allows us to deterministically generate a series of random numbers from our seed, without having to rely on calls to `keccak256` inside of the loop. It is not perfectly uniform but for our applications it served as a good balance between gas costs and complexity. I also recommend, to keep in mind that the `MAX_ARRAY_SIZE` constant can be adjusted according to your gas budgets.

**Considerations and Optimizations:**

*   **Gas Costs:** Generating a large array of random numbers on-chain is computationally expensive. It is paramount to benchmark the gas consumption of the code for various `MAX_ARRAY_SIZE` values. You’ll want to optimize based on the trade-offs between array size and costs.

*   **Uniformity:** The simple hash-based method is computationally efficient, but it might not produce perfectly uniformly distributed random numbers. For more statistically robust random number generation within the loop, you can explore methods like a Galois Linear Feedback Shift Register (GLFSR), although the complexity is slightly higher. Refer to "The Art of Computer Programming, Volume 2: Seminumerical Algorithms" by Donald Knuth, for more theoretical background on generating high-quality pseudo-random sequences.

*   **Seed Management:** In our example we're using a single seed from VRF. For certain applications, where you need to preserve the seed for audits, storing the original VRF value separately might be useful, while the array is generated using our method. It can be beneficial in situations where you want to re-create the random array off-chain or verify the results later on.

*   **Batch Requests:** If you find yourself needing to generate multiple random arrays, consider batching your requests using `requestRandomWords`. Requesting several random words at once is generally more gas-efficient than making individual requests each time, as the oracle call process includes a non-trivial setup overhead.

*   **Client-Side Operations:** Instead of expanding the array on-chain, if you don’t need the verification of the resulting array to be on-chain, another alternative that I used in some of the projects was to pass the single random number to the client-side application. From here, I would implement a client-side random number generator with the random number passed from the smart contract as the seed. This solution works in scenarios where it is acceptable to move a part of the randomness generation to the off-chain side, and provides significant gas savings. This is a classic trade off in blockchain development that needs to be taken into account.

```javascript
// example client-side random array generation (js)
function generateRandomArrayClientSide(seed, arraySize) {
    let current = seed;
    const randomArray = [];
    for (let i = 0; i < arraySize; i++) {
        current = parseInt(ethers.utils.keccak256(ethers.utils.solidityPack(['uint256','uint256'], [current, i])) , 16);
        randomArray.push(current);
    }
    return randomArray;
}
// ethers is used here as it is an established javascript library to interface with blockchain.
```

In addition, here's another example to demonstrate the use of different modulus to have random values from 0 to an specific bound in the array.

```solidity
// modified array generation to use modulus
function generateRandomArrayWithBounds(uint256 seed, uint256 upperBound) internal returns (uint256[] memory){
        uint256[] memory localRandomArray = new uint256[](MAX_ARRAY_SIZE);
        uint256 current = seed;

        for (uint i=0; i < MAX_ARRAY_SIZE; i++) {
            current = uint256(keccak256(abi.encode(current, i)));
            localRandomArray[i] = current % upperBound;

        }
        return localRandomArray;
    }
```
In this snippet, you can see the `localRandomArray` that can be declared locally to not update the contract state variable directly. We also show how to use the modulo operator to make sure the returned numbers do not surpass an specific upper bound. These techniques are all quite handy to adapt the random numbers to a specific use case.

**Further Reading:**

*   **Chainlink documentation:** The official Chainlink documentation is a fantastic resource for understanding the nuances of VRF integration, including different configuration options, key management, and practical examples.
*   **"Mastering Ethereum" by Andreas M. Antonopoulos:** This book provides a deep dive into Ethereum, including a detailed explanation of smart contract development, which is vital when working with complex systems like Chainlink.
*   **"Handbook of Applied Cryptography" by Alfred J. Menezes, Paul C. van Oorschot, and Scott A. Vanstone:** While not directly related to blockchain, this book serves as a valuable theoretical foundation to understand secure cryptographic random number generation.

In conclusion, generating a random number array using Chainlink VRF involves requesting secure randomness, receiving the seed value, and then expanding this seed into your desired array, keeping in mind the gas implications and the intended use case. It’s a method I’ve employed extensively, and with a bit of care and testing, it’ll prove to be a powerful tool for your decentralized applications.
