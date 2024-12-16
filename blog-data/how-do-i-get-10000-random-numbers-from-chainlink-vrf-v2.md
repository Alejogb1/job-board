---
title: "How do I get 10,000 random numbers from Chainlink VRF V2?"
date: "2024-12-16"
id: "how-do-i-get-10000-random-numbers-from-chainlink-vrf-v2"
---

Okay, let's talk about generating a large batch of random numbers using Chainlink VRF V2. It’s a problem I’ve faced before, particularly back in 2022 when working on a generative art project that required a substantial amount of entropy for unique creation variations. We quickly discovered that calling VRF directly for every single number was not only inefficient but would also have dramatically increased gas costs. The key here isn't just about getting random numbers; it's about doing so in a manner that's both economical and practical.

The fundamental limitation of VRF, or any on-chain randomness solution, is the cost of each request. Each call to the VRF contract, involving cryptographic operations and oracle interaction, is a fairly gas-intensive process. So, fetching 10,000 independent random values the straightforward way, i.e., issuing 10,000 separate requests, is almost certainly a bad approach. The approach needs to be optimized to make efficient use of the VRF mechanism.

There are a couple of techniques I’ve found helpful. The most effective strategy, which I generally favor due to its simplicity and gas efficiency, is to request a single random value, but then derive multiple random numbers from that single source using a deterministic method. The concept leans on the fact that a truly random value, combined with a deterministic algorithm, can effectively generate a sequence of other statistically random-seeming values.

This does not create completely independent random numbers in the strictest sense of cryptography, but for most use cases, especially those which aren’t about high-security or financial transactions, the numbers produced are sufficiently random and unpredictable. We are, essentially, using pseudorandomness derived from cryptographically secure randomness which is adequate for most applications.

Here's a high-level breakdown of the process:

1.  **Request a Single Random Value:** Initiate a single VRF request to the Chainlink contract.
2.  **Receive the Random Value:** In your fulfillment function, retrieve the random value provided by the VRF oracle.
3.  **Generate Multiple Numbers:** Using this single value as a seed, apply a deterministic hashing or calculation method to generate your desired 10,000 numbers. This step is crucial to make it cost-effective.
4. **Use the Numbers:** Then you can employ the generated random numbers within your application as needed.

Now, let's dive into some actual code examples in Solidity, illustrating different deterministic methods:

**Example 1: Using a Simple Increment and Hash**

This first example uses a simple incrementing counter along with a hashing function like `keccak256`.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";

contract RandomBatch is VRFConsumerBaseV2 {
    VRFCoordinatorV2Interface COORDINATOR;
    bytes32 keyHash;
    uint64 subscriptionId;
    uint16 requestConfirmations;
    uint32 callbackGasLimit;
    uint32 numWords;

    uint256 public randomNumberSeed;
    uint256[] public generatedRandomNumbers;

    constructor(address _vrfCoordinator, bytes32 _keyHash, uint64 _subscriptionId, uint16 _requestConfirmations, uint32 _callbackGasLimit)
    VRFConsumerBaseV2(_vrfCoordinator){
        COORDINATOR = VRFCoordinatorV2Interface(_vrfCoordinator);
        keyHash = _keyHash;
        subscriptionId = _subscriptionId;
        requestConfirmations = _requestConfirmations;
        callbackGasLimit = _callbackGasLimit;
        numWords = 1; // we need only one random word from VRF
    }

     function requestRandomNumbers() external {
        uint256 requestId = COORDINATOR.requestRandomWords(
            keyHash,
            subscriptionId,
            requestConfirmations,
            callbackGasLimit,
            numWords
        );

    }

     function fulfillRandomWords(uint256, uint256[] memory randomWords) internal override {
         randomNumberSeed = randomWords[0];
         generateNumbers();
    }


    function generateNumbers() private {
        uint256 numToGenerate = 10000;
        generatedRandomNumbers = new uint256[](numToGenerate);

        for(uint256 i = 0; i < numToGenerate; i++){
            generatedRandomNumbers[i] = uint256(keccak256(abi.encode(randomNumberSeed, i)));
        }
    }
}
```

In this example, we request a single random seed from VRF. Inside `generateNumbers()`, we then loop 10,000 times. Each iteration concatenates the random seed with an incremental value and hashes it, producing a new seemingly random number.

**Example 2: Using a Linear Congruential Generator (LCG)**

Another popular approach is employing an LCG, which is known for its simplicity and speed. Note that the quality of LCG-generated sequences depends heavily on the parameters chosen, so you should consider thoroughly researching suitable values, preferably prime numbers with good mathematical properties, before incorporating them. Here is a basic illustration:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";

contract LcgBatch is VRFConsumerBaseV2 {
    VRFCoordinatorV2Interface COORDINATOR;
    bytes32 keyHash;
    uint64 subscriptionId;
    uint16 requestConfirmations;
    uint32 callbackGasLimit;
    uint32 numWords;

    uint256 public randomNumberSeed;
    uint256[] public generatedRandomNumbers;


    uint256 public constant a = 1664525;
    uint256 public constant c = 1013904223;
    uint256 public constant m = 2**32;

    constructor(address _vrfCoordinator, bytes32 _keyHash, uint64 _subscriptionId, uint16 _requestConfirmations, uint32 _callbackGasLimit)
    VRFConsumerBaseV2(_vrfCoordinator){
        COORDINATOR = VRFCoordinatorV2Interface(_vrfCoordinator);
        keyHash = _keyHash;
        subscriptionId = _subscriptionId;
        requestConfirmations = _requestConfirmations;
        callbackGasLimit = _callbackGasLimit;
         numWords = 1;
    }

     function requestRandomNumbers() external {
         uint256 requestId = COORDINATOR.requestRandomWords(
            keyHash,
            subscriptionId,
            requestConfirmations,
            callbackGasLimit,
            numWords
        );

    }

     function fulfillRandomWords(uint256, uint256[] memory randomWords) internal override {
         randomNumberSeed = randomWords[0];
         generateNumbers();
    }

    function generateNumbers() private {
        uint256 numToGenerate = 10000;
        generatedRandomNumbers = new uint256[](numToGenerate);
        uint256 current = randomNumberSeed;

        for(uint256 i = 0; i < numToGenerate; i++){
            current = ((a * current) + c) % m;
            generatedRandomNumbers[i] = current;
        }
    }
}

```

Here, we define a constant `a`, `c`, and `m` as the LCG parameters. Within the `generateNumbers()` function, we implement the LCG iterative process to derive 10,000 numbers.

**Example 3: Using a Combination of Methods**

For added complexity or if you have more specific requirements, you might combine a hash-based approach with an LCG. This will have higher gas costs than previous examples but should still be more efficient than requesting 10,000 direct random numbers from Chainlink. For instance, you might use the initial random number to set the LCG parameters or seed, with a hash function to derive a new seed for each batch of 100 numbers, creating "mini" LCG sequences. For brevity, I will not provide a full code example here, but you can combine elements of the previous two examples. The strategy allows for greater control over the statistical properties of the derived numbers.

**Key Considerations and Further Reading:**

*   **Gas Optimization:** Always be mindful of the gas consumption of your deterministic calculations. More complex calculations will increase gas usage. The LCG tends to be the most gas-efficient approach here.
*   **Quality of Pseudorandomness:** The pseudorandom sequences derived via deterministic methods will have statistical properties that may not be perfect for all applications. For cryptographic applications, relying on deterministic output from a single seed is not sufficient.
* **Seed Management:** While the VRF's output serves as a good seed, be mindful of managing it and avoid exposing the seed publicly until you’ve generated the full sequence you need.

For deeper understanding of pseudorandom number generation, consider the following:

*   **"The Art of Computer Programming, Volume 2: Seminumerical Algorithms" by Donald E. Knuth:** This book is considered a cornerstone resource for algorithms, including pseudorandom number generators. It covers different methods, their properties, and the mathematical theory behind them.
*   **"Handbook of Applied Cryptography" by Alfred J. Menezes, Paul C. van Oorschot, and Scott A. Vanstone:** A classic text in cryptography, provides insights into cryptographic randomness, hash functions and the limitations of pseudo randomness in high-security use cases.
* **Research Papers on LCG:** There are extensive academic publications on the theory and implementation of LCG, which would provide a detailed understanding of the trade-offs in selecting specific parameters. Specifically look for papers focused on statistical testing and analysis of LCG generated sequences.

In conclusion, fetching 10,000 random numbers via direct VRF requests is impractical. Generating derived pseudo-random sequences from a single VRF seed, using methods like hashing or an LCG, is the appropriate approach for cost-effective, large-scale number generation within blockchain environments. Always evaluate your requirements concerning security and the necessity for “true” randomness before settling on a solution.
