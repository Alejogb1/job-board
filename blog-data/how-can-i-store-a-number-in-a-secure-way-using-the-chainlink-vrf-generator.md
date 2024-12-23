---
title: "How can I store a number in a secure way using the Chainlink VRF generator?"
date: "2024-12-23"
id: "how-can-i-store-a-number-in-a-secure-way-using-the-chainlink-vrf-generator"
---

Okay, let's talk about secure number storage in conjunction with Chainlink VRF. It's a topic I've dealt with quite a bit, particularly in some early-stage blockchain gaming projects where predictable outcomes were... well, let’s just say they were actively discouraged. The core challenge, as you’ve probably already surmised, isn't simply *getting* a random number from VRF, but rather ensuring that once received, it's handled in a manner that preserves its integrity and unpredictability before it’s used by your application. This is where things can get tricky.

Firstly, let's be explicit: storing the *raw* output of a VRF call directly, in plaintext, on-chain, or even in an easily accessible off-chain database, is almost always a bad idea. While the VRF output itself is cryptographically secure at generation, its vulnerability arises *after* you receive it. Leaking it compromises everything you might have built on it, allowing malicious actors to predict future outcomes or manipulate your application to their advantage. My experience on those early projects painfully underscored how quickly things could unravel when even seemingly small security gaps like this were overlooked.

The fundamental principle is to treat this number as highly sensitive information, much like a private key, except we want to use it and then destroy it or transform it into a more palatable format. We can't simply store it and reuse it later, that breaks the fundamental requirement for each usage. Therefore, the primary method for securing a VRF-generated number involves utilizing it in a one-time manner, immediately after retrieval, and *transforming* it into a form suitable for your application. This transformation should be irreversible or sufficiently complex to make it computationally infeasible to derive the original number.

Here's what a typical approach to this might look like, along with some real code snippets (in Solidity, assuming a smart contract context, but the concepts readily apply to other environments as well):

**Snippet 1: Simple Hashing**

This is the most basic form of transformation: hashing the VRF output. It's not ideal, as a single hash may still be vulnerable in some contexts, but it’s a necessary first step. It's important to stress, I wouldn't rely solely on this for anything high stakes.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@openzeppelin/contracts/utils/Strings.sol"; // For string conversion if needed

contract ExampleVRFConsumer is VRFConsumerBaseV2 {
    uint256 public randomNumber;
    uint256 public hashedRandomNumber;


    constructor(
        uint64 _subscriptionId,
        address _vrfCoordinator,
        bytes32 _keyHash
    ) VRFConsumerBaseV2(_vrfCoordinator) {
      keyHash = _keyHash;
      subscriptionId = _subscriptionId;
    }


    function requestRandomWords() external {
        requestRandomWords(1); // Request 1 random word
    }

    function fulfillRandomWords(uint256, uint256[] memory randomWords) internal override {
        randomNumber = randomWords[0];
        hashedRandomNumber = uint256(keccak256(abi.encode(randomWords[0])));
        // Now use hashedRandomNumber; never use randomNumber directly
    }


    // Add chainlink specific variables and request function
  uint64 private immutable subscriptionId;
  bytes32 private immutable keyHash;
  uint32 private immutable callbackGasLimit = 500000;
  uint16 private immutable requestConfirmations = 3;


   function requestRandomWords(uint32 numWords) internal returns (uint256 requestId) {
        // Will revert if subscription is not enough to fulfill the request.
        requestId = requestRandomWords(
            keyHash,
            subscriptionId,
            requestConfirmations,
            callbackGasLimit,
            numWords
        );
    }
}

```

In this snippet, we request a random number, receive it in the `fulfillRandomWords` function, and then immediately hash it using `keccak256`. The `randomNumber` is never used directly; instead, the hashed value, `hashedRandomNumber`, is used in the contract’s logic. This is a better approach than using raw output, but it still isn’t perfect.

**Snippet 2: Applying a Salt**

To further obscure the value and increase security, we can add a salt. A salt is a random or unique value added before the hashing process. This makes it more difficult for attackers to precompute the hashes. This will help when we are generating several numbers within the scope of the same contract.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";

contract SaltedVRFConsumer is VRFConsumerBaseV2 {
    uint256 public saltedHash;
    uint256 public randomNumber;
    uint256 private salt;

    constructor(
        uint64 _subscriptionId,
        address _vrfCoordinator,
        bytes32 _keyHash
    ) VRFConsumerBaseV2(_vrfCoordinator) {
      keyHash = _keyHash;
      subscriptionId = _subscriptionId;
      salt = uint256(keccak256(abi.encode(block.timestamp, msg.sender))); // create initial salt
    }

    function requestRandomWords() external {
        requestRandomWords(1);
    }

    function fulfillRandomWords(uint256, uint256[] memory randomWords) internal override {
        randomNumber = randomWords[0];
        salt = uint256(keccak256(abi.encode(salt,block.timestamp)));  // Update salt
        saltedHash = uint256(keccak256(abi.encode(randomNumber, salt))); // Incorporate the salt before hashing
         // Now use saltedHash; never use randomNumber directly
    }

  uint64 private immutable subscriptionId;
  bytes32 private immutable keyHash;
  uint32 private immutable callbackGasLimit = 500000;
  uint16 private immutable requestConfirmations = 3;


   function requestRandomWords(uint32 numWords) internal returns (uint256 requestId) {
        // Will revert if subscription is not enough to fulfill the request.
        requestId = requestRandomWords(
            keyHash,
            subscriptionId,
            requestConfirmations,
            callbackGasLimit,
            numWords
        );
    }
}
```

In this example, a salt is initialized based on block timestamp and message sender and updated every call. The `randomWords[0]` is then combined with the current salt before hashing. This further complicates any reverse engineering efforts. This is still a simple operation and is still not safe enough for high security applications, but it shows the basic concepts involved.

**Snippet 3: Advanced usage via multiple derived random numbers**

For situations requiring multiple different random numbers within the same contract call, we can use the salt we just described to create different unique values from a single VRF request, each used for a different purpose.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";


contract MultiRandomConsumer is VRFConsumerBaseV2 {
    uint256 public randomNumber;
    uint256 public firstDerivedRandom;
    uint256 public secondDerivedRandom;
    uint256 private salt;

   constructor(
        uint64 _subscriptionId,
        address _vrfCoordinator,
        bytes32 _keyHash
    ) VRFConsumerBaseV2(_vrfCoordinator) {
      keyHash = _keyHash;
      subscriptionId = _subscriptionId;
      salt = uint256(keccak256(abi.encode(block.timestamp, msg.sender)));
    }

    function requestRandomWords() external {
      requestRandomWords(1);
    }

    function fulfillRandomWords(uint256, uint256[] memory randomWords) internal override {
        randomNumber = randomWords[0];
        salt = uint256(keccak256(abi.encode(salt,block.timestamp))); // Update salt
        firstDerivedRandom = uint256(keccak256(abi.encode(randomWords[0], salt, 1))); // different number for different purposes
        secondDerivedRandom = uint256(keccak256(abi.encode(randomWords[0], salt, 2))); // another different number


         // Now use firstDerivedRandom and secondDerivedRandom; never use randomNumber directly
    }


  uint64 private immutable subscriptionId;
  bytes32 private immutable keyHash;
  uint32 private immutable callbackGasLimit = 500000;
  uint16 private immutable requestConfirmations = 3;


   function requestRandomWords(uint32 numWords) internal returns (uint256 requestId) {
        // Will revert if subscription is not enough to fulfill the request.
        requestId = requestRandomWords(
            keyHash,
            subscriptionId,
            requestConfirmations,
            callbackGasLimit,
            numWords
        );
    }

}
```
This snippet generates two different random numbers, derived from the original VRF output. The salt and an additional number act as parameters to the `keccak256` function ensuring the derived randoms are different from the base random number and each other. This means that we can use these different derived randoms to different parts of our application, and if one is somehow leaked or compromised, the other is not affected.

**Key Points to Keep in Mind:**

*   **Use a secure hash function:** `keccak256` (SHA-3) is commonly used for blockchain applications. Ensure that you are using an appropriate hash function for your environment.
*   **Salt Appropriately**: The salt should be unpredictable; avoid constants, and try to make it unique to every request if possible. If storing the salt on-chain, ensure it is updated each usage to reduce predictability.
*   **Never expose the raw output**: The original number from VRF should be considered a secret. It's only safe for one-time use, immediately followed by transformation.
*   **Consider application needs**: The complexity of your transformation (salt, multiple derived values, etc.) should align with the sensitivity of the application. Highly sensitive operations require higher complexity.

**Further Reading:**

For a deeper understanding, I’d recommend diving into these resources:

*   **"Applied Cryptography" by Bruce Schneier:** A classic, albeit somewhat hefty, book covering many of the fundamentals of cryptographic techniques. It covers how to apply cryptographic techniques safely.
*   **The Chainlink documentation on VRF:** It provides critical details on the VRF mechanism and best practices, including information on secure implementation patterns. Always start with the official documentation!
*   **NIST Special Publication 800-57:** This document series, from the National Institute of Standards and Technology, provides guidelines and recommendations for key management, which is highly relevant to handling sensitive data like VRF outputs.

In summary, securing VRF outputs is about employing appropriate cryptographic techniques to transform the raw output into a form usable in the context of your specific application, while also ensuring it maintains the randomness and unpredictability required. In almost all cases, the raw output should be transformed in some way before being used within your application. This typically involves hashing with a carefully generated salt to ensure safety. Remember to always prioritize security and think through the potential risks associated with any random number generation and storage approach. It might seem tedious at first, but it’s vital for ensuring the robustness and trustworthiness of any system depending on random outcomes.
