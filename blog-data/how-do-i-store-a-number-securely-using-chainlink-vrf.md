---
title: "How do I store a number securely using Chainlink VRF?"
date: "2024-12-16"
id: "how-do-i-store-a-number-securely-using-chainlink-vrf"
---

, let’s talk about securing numbers with Chainlink VRF. This isn’t as straightforward as it might initially appear, and I’ve seen a few projects stumble over this particular aspect. The core issue is that VRF, or Verifiable Random Function, isn't designed to store values; it’s designed to generate cryptographically secure, unpredictable random numbers. Using it improperly as storage creates a security gap. Over the years, I’ve worked on several projects where a similar misunderstanding led to vulnerabilities, and I learned some valuable lessons. The most important being: don’t treat a random number generator as a storage mechanism.

Here's a breakdown of why that’s problematic and how to address the challenge effectively.

First, let’s clarify what Chainlink VRF actually does. It provides a publicly verifiable source of randomness. When your smart contract requests a random number, Chainlink’s network uses a cryptographic protocol that produces both the random number and proof that this number was generated in a genuinely unpredictable way. This is crucial for applications that rely on fairness, such as blockchain games, lotteries, or any system where a predictable outcome is unacceptable. The output you receive is indeed a number, often a large unsigned integer, but its purpose is not to be *stored and used later*. Treating it as such risks exposing that number and compromising its unpredictability.

The central problem is that this output number from VRF, while random, is not secret once it has been returned. Because the verification process has to be public (for everyone to be able to verify the proof), the generated random number becomes publicly available on the blockchain once your contract calls back and receives the fulfillment. You’re essentially shouting it from the rooftops. If this is the value you're trying to store securely, you have a big problem. Malicious actors can see it, understand how it was generated (from the publicly viewable proof), and potentially exploit it in future interactions.

The common mistake is to use that random number as, say, the key to select a prize or a particular outcome from a fixed list or array, essentially hardcoding the business logic into using the number as a storage element. While this seems convenient, it opens the door for manipulation. Anyone who can read the blockchain can potentially predict and influence the system.

So, how do we properly approach this problem? Instead of storing the VRF output *directly*, we should use it to derive a secret state in a way that does not directly reveal the raw number. Here are a few approaches I’ve used in past projects with success.

**Approach 1: Utilizing the VRF Output as a Seed for Deterministic State Generation**

This method involves using the VRF output as a seed for a deterministic algorithm that generates a different, more suitable stored state. The seed itself is publicly known, but what's generated using this seed does not allow a reverse calculation for the original number and it’s not predictable, even if an attacker knows the seed and function being used.

Here’s an example using a simple mapping and hashing function (although more complex deterministic functions can be used):

```solidity
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";

contract SecureNumberStorage is VRFConsumerBaseV2 {
    VRFCoordinatorV2Interface COORDINATOR;
    uint64 s_subscriptionId;
    bytes32 s_keyHash;
    uint32 s_callbackGasLimit;

    uint256 public lastRandomNumber; // For demonstration, not storage
    mapping(uint256 => uint256) public secureState;

    constructor(
        address _vrfCoordinator,
        uint64 _subscriptionId,
        bytes32 _keyHash,
        uint32 _callbackGasLimit
    ) VRFConsumerBaseV2(_vrfCoordinator) {
        COORDINATOR = VRFCoordinatorV2Interface(_vrfCoordinator);
        s_subscriptionId = _subscriptionId;
        s_keyHash = _keyHash;
        s_callbackGasLimit = _callbackGasLimit;
    }

    function requestRandomNumber() external {
        uint256 requestId = COORDINATOR.requestRandomWords(
            s_keyHash,
            s_subscriptionId,
            1,
            s_callbackGasLimit
        );
    }

    function fulfillRandomWords(
        uint256 /*requestId*/,
        uint256[] memory randomWords
    ) internal override {
        lastRandomNumber = randomWords[0];
        uint256 stateId = uint256(keccak256(abi.encode(randomWords[0]))); //Generate a deterministic state.
        secureState[stateId] = block.timestamp; //Store block timestamp as our secured value, related to the hash.
    }
}
```

In this example, the `lastRandomNumber` variable is merely there to show the output and is not used for storage. Instead, the secure state we store is a hash derived from the random number, along with an arbitrary value. It’s not reversible but deterministically generated.

**Approach 2: Combining the VRF Output with Other Secret Data for Masking**

This involves using VRF to add a layer of unpredictable noise to your actual secret number. If your “secret” is known in any other way, it would completely undermine the purpose of the random number. The goal is to create a value from which the original secret cannot be easily extracted without knowing *both* the original secret *and* the random number used to mask it.

Here’s a simplified code example, using a simple XOR, although more sophisticated masking schemes should be considered for production:

```solidity
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";

contract MaskedNumberStorage is VRFConsumerBaseV2 {
    VRFCoordinatorV2Interface COORDINATOR;
    uint64 s_subscriptionId;
    bytes32 s_keyHash;
    uint32 s_callbackGasLimit;

    uint256 public lastRandomNumber; // For demonstration, not storage
    uint256 private mySecretNumber = 12345; // Replace with your actual secret
    uint256 public maskedSecret;


    constructor(
        address _vrfCoordinator,
        uint64 _subscriptionId,
        bytes32 _keyHash,
        uint32 _callbackGasLimit
    ) VRFConsumerBaseV2(_vrfCoordinator) {
        COORDINATOR = VRFCoordinatorV2Interface(_vrfCoordinator);
        s_subscriptionId = _subscriptionId;
        s_keyHash = _keyHash;
        s_callbackGasLimit = _callbackGasLimit;
    }


    function requestRandomNumber() external {
        uint256 requestId = COORDINATOR.requestRandomWords(
            s_keyHash,
            s_subscriptionId,
            1,
            s_callbackGasLimit
        );
    }

    function fulfillRandomWords(
        uint256 /*requestId*/,
        uint256[] memory randomWords
    ) internal override {
        lastRandomNumber = randomWords[0];
        maskedSecret = mySecretNumber ^ randomWords[0];
    }
}
```

Here, `mySecretNumber` is our "secret" and `maskedSecret` is stored after being XORed with the random number. Anyone observing `maskedSecret` will not know the original secret without knowing the random number. Even then, XOR can easily be reversed with the same seed so more sophisticated methods can be used with the same idea. The secret here is also stored in the contract, which is publicly visible, so consider this approach for cases where the secret is generated externally or derived from user input, and then masked with the random number.

**Approach 3: Using a Commit-Reveal Scheme**

This approach is more complex but offers a higher level of security in some specific contexts. It involves committing to a value (or action) in advance using a secret key or a cryptographic hash, and revealing it later with the VRF output. The commitment phase hides the chosen value until the reveal phase. This can be useful in cases where you want to decide on something now but only reveal it later, ensuring it's not influenced by future events or by your own actions. This involves pre-committing to an action or decision using a hash of your secret and the random number, then revealing that secret and action later.

Here’s a rough example:

```solidity
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";

contract CommitRevealScheme is VRFConsumerBaseV2 {
    VRFCoordinatorV2Interface COORDINATOR;
    uint64 s_subscriptionId;
    bytes32 s_keyHash;
    uint32 s_callbackGasLimit;

    uint256 public lastRandomNumber; // For demonstration, not storage
    bytes32 public commitment;
    uint256 private secretValue;
    uint256 public revealedValue;


    constructor(
        address _vrfCoordinator,
        uint64 _subscriptionId,
        bytes32 _keyHash,
        uint32 _callbackGasLimit
    ) VRFConsumerBaseV2(_vrfCoordinator) {
        COORDINATOR = VRFCoordinatorV2Interface(_vrfCoordinator);
        s_subscriptionId = _subscriptionId;
        s_keyHash = _keyHash;
        s_callbackGasLimit = _callbackGasLimit;
    }

    function preCommit(uint256 _secretValue) external {
        secretValue = _secretValue;
        commitment = keccak256(abi.encode(_secretValue, 0)); // initial commit
    }

    function requestRandomNumber() external {
        uint256 requestId = COORDINATOR.requestRandomWords(
            s_keyHash,
            s_subscriptionId,
            1,
            s_callbackGasLimit
        );
    }

    function fulfillRandomWords(
        uint256 /*requestId*/,
        uint256[] memory randomWords
    ) internal override {
        lastRandomNumber = randomWords[0];
        revealedValue = secretValue + randomWords[0];
        commitment = keccak256(abi.encode(secretValue, lastRandomNumber)); // Finalize reveal.
    }
}

```

In this case, `secretValue` is first committed with an initial commitment. Later, the VRF output is used to finalize the commitment and reveal the `revealedValue` in a way where anyone can verify that the revealed value corresponds to the initial commitment and the random number, but is not easily predictable.

In all these approaches, the key point is that the VRF output is never directly used for storage. It acts as a source of randomness for more sophisticated state generation or masking.

For further reading, I would highly recommend looking at "Mastering Ethereum" by Andreas Antonopoulos and Gavin Wood for an in-depth look at smart contract security and design patterns. Also, research papers detailing cryptographic commitments and masking techniques, readily available on IEEE Xplore or ACM Digital Library, would be very beneficial for more in-depth knowledge of these concepts. Understanding these fundamental building blocks will help in building robust and secure applications. Remember, the key is not to rely on the perceived secrecy of a random number, but to use that randomness to create secure and verifiable state.
