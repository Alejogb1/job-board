---
title: "How to securely store a number using Chainlink VRF?"
date: "2024-12-16"
id: "how-to-securely-store-a-number-using-chainlink-vrf"
---

Alright, let's tackle this. It's a question that blends the seemingly disparate realms of on-chain randomness and secure storage, which, in my experience, often presents unique challenges. I've spent more than a few late nights debugging smart contracts, so I can shed some light on this. The core issue here is that Chainlink VRF, while fantastic for generating verifiable random numbers, doesn't handle storage itself. It's your responsibility as the developer to ensure that the number, once received from VRF, is stored securely and reliably within your smart contract. You’re basically taking a piece of extremely valuable cryptographic data – the randomness – and putting it somewhere safe, like a very digital vault.

First, let’s clarify a fundamental misconception: Chainlink VRF doesn’t provide storage; it provides *verifiable* randomness. That means your contract must handle the entire process from initiating the VRF request to safely persisting the result. We'll break this down into a few key areas: handling the callback, storage within the contract’s state, and considerations for potential vulnerabilities.

Now, I recall one project I was on, where we were building a lottery dapp. We got lazy (as we sometimes do) with storage early on. We initially stored the generated random number directly into a public variable. Huge mistake, of course, and we quickly learned a valuable lesson about potential exploits. Essentially, anyone could read that random number from the blockchain history, potentially predict subsequent draws if they were clever, or even manipulate the results.

The crucial concept here is making the storage *private* within your smart contract. Ethereum, by its nature, has transparent storage at the contract level. Therefore, the ‘privacy’ is managed using access control mechanisms and through the contract itself managing the data rather than storing it ‘off-chain’ somewhere else in the usual sense. We'll be leveraging solidity’s `private` keyword, along with functions to manage access to the generated randomness.

Here’s a basic example in solidity. This snippet focuses on the core storage aspect *after* the random number has been received by our smart contract:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";
import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";

contract SecureRandomNumber is VRFConsumerBaseV2 {
    uint256 private _randomNumber;
    bool private _randomnessFulfilled;

    event RandomNumberRequested(uint256 requestId);
    event RandomNumberReceived(uint256 randomNumber);

    VRFCoordinatorV2Interface COORDINATOR;
    uint64 s_subscriptionId;
    bytes32 keyHash;
    uint32 callbackGasLimit;

     constructor(
        uint64 subscriptionId,
        address vrfCoordinator,
        bytes32 _keyHash,
        uint32 _callbackGasLimit
     ) VRFConsumerBaseV2(vrfCoordinator)
     {
        s_subscriptionId = subscriptionId;
        COORDINATOR = VRFCoordinatorV2Interface(vrfCoordinator);
        keyHash = _keyHash;
        callbackGasLimit = _callbackGasLimit;
    }

    function requestRandomNumber() public {
        uint256 requestId = COORDINATOR.requestRandomWords(
            keyHash,
            s_subscriptionId,
            3,
            callbackGasLimit
        );
        emit RandomNumberRequested(requestId);
    }

    function fulfillRandomWords(
        uint256,
        uint256[] memory randomWords
    ) internal override {
        _randomNumber = randomWords[0]; // Store the random number
        _randomnessFulfilled = true;
        emit RandomNumberReceived(_randomNumber);
    }

    function getRandomNumber() public view returns (uint256) {
        require(_randomnessFulfilled, "Random number not yet received");
        return _randomNumber;
    }
}
```

In this code, `_randomNumber` is declared as `private`, meaning that external contracts cannot access it directly. The `getRandomNumber` function provides a *controlled* means to access this random number. Note that I’ve also included `_randomnessFulfilled`, a boolean flag to prevent anyone from trying to retrieve the number before it exists. This stops potential errors and helps with predictable contract behavior, especially when used in complex dapps where the contract interactions can happen very quickly.

The above code also includes the event `RandomNumberReceived`, which logs the value of the random number to the blockchain. While events are technically readable by anyone, they are *not* accessible for smart contract logic. Hence, the random number is not leaked for use by an exploit using only the event. Events are more for user interfaces or other off-chain applications to access the data. Events are therefore primarily designed for observation rather than reading the stored data.

Now, let's refine that example and introduce a common pattern: using the randomness as a seed and not necessarily directly exposing the raw number itself. This is useful for a variety of purposes. For instance, if you wanted to randomize the item an in-game character receives, you would not expose the random number, but the result based on the random number.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";
import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";

contract SecureSeedRandomizer is VRFConsumerBaseV2 {
    uint256 private _randomSeed;
    bool private _seedFulfilled;
    
    event SeedRequested(uint256 requestId);
    event SeedReceived(uint256 seed);

    VRFCoordinatorV2Interface COORDINATOR;
    uint64 s_subscriptionId;
    bytes32 keyHash;
    uint32 callbackGasLimit;

     constructor(
        uint64 subscriptionId,
        address vrfCoordinator,
        bytes32 _keyHash,
        uint32 _callbackGasLimit
     ) VRFConsumerBaseV2(vrfCoordinator)
     {
        s_subscriptionId = subscriptionId;
        COORDINATOR = VRFCoordinatorV2Interface(vrfCoordinator);
        keyHash = _keyHash;
        callbackGasLimit = _callbackGasLimit;
    }

    function requestRandomSeed() public {
      uint256 requestId = COORDINATOR.requestRandomWords(
            keyHash,
            s_subscriptionId,
            3,
            callbackGasLimit
        );
        emit SeedRequested(requestId);
    }

    function fulfillRandomWords(
        uint256,
        uint256[] memory randomWords
    ) internal override {
        _randomSeed = randomWords[0];
        _seedFulfilled = true;
        emit SeedReceived(_randomSeed);
    }


    function randomize(uint256 maxValue) public view returns (uint256) {
        require(_seedFulfilled, "Seed not yet generated");
        return (_randomSeed % maxValue);
    }
}
```

Here, instead of directly retrieving the random value, the contract provides a `randomize` function. This method takes a `maxValue` and returns a new number based on the modulus of the stored random number (our secure seed) and `maxValue`. This is a basic illustration; you could use the seed within more complex logic. The point is that the raw random number is *never* directly returned, adding another layer of protection.

Finally, let’s consider a scenario where you have multiple calls and need to ensure that each usage has a fresh random number. One naive implementation might be to just call `requestRandomNumber` again and store the new result. However, that can result in the contract requiring multiple VRF requests and might not be optimal in terms of gas. A better approach is to use multiple random words returned from a single request. This could be done by leveraging the `randomWords` array that `fulfillRandomWords` provides, as can be seen in the prior code snippets.

Here’s an example of how that could look:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";
import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";

contract MultiRandomValues is VRFConsumerBaseV2 {
    uint256[] private _randomValues;
    uint256 private _valuesReturned;
    bool private _randomnessFulfilled;

    event RequestForValues(uint256 requestId);
    event ValuesReceived(uint256 count);

    VRFCoordinatorV2Interface COORDINATOR;
    uint64 s_subscriptionId;
    bytes32 keyHash;
    uint32 callbackGasLimit;

     constructor(
        uint64 subscriptionId,
        address vrfCoordinator,
        bytes32 _keyHash,
        uint32 _callbackGasLimit
     ) VRFConsumerBaseV2(vrfCoordinator)
     {
        s_subscriptionId = subscriptionId;
        COORDINATOR = VRFCoordinatorV2Interface(vrfCoordinator);
        keyHash = _keyHash;
        callbackGasLimit = _callbackGasLimit;
    }

    function requestMultipleRandomValues(uint256 numWords) public {
      uint256 requestId = COORDINATOR.requestRandomWords(
          keyHash,
          s_subscriptionId,
          numWords,
          callbackGasLimit
      );
       emit RequestForValues(requestId);
    }

    function fulfillRandomWords(
        uint256,
        uint256[] memory randomWords
    ) internal override {
        _randomValues = randomWords;
        _randomnessFulfilled = true;
        _valuesReturned = 0;
        emit ValuesReceived(randomWords.length);
    }

    function getRandomValue() public returns (uint256) {
        require(_randomnessFulfilled, "Random values not yet generated");
        require(_valuesReturned < _randomValues.length, "All values have been returned");
        uint256 value = _randomValues[_valuesReturned];
        _valuesReturned++;
        return value;
    }
}
```

In this example, a request for multiple random values is initiated using `requestMultipleRandomValues`. The `fulfillRandomWords` function stores *all* the returned values into the private array `_randomValues`. The function `getRandomValue` then returns each of these values sequentially. Notice how the state variable `_valuesReturned` is used to keep track of which random value should be returned.

For further study, I recommend delving into 'Mastering Ethereum' by Andreas M. Antonopoulos, specifically the chapters on contract security and development patterns. Also, the Chainlink documentation (available directly from their site) is an invaluable resource for understanding their specific implementation details and best practices. Additionally, research papers on cryptography such as ‘A Survey of Practical Random Number Generation for Blockchains’ will further enhance the understanding of the nuances of randomness in the blockchain context.

In conclusion, storing a random number obtained from Chainlink VRF securely isn't about the VRF service itself but about responsible contract design practices within your smart contract. By using private variables, control mechanisms for access to these variables, and considering other design patterns, we can create secure and reliable systems. Always remember that security is a continuous process, not a checklist. It demands diligence and a keen understanding of the nuances involved in creating decentralized applications.
