---
title: "How to store numbers securely with Chainlink VRF?"
date: "2024-12-23"
id: "how-to-store-numbers-securely-with-chainlink-vrf"
---

Alright, let's unpack storing numbers securely with Chainlink vrf. This isn't just a theoretical exercise for me; I’ve actually had to implement this in a few production systems, dealing with everything from random lottery draws to dynamic non-fungible token (nft) metadata generation. The core issue, as always with randomness in distributed systems, is ensuring that the generated value is both verifiable and impervious to manipulation. Simply storing the raw output of a vrf call directly on-chain, while seemingly straightforward, is not always the optimal—or safest—approach.

The critical component here is that Chainlink vrf provides *provable* randomness. It's cryptographically backed, meaning that any attempt to tamper with the result would invalidate the proof and be detectable. However, how you use and store this raw randomness output is where the subtleties come into play. Let’s start with why the direct storage method can be problematic, and then dive into better strategies.

My first significant project using vrf, about four years back, involved creating a decentralized virtual card game. We initially stored the *raw* generated random number on-chain. What we quickly realized is that this approach, while easy, meant that *every* operation which needed randomness had to re-execute the vrf process from scratch. This resulted in dramatically increased gas costs for even simple game mechanics. The more pressing concern though was that the raw number, while random, offered very little context and was vulnerable in certain scenarios – for example, someone could potentially observe the raw number before use, make some inferences about game state changes, and abuse the system.

Therefore, the first thing we need to consider is that what we store on the blockchain isn't always directly the *random* value itself. Often, what we’re looking to do is to use the random value as *seed* or *source* for generating derived values that are then stored or used for our applications.

Here's my first code snippet, demonstrating a simplified version of this concept in Solidity. This is how I typically handle the vrf response:

```solidity
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

contract SecureRandomNumberGenerator is VRFConsumerBaseV2 {
    uint256 public randomNumber;
    uint256 public randomNumberDerived;
    bytes32 private requestHash;

    constructor(uint64 _subscriptionId, address _vrfCoordinator, bytes32 _keyHash) VRFConsumerBaseV2(_vrfCoordinator) {
       // ... setup for VRF
    }

    function requestRandomNumber() external {
      // ... generate a random word with VRF
    }

    function fulfillRandomWords(uint256 requestId, uint256[] memory randomWords) internal override {
        require(requestHash != bytes32(0), "Request hash not set");
        require(requestHash == request.keyHash, "Incorrect Request Hash");
        requestHash = bytes32(0);
        randomNumber = randomWords[0]; // Raw vrf output

        // Example of derivation - generating a number between 1 and 100
        randomNumberDerived = (randomNumber % 100) + 1;
    }
}
```

In this example, `randomNumber` represents the raw value from the vrf call, and `randomNumberDerived` is a derived value. We use modulus to constrain it to a smaller range. This illustrates that what you *store* can be the outcome of additional operations, not directly the raw vrf value. This allows better flexibility and application-specific customization.

My second piece of advice is to understand the need for context. Storing a raw number without any context is asking for trouble. If you’re using a series of random numbers, there isn’t any real association among them and they can be easily reused. We need to add some contextual elements to the mix. For example, in a smart contract, you could tie the random number to a user’s address or an item's unique identifier. A common pattern is to use the `requestid` from the Chainlink vrf request, it's an excellent and verifiable piece of context since it is unique to that request.

Let's look at this next example of how to derive random numbers based on the requestid:

```solidity
    function fulfillRandomWords(uint256 requestId, uint256[] memory randomWords) internal override {
        require(requestHash != bytes32(0), "Request hash not set");
        require(requestHash == request.keyHash, "Incorrect Request Hash");
        requestHash = bytes32(0);
        randomNumber = randomWords[0]; // Raw vrf output

        // using request id for more contextual generation
        uint256 combined = uint256(keccak256(abi.encode(randomNumber, requestId)));
        randomNumberDerived = (combined % 100) + 1;
    }
```

Here we hash the raw random number together with the request id before using modulo. This ensures a unique derived value for every vrf request and helps in preventing precomputation, which could have been exploited for an attack.

Finally, consider the storage itself. Storing large, directly usable values on-chain might prove costly in gas and, more importantly, sometimes even unnecessary. For many applications, you might only need an *index* or a *reference* on-chain. That index can then map to a larger data set stored off-chain. For example, let's say I needed to create an application with a pool of 500 unique card artworks. I would pre-generate those artworks (and their metadata) and store them in a decentralized storage solution like ipfs. On-chain, I would only store the index of the selected artwork. That way, I reduce the on-chain gas costs by only having to manipulate the index.

Here's an illustrative code snippet showing this indirect storage mechanism. Note that I’m not including actual IPFS interaction, but focusing on the mapping of index generated by randomness:

```solidity
    uint256 public artworkIndex;
    mapping(uint256 => string) public ipfsMetadataUris;

    function setArtworkMapping(uint256 _index, string memory _uri) external {
        ipfsMetadataUris[_index] = _uri;
    }

   function fulfillRandomWords(uint256 requestId, uint256[] memory randomWords) internal override {
        require(requestHash != bytes32(0), "Request hash not set");
        require(requestHash == request.keyHash, "Incorrect Request Hash");
        requestHash = bytes32(0);
        randomNumber = randomWords[0]; // Raw vrf output

       //  Generating a random index (assuming 500 artworks)
        artworkIndex = (randomNumber % 500);
    }

    function getArtworkURI() external view returns(string memory){
         return ipfsMetadataUris[artworkIndex];
    }
```

In this simplified scenario, the smart contract stores the `artworkIndex` derived from vrf randomness. Off-chain processes can then retrieve the related metadata stored elsewhere via the `ipfsMetadataUris` mapping. This method ensures that the on-chain cost is significantly reduced.

In terms of resources to dive deeper, I highly recommend diving into the official Chainlink documentation itself, particularly the sections on vrf and secure smart contract best practices. Also, for a broader context, “Mastering Ethereum” by Andreas Antonopoulos is a valuable resource for understanding the underpinning technologies. Furthermore, for a more academic treatment of random number generation, the book "The Art of Computer Programming, Volume 2: Seminumerical Algorithms" by Donald Knuth is a classic. These should equip you with a more comprehensive view of the landscape.

Storing numbers from vrf securely is all about context, derivation, and strategic storage. Don't just blindly store the raw result; think about what you're trying to achieve, and adapt your strategy accordingly. Remember that in the distributed world, careful design and best practice are paramount to secure applications.
