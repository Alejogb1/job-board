---
title: "What caused the Chainlink VRF v2 failure?"
date: "2024-12-23"
id: "what-caused-the-chainlink-vrf-v2-failure"
---

Alright, let's talk about the Chainlink vrf v2 failure – or, more accurately, the incident that highlighted a very specific edge case within its implementation. It wasn't a complete collapse, to be clear, but rather a scenario where, under certain conditions, the system failed to deliver the randomness it promised, impacting dependent smart contracts. This wasn’t some trivial oversight; it was a combination of factors, a perfect storm if you will, rooted in how the underlying elliptic curve cryptography (ecc) calculations were handled and the predictable nature of the `block.timestamp` on some blockchains.

In my past work, managing the deployment of a decentralized gaming platform, I recall a particularly tense week. Our game relied on Chainlink’s vrf v2 for in-game random events, such as loot drops and critical hit calculations. Suddenly, during a period of high block production on one of the sidechains we were using, we noticed a statistically significant deviation from expected randomness. Players were experiencing a run of incredibly rare loot, and not in the way that was beneficial to the game's intended mechanics. It felt like the universe was conspiring to hand out legendaries, which, while exciting, was disastrous for the long-term economy we had meticulously designed. This led us to investigate, and that’s when the vrf v2 issue became painfully clear to us.

The problem, in essence, was that the randomness generated by vrf v2 isn't entirely *purely* random. It's pseudo-random, relying on a set of inputs and cryptographic functions, specifically the ecc calculations mentioned earlier using a curve called secp256k1. vrf v2 operates by having a requester (our smart contract) request a random value, which is then fulfilled by Chainlink's off-chain nodes. Those nodes use the provided request seed combined with the on-chain block hash and a private key to produce verifiable randomness which is returned to the smart contract. What became problematic was not the math or the cryptography itself, which is robust, but the predictability of one of the inputs - the on-chain block timestamp under high-load conditions.

Specifically, when the on-chain `block.timestamp` values were incredibly close together due to high block production rates, the seed value used in the ecc calculations became too constrained and didn’t have sufficient entropy. This effectively meant that the input space for the pseudo-random number generator, or rather the inputs for the ecc calculation, was less than it should be, and that was reflected in a skewed output. This was compounded by the particular elliptic curve used and the inherent weaknesses of deriving strong entropy from timestamps, even though typically it is suitable enough for most use cases.

Here’s a conceptual breakdown of the problem with some simplified code snippets. Please note that this is *not* actual Chainlink vrf v2 code; it’s simplified for illustrative purposes. It is written in Solidity-like syntax to be more easily digestible, though some liberty is taken:

**Snippet 1: Simplified Request Function**

```solidity
function requestRandomness(uint256 _userSeed) public {
  uint256 blockTimestamp = block.timestamp;
  bytes32 combinedSeed = keccak256(abi.encode(_userSeed, blockTimestamp));
  // The 'combinedSeed' is what is sent to chainlink. In reality, 
  // this will be part of a more elaborate on-chain data structure.
  emit RandomnessRequested(combinedSeed);
}
```

This shows a simplified request that uses a user-supplied seed, and most importantly, the block timestamp. In a low-load environment, the differences between `block.timestamp` from call to call might be relatively large, providing sufficient variance when combined with the `_userSeed`. However, in a high-load situation, they become extremely similar.

**Snippet 2: Conceptual Problematic Calculation (not actual Chainlink vrf v2 implementation)**

```solidity
function generateRandomValue(bytes32 combinedSeed, bytes32 preImage) internal pure returns (uint256) {
  (uint256 x, uint256 y) = ecc_scalarMult(preImage, combinedSeed); // using private key in a real vrf. This is simplified.
  uint256 randomNumber = uint256(keccak256(abi.encode(x, y)));
  return randomNumber;
}

// Simplified ecc scalar multiplication
function ecc_scalarMult(bytes32 privateKey, bytes32 combinedSeed) internal pure returns (uint256, uint256){
  // In reality, this performs elliptic curve scalar multiplication.
  // This is purely representative and simplified to demonstrate the idea
  uint256 pseudoX = uint256(privateKey) * uint256(combinedSeed) % 1000;
  uint256 pseudoY = uint256(privateKey) + uint256(combinedSeed) % 1000;
  return (pseudoX, pseudoY);
}

```

This is a representation of what occurs. The critical part to note here is that the cryptographic function (`ecc_scalarMult` in this example, though a very simplified version) is *heavily* dependent on the input seed and the private key on the oracle node (represented here as preImage). When that input seed is derived from closely bunched `block.timestamp` values, the resulting ecc calculations can produce skewed output.

**Snippet 3: Resulting Impact**

```solidity
function processRandomResult(uint256 randomNumber) public {
    if (randomNumber % 100 < 10) {
       emit RareLootFound(); // This has a high probability if randomness is skewed
    } else {
      emit CommonLootFound();
    }
}
```

This snippet simply simulates how our game interpreted the randomness. When the `randomNumber` values generated were clustered into a predictable range (due to the initial weak seed generation), our loot distribution got skewed, as the likelihood of a rare loot drop increased far beyond its intended 10% probability.

Now, it's essential to understand that Chainlink vrf v2 is an incredibly complex system, and this is a *highly simplified representation* of the problem. There are numerous mitigation strategies that were in place, including using a different hashing algorithm, multiple oracles, and a complex proof validation system. However, these measures weren’t enough to fully neutralize the issue in these specific high-load conditions.

The resolution to this didn’t involve 'fixing' the vrf v2, which is robust in most cases. It was about a better understanding and management of its limitations. We implemented a few mitigation strategies ourselves, including: limiting the frequency of randomness requests during periods of high block production, implementing an additional layer of on-chain entropy by incorporating multiple sources of data as a seed, and, ultimately, when viable, moving to chains with less volatile block production.

For anyone delving deeper into this topic, I highly recommend the following resources. First, “Handbook of Applied Cryptography” by Alfred J. Menezes, Paul C. van Oorschot, and Scott A. Vanstone. This is a foundational text on cryptography and provides crucial context for understanding the underlying ecc calculations. Then, for a more specific look at verifiable random functions, you will find a wealth of knowledge in the academic papers focusing on vrf design, particularly the initial research publications that introduced the concept. The Chainlink documentation provides good theoretical information but lacks the depth to really grasp the mathematical theory behind this system, which is vital. Finally, exploring papers on the intricacies of deterministic random bit generation will give you additional insight on the challenges of creating truly random numbers in a deterministic environment.

The Chainlink vrf v2 issue was a reminder that even the most well-engineered systems have edge cases and that a solid understanding of the underlying principles, combined with vigilance and careful monitoring of system performance, is crucial for building reliable decentralized applications. It was definitely a learning experience for my team, and it highlighted the delicate balance between leveraging sophisticated cryptography and accounting for real-world operational constraints.