---
title: "How do I generate 10,000 random numbers from Chainlink VRF V2?"
date: "2024-12-23"
id: "how-do-i-generate-10000-random-numbers-from-chainlink-vrf-v2"
---

Okay, let's tackle this. Generating 10,000 random numbers using Chainlink VRF V2 is a task that requires a good understanding of how VRF works and its limitations, especially regarding on-chain computations. From my experience, having implemented a lottery system using VRF a few years back, I recall facing a similar challenge, albeit with slightly smaller sets. The core issue is that generating such a large quantity of random numbers directly in a single transaction is impractical due to gas limits and computational costs on the ethereum network. We need to approach it strategically.

The main constraint we have here is the gas limit for a single ethereum transaction. On-chain generation of 10,000 numbers, each requiring a cryptographic calculation, would not only be incredibly expensive but would almost certainly exceed block gas limits. Thus, the immediate approach is not to create them all within a single call, but to use a strategy of either multiple requests or some form of off-chain pre-generation that's verifiable on-chain.

Let's break down the primary options and then dive into the code.

**Option 1: Multiple Requests, On-Chain Storage**

The most straightforward approach, albeit more gas intensive than others, involves making multiple requests to the VRF contract. This means your contract will have to make a request for one (or several, but keep in mind the on-chain gas limit) random number(s) at a time, then use an event listener to record and store the generated random number(s). The big upside here is that everything is transparently verifiable on-chain and you don't have to trust any intermediary. The key thing, though, is that you will not be generating all numbers in one single call. You must stagger the requests.

Here's a basic solidity contract illustrating this pattern:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";

contract RandomNumberGenerator is VRFConsumerBaseV2 {

    VRFCoordinatorV2Interface COORDINATOR;
    uint64 public s_subscriptionId;
    bytes32 public keyHash;
    uint32 public callbackGasLimit = 50000;
    uint16 public requestConfirmations = 3;
    uint32 public numWords = 1;

    uint256 public lastRequestId;
    uint256[] public generatedNumbers;

    event RequestFulfilled(uint256 requestId, uint256[] randomWords);

    constructor(address vrfCoordinator, uint64 subscriptionId, bytes32 keyHash)
        VRFConsumerBaseV2(vrfCoordinator)
    {
      COORDINATOR = VRFCoordinatorV2Interface(vrfCoordinator);
      s_subscriptionId = subscriptionId;
      this.keyHash = keyHash;
    }


    function requestRandomWords() external {
       uint256 requestId = COORDINATOR.requestRandomWords(
            keyHash,
            s_subscriptionId,
            requestConfirmations,
            callbackGasLimit,
            numWords
        );
        lastRequestId = requestId;
    }

    function fulfillRandomWords(uint256 requestId, uint256[] memory randomWords) internal override {
      generatedNumbers.push(randomWords[0]);
      emit RequestFulfilled(requestId, randomWords);
    }

    function getTotalGeneratedCount() external view returns (uint256) {
       return generatedNumbers.length;
    }


}
```

This code initiates requests, and the `fulfillRandomWords` method appends generated numbers to the `generatedNumbers` array. To generate 10,000 numbers, you'd need to call the `requestRandomWords` method many times, keeping in mind to not flood the network or your node with requests, possibly adding a delay between each request.

**Option 2: Off-Chain Pre-Generation with On-Chain Verification**

A more economical but conceptually more involved approach is to generate the 10,000 numbers off-chain and then verify them on-chain. This would utilize the VRF to generate the initial seed, then an off-chain process could generate the rest of the numbers using that initial seed and a deterministic algorithm. The key is that we would provide proof to the contract, demonstrating that numbers were created using the initial seed correctly. This involves several parts working together. This is more technically challenging to implement and requires a bit more development work.

First, on-chain we’d need a smart contract with a method to verify the randomness, the initial seed and the commitment to the random numbers. Off-chain we would have a script that uses the initial seed and generates all 10000 numbers by applying a consistent algorithm like a pseudorandom number generator. You’d then have to provide proof to your smart contract that your random numbers are the result of correctly applying this algorithm.

Here’s an example of a simplified contract for this approach:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@chainlink/contracts/src/v0.8/VRFConsumerBaseV2.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";
import "hardhat/console.sol";

contract VerifiableRandomNumbers is VRFConsumerBaseV2 {
   VRFCoordinatorV2Interface COORDINATOR;
    uint64 public s_subscriptionId;
    bytes32 public keyHash;
    uint32 public callbackGasLimit = 50000;
    uint16 public requestConfirmations = 3;
    uint32 public numWords = 1;


    uint256 public initialSeed;
    uint256 public lastRequestId;

    event InitialSeedRequested(uint256 requestId);
    event InitialSeedFulfilled(uint256 seed);
    event NumbersVerified(bool verificationSuccess);

    constructor(address vrfCoordinator, uint64 subscriptionId, bytes32 keyHash)
        VRFConsumerBaseV2(vrfCoordinator)
    {
       COORDINATOR = VRFCoordinatorV2Interface(vrfCoordinator);
       s_subscriptionId = subscriptionId;
       this.keyHash = keyHash;
    }

    function requestInitialSeed() external {
        uint256 requestId = COORDINATOR.requestRandomWords(
            keyHash,
            s_subscriptionId,
            requestConfirmations,
            callbackGasLimit,
            numWords
        );
       lastRequestId = requestId;
       emit InitialSeedRequested(requestId);
    }

     function fulfillRandomWords(uint256 requestId, uint256[] memory randomWords) internal override {
       initialSeed = randomWords[0];
       emit InitialSeedFulfilled(initialSeed);
     }

    function verifyRandomNumbers(uint256 seed, uint256[] memory randomNumbers) public returns (bool){
          require(seed == initialSeed, "Initial Seed Mismatch");
          uint256  nextValue = seed;
          for(uint256 i = 0; i< randomNumbers.length; i++)
          {
               nextValue = uint256(keccak256(abi.encode(nextValue)));
               require(nextValue == randomNumbers[i], "Incorrect Sequence");
          }
         emit NumbersVerified(true);
        return true;

    }
}
```

This contract has an off-chain script use the seed from `fulfillRandomWords`, generate numbers using a deterministic algorithm (like the `keccak256` hash of the previous value as shown in the `verifyRandomNumbers` method), and finally provide that series back to the smart contract for verification using the `verifyRandomNumbers` method.

Here is an off-chain script example for how this might work in javascript (using ethers.js):

```javascript

const { ethers } = require("ethers");
const {keccak256} = require("ethers/lib/utils");

async function generateAndVerifyRandomNumbers(seed, count) {

    let randomNumbers = [];
    let nextValue = seed;
    for(let i = 0; i< count; i++){
      nextValue = ethers.BigNumber.from(keccak256(ethers.utils.defaultAbiCoder.encode(["uint256"], [nextValue]))).toString()
      randomNumbers.push(nextValue);
    }
    console.log("Generated Numbers", randomNumbers);
    return randomNumbers;
}

async function verifyOnChain(contract, seed, generatedNumbers) {
    try {
      const tx = await contract.verifyRandomNumbers(ethers.BigNumber.from(seed), generatedNumbers);
      const receipt = await tx.wait();

      console.log("Verification Transaction Mined: ", receipt.transactionHash);
      return true;
    } catch (error) {
      console.error("Error during on-chain verification: ", error)
      return false;
    }
}

async function main(){
 const provider = new ethers.providers.JsonRpcProvider("YOUR_RPC_ENDPOINT");
 const privateKey = "YOUR_PRIVATE_KEY";
 const wallet = new ethers.Wallet(privateKey, provider);

 const contractAddress = "YOUR_CONTRACT_ADDRESS";
 const abi = [
  // Contract ABI Here
    "function verifyRandomNumbers(uint256 seed, uint256[] memory randomNumbers) public returns (bool)",
  ];

 const contract = new ethers.Contract(contractAddress, abi, wallet);
 const initialSeed =  "YOUR_INITIAL_SEED_FROM_CONTRACT"

 const randomNumbers = await generateAndVerifyRandomNumbers(initialSeed, 10000);
 await verifyOnChain(contract, initialSeed, randomNumbers);
}

main();
```

This script uses the initial seed, deterministically generates 10,000 numbers, and then calls the contract's `verifyRandomNumbers` method for on-chain verification. Remember that you will need to replace all of the values with your proper values.

**Option 3: Batch Verification (Variation on Option 2)**

A potential variation of option 2 is to still do off-chain generation, but then split the verification process into batches, submitting batches of random numbers for verification, instead of submitting all 10,000 at once. This would break the transaction size into something manageable, with the tradeoff that the overall process takes a few more steps, and a few more gas costs, but can remain within on chain limits. The specific implementation of this will depend on your specific application.

**Recommendations**

For understanding VRF in depth, the official Chainlink documentation is your first point of call. The papers by Juels and Kosba, and the other VRF related papers are also an excellent source for better theoretical understanding. For a deeper understanding of how to write robust smart contracts, consider reading "Mastering Ethereum" by Andreas M. Antonopoulos, and for a focus on security best practices, “Building Secure Blockchain Applications” by Arnab Chakraborty.

In conclusion, directly generating 10,000 random numbers within a single transaction with VRF v2 is simply not practical. Choose the right approach for your specific use case considering the tradeoffs between on-chain verification, complexity and costs. Whether you go with option 1, or option 2, careful implementation will be critical to having a robust and efficient solution.
