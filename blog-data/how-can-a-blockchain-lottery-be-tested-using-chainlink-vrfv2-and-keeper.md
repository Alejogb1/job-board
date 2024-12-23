---
title: "How can a blockchain lottery be tested using Chainlink VRFv2 and Keeper?"
date: "2024-12-23"
id: "how-can-a-blockchain-lottery-be-tested-using-chainlink-vrfv2-and-keeper"
---

Right, let's talk about testing a blockchain lottery using Chainlink vrfv2 and keeper. This isn't some theoretical exercise for me; I’ve actually built and deployed a few of these in the past, and let me tell you, getting the testing right is crucial for anything with financial implications, especially when randomness is involved. I'm not going to gloss over the details, but I will try to keep it practical and accessible.

So, where to start? We're aiming to rigorously test a system that relies on both verifiable randomness (vrf) and automated contract calls (keeper). Testing these components independently is a good first step, but the real challenge is verifying that they work *together* seamlessly in a way that is secure and predictable.

Let’s break down the core components: Chainlink VRFv2 provides the random number. The goal here is not to generate a ‘random number’ in the traditional sense (pseudo-random), but to produce an unpredictable number that can be cryptographically verified as being generated without any manipulation from any party, including the smart contract or its deployer. The keeper, on the other hand, is the automation system that triggers the lottery draw when specific conditions are met, such as time or a target number of participants.

Now, you can't realistically exhaustively test the entire randomness space due to the large numbers involved, but you can establish confidence in your implementation. This boils down to focusing on specific aspects of its behavior, rather than expecting to check every possible result.

Here's how I generally approach this testing. First, let's focus on the VRF component. You can simulate VRF request and fulfillment within your test suite using helper functions that wrap the actual Chainlink contracts. The key idea is to bypass the actual on-chain transaction to VRF, instead, directly calling the VRF fulfill function with pre-generated data during tests.

Here's a snippet of how you can achieve this using a testing framework like hardhat and ethers.js:

```javascript
const { ethers } = require("hardhat");
const { solidity } = require("ethereum-waffle");
const { BigNumber } = require('ethers');
const chai = require("chai");
chai.use(solidity);
const { expect } = chai;

describe("Lottery with VRF Simulation", function () {
  let lotteryContract;
  let mockVRFCoordinator;
  let mockLinkToken;

  beforeEach(async function () {
      //deploy mock contracts
    const VRFCoordinatorMock = await ethers.getContractFactory("VRFCoordinatorV2Mock");
    mockVRFCoordinator = await VRFCoordinatorMock.deploy();
    await mockVRFCoordinator.deployed();

    const LinkTokenMock = await ethers.getContractFactory("LinkToken");
    mockLinkToken = await LinkTokenMock.deploy();
    await mockLinkToken.deployed();

    //deploy the lottery contract
    const Lottery = await ethers.getContractFactory("Lottery");
    lotteryContract = await Lottery.deploy(mockVRFCoordinator.address, mockLinkToken.address, "0xd89b2bf150e3b9e13446986e57157a967944723c");
    await lotteryContract.deployed();
  });

   it("Should fulfill a VRF request directly in a test.", async function () {
    const requestId = 123;
    const randomNumber = 456;
    const keyHash = "0xd89b2bf150e3b9e13446986e57157a967944723c";
    const preSeed = 789;
    await mockVRFCoordinator.fulfillRandomWords(requestId, lotteryContract.address, [randomNumber], keyHash);
    const currentRandom = await lotteryContract.currentRandomNumber();

    expect(currentRandom).to.equal(randomNumber);
  });
});
```

This code establishes mock VRF contracts, allowing you to simulate the response directly. We are directly calling the `fulfillRandomWords` on the Mock VRF Coordinator, bypassing the usual process. We verify the expected outcome is what we received in the `currentRandomNumber`.

Now, let’s move onto the keeper aspect of testing. For this, we need to verify that the keeper is executing the `performUpkeep` function only under the conditions we set in the contract logic. We'll directly call the checkUpkeep function (this is what the Chainlink keeper calls to decide whether to execute performUpkeep), to ensure it evaluates to true or false under the correct circumstances, without interacting with the keepers themselves during tests.

Here’s a code snippet demonstrating this:

```javascript
  it("Should trigger performUpkeep when conditions are met.", async function () {
      await lotteryContract.startLottery();
      await mockLinkToken.transfer(lotteryContract.address, ethers.utils.parseEther("100"));
    const checkData = "0x";
    const { upkeepNeeded } = await lotteryContract.callStatic.checkUpkeep(checkData);
    expect(upkeepNeeded).to.be.true;

    const tx = await lotteryContract.performUpkeep(checkData);
    const receipt = await tx.wait();
    const event = receipt.events?.find(event => event.event === 'LotteryDrawn');
      expect(event).to.not.be.undefined;

  });

  it("Should not trigger performUpkeep when conditions are not met.", async function () {
      const checkData = "0x";
      const { upkeepNeeded } = await lotteryContract.callStatic.checkUpkeep(checkData);
      expect(upkeepNeeded).to.be.false;

    await expect(lotteryContract.performUpkeep(checkData)).to.be.revertedWith("Lottery: upkeep not needed");
  });
```

Here, we are directly calling checkUpkeep to see if the upkeep is needed. We then verify performUpkeep executes, or reverts with the correct error, depending on the upkeep evaluation. We also examine the emitted event of a LotteryDrawn.

Finally, testing the combined system requires more integrated tests. You’d typically simulate the entire process from lottery start, participant entry, and final drawing and payout with mock data and direct function calls (similar to the approach shown above). You want to simulate various scenarios such as no entrants, a single entrant, and multiple entrants. In your tests you'll use the keeper's `performUpkeep`, and mock VRF fulfill functions to move the lottery along. Here is an example:

```javascript
  it("Should run an end-to-end lottery simulation", async function () {
        await mockLinkToken.transfer(lotteryContract.address, ethers.utils.parseEther("100"));
      await lotteryContract.startLottery();
    // Add some players
    const accounts = await ethers.getSigners();
    await lotteryContract.connect(accounts[1]).enter({value: ethers.utils.parseEther("1")});
    await lotteryContract.connect(accounts[2]).enter({value: ethers.utils.parseEther("1")});
    await lotteryContract.connect(accounts[3]).enter({value: ethers.utils.parseEther("1")});

    const checkData = "0x";
    const { upkeepNeeded } = await lotteryContract.callStatic.checkUpkeep(checkData);
    expect(upkeepNeeded).to.be.true;

    const tx = await lotteryContract.performUpkeep(checkData);
    const receipt = await tx.wait();
    const event = receipt.events?.find(event => event.event === 'LotteryDrawn');
    expect(event).to.not.be.undefined;
    const requestId = event?.args[0];
    const keyHash = "0xd89b2bf150e3b9e13446986e57157a967944723c";
    const randomNumber = 456;
    await mockVRFCoordinator.fulfillRandomWords(requestId, lotteryContract.address, [randomNumber], keyHash);

    const winner = await lotteryContract.winner();
        expect(winner).to.not.equal("0x0000000000000000000000000000000000000000");
   });
```

In this test, we simulate player entries, we perform the upkeep and verify its executed. We simulate the VRF response using the mock and verify we have a winner.

It's important to emphasize that while these examples help establish confidence, they do *not* eliminate all risk. Real-world blockchain interactions involve numerous complexities, including network latency, gas fee fluctuations, and potential vulnerabilities in the underlying smart contract code and deployed infrastructure.

For deeper study, I strongly recommend "Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood, and specifically chapter 12 which focuses on contract testing. The Chainlink documentation, although not a book, is invaluable for understanding VRF and keeper functionality at an in-depth technical level. Also, reading the original research papers related to VRF design would enhance understanding, for example, you could search for academic works on Verifiable Random Functions from institutions such as MIT or Stanford. Finally, carefully review the security audits of Chainlink's contracts and your own smart contracts, this is essential practice before any production deployments.

Testing such systems is an ongoing process. Start with smaller, isolated tests, then move toward integrated scenarios, and always remain vigilant. As you've seen, practical testing methodologies are fundamental for building secure and functional blockchain-based systems, so do put in the necessary effort and attention to detail.
