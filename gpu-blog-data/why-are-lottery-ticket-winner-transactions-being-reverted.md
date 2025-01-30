---
title: "Why are lottery ticket winner transactions being reverted by the EVM?"
date: "2025-01-30"
id: "why-are-lottery-ticket-winner-transactions-being-reverted"
---
The most probable cause for lottery ticket winner transactions being reverted on the Ethereum Virtual Machine (EVM) stems from a failure in the state transition logic within the smart contract handling the lottery. Specifically, these errors are rarely inherent in the EVM itself; instead, they arise from issues in how the smart contract manages fund disbursement, winner verification, or its internal state during the transaction process. I've personally encountered this issue on multiple occasions while auditing and developing decentralized applications utilizing random number generation and complex state management.

The root problem often lies in unintended interactions between several critical areas: randomness generation, winner selection, payout calculation, and the contract’s gas limits. Let’s unpack these areas and illustrate them with concrete examples, based on situations I've witnessed and debugged.

**1. Inconsistent Random Number Generation**

Deterministic randomness in smart contracts is a significant concern. Simply using the `block.timestamp` or `block.number` as the seed for random number generation is demonstrably insecure and can be manipulated by miners. However, even seemingly secure methods, if implemented incorrectly, can still lead to reverts. One common mistake is failing to properly handle edge cases in the randomness algorithm or failing to ensure its consistency across multiple function calls within the same transaction. Consider the following:

```solidity
pragma solidity ^0.8.0;

contract Lottery {
    uint public lastBlock;
    uint public winnerNumber;
    uint public ticketPrice = 1 ether;
    uint public prizePool;
    mapping(address => bool) public participants;

    function buyTicket() public payable {
        require(msg.value == ticketPrice, "Incorrect ticket price.");
        participants[msg.sender] = true;
        prizePool += msg.value;
    }

    function drawWinner() public {
        require(block.number > lastBlock, "Must wait for a new block.");
        lastBlock = block.number;
        uint randomNumber = uint(keccak256(abi.encode(block.timestamp, msg.sender))); //Insecure method, but for illustrative purposes
        uint numberOfParticipants = 0;
        for (uint i = 0; i < participants.length; i++){
            if(participants[address(i)]) numberOfParticipants++;
        }
        require(numberOfParticipants > 0, "No participants.");
        winnerNumber = randomNumber % numberOfParticipants;

        //... payout logic is missing for brevity...
    }

}
```

Here, the randomness is derived from `keccak256(abi.encode(block.timestamp, msg.sender))`. While it uses `keccak256`, it is still vulnerable to manipulation or block timestamp collisions. Furthermore, using the number of participants in a loop can lead to unexpected errors when modifying mappings. This implementation is also highly susceptible to frontrunning. Even with these known limitations, the issue that I have witnessed directly is a reversion due to the inconsistent application of `winnerNumber`. If the `drawWinner` function is called multiple times in the same block or rapidly within a few blocks, the `randomNumber` may result in an index outside of the current list of participants, thus potentially leading to an out-of-bounds access when an array (that is absent) is used to locate the winner and attempt a payout. The revert occurs because the code is designed to expect a specific range of participants, but the randomness generates a value outside of this established range, especially when the number of participants is dynamic and modified during the function's execution, which is common in practice.

**2. Insufficient Gas Limit**

Smart contracts execute within a gas-constrained environment. Operations like loops, complex calculations, or storage updates consume gas. If a function attempts to perform an operation that exceeds the transaction’s gas limit, it will revert. The complexity of state manipulation in a lottery contract—involving iterating through participants, calculating winnings, and updating balances—often leads to this. I have often seen this occur when developers fail to account for growth in the number of participants. Consider this example:

```solidity
pragma solidity ^0.8.0;
contract Lottery {
    mapping(address => uint) public balances;
    address[] public participants;
    uint public prizePool;
    uint public ticketPrice = 1 ether;

    function buyTicket() public payable {
         require(msg.value == ticketPrice, "Incorrect ticket price.");
        participants.push(msg.sender);
        balances[msg.sender] += msg.value;
        prizePool += msg.value;
    }

    function drawWinner(uint winnerIndex) public {
      require(participants.length > 0, "No participants.");
      address winner = participants[winnerIndex];
      uint payout = prizePool;
        for (uint i = 0; i < participants.length; i++){
            if(participants[i] != winner)
              balances[participants[i]] = 0; //Resets balances of losers
        }
      (bool success, ) = winner.call{value: payout}("");
       require(success, "Transfer failed.");
      prizePool = 0;
    }
}
```

In this example, the `drawWinner` function iterates through the `participants` array, resetting the balances of losing participants. As the number of participants grows, the number of iterations increases, causing a proportional increase in gas consumption. This often leads to a "out-of-gas" error and thus a revert of the transaction. The core problem is performing a linear iteration across an unbounded array, with costly storage updates within the loop. While the actual payout is a simple call, the loop required to clear losers may consume an exorbitant amount of gas, causing the entire transaction to fail. I have observed this scenario being a primary culprit for transaction reversions with contracts that accrue an unexpectedly high user count.

**3. Reentrancy Vulnerabilities and Incorrect Payout Logic**

Reentrancy vulnerabilities arise when an external call is made from a contract before its internal state is completely updated. A malicious contract can then re-enter the vulnerable function before the state update is complete, potentially creating an exploit. The payout logic must be carefully designed to avoid allowing recursive calls to exploit the system. Consider the following:

```solidity
pragma solidity ^0.8.0;

contract Lottery {
    mapping(address => uint) public balances;
    address payable public winner;
    uint public prizePool;

    function buyTicket() public payable {
       balances[msg.sender] += msg.value;
       prizePool += msg.value;
    }

  function setWinner(address payable _winner) public {
        winner = _winner;
    }
    function claimWinnings() public  {
      uint payout = balances[msg.sender];
      require(msg.sender == winner, "Only the winner can claim.");
      require(payout > 0, "No winnings to claim.");
      balances[msg.sender] = 0;
      (bool success, ) = msg.sender.call{value: payout}("");
      require(success, "Transfer failed.");

    }
}
```

Here, a winner is set using `setWinner`, and then they can call `claimWinnings` to get the payout. The payout mechanism uses a call that sends funds directly. If the recipient contract (the winner) were malicious and implemented a `fallback` function that called back into `claimWinnings` before the `balances[msg.sender] = 0;` line is reached, it could claim the payout multiple times. This reentrancy issue is very dangerous and would ultimately cause the transaction to fail (due to the state mismatch) after the exploited re-entry loop concludes. While not a direct reversion cause, the subsequent call in the loop would eventually fail, and if unchecked, would cause a revert due to insufficient funds in the smart contract, and would potentially cause the transaction in question to revert. In my experience, ensuring that state changes are applied before external calls is essential and should be a cornerstone of smart contract design, using patterns like “Checks-Effects-Interactions”.

**Resource Recommendations:**

For a deeper understanding of secure smart contract development, I recommend focusing on the following areas using available resources:

1.  **Solidity Documentation:** The official Solidity documentation provides comprehensive details about language features and best practices. Thoroughly understanding the intricacies of storage, gas costs, and function call semantics is crucial.
2.  **Ethereum Improvement Proposals (EIPs):** Reviewing relevant EIPs, particularly those related to contract standards, gas metering, and randomness generation can give valuable context.
3.  **Auditing Checklists:** Using auditing checklists is paramount. These resources usually cover common vulnerabilities, including gas-related issues, reentrancy attacks, and flaws in randomness generation.
4.  **Smart Contract Development Books:** Invest time into specialized literature covering best practices for EVM-based development.

By focusing on the nuances of randomness, gas limits, and reentrancy, along with consistent testing and auditing, one can mitigate the majority of causes for reverted transactions observed in EVM-based lottery contracts, and beyond.
