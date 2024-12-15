---
title: "How to do an Anchor smart contract where two parties pays the contract and one of them get's the amount?"
date: "2024-12-15"
id: "how-to-do-an-anchor-smart-contract-where-two-parties-pays-the-contract-and-one-of-them-gets-the-amount"
---

alright, so, you're looking at creating an anchor smart contract where two participants deposit funds, and then, under specific conditions, one of them receives the total deposited amount. i've definitely been down this road before, it's a pretty common pattern when setting up escrows or conditional payments on the blockchain. let me walk you through how i usually approach this, hopefully it will save you some headaches i've suffered.

first, let's break down the core functionality, it's all about managing states and events on the blockchain:

*   **initialization:** the contract starts with a state where no funds have been deposited.
*   **funding:** two parties make deposits to the contract. we need to track these deposits, ensuring both parties participate.
*   **trigger condition:** this is the heart of your logic, the condition that determines when the funds should be released. this could be anything: a flag set by one of the parties, the outcome of an external oracle, a time-based trigger, or any combination of these.
*   **payout:** once the trigger condition is met, the designated recipient receives the total sum.
*   **failure scenario:** what happens if the trigger condition isn't met within a reasonable timeframe? in some cases, the funds might need to be returned to the original depositors.
*   **security:** of course, we need to consider all security implications such as reentrancy, overflow, underflow, all the blockchain usual suspects.

so, let's jump into some code examples to see how this can be implemented with solidity, which is what i usually use. this first example is simplified for clarity, and ignores some security and gas optimizations:

```solidity
pragma solidity ^0.8.0;

contract simpleanchor {
    address payable public partyA;
    address payable public partyB;
    uint256 public depositA;
    uint256 public depositB;
    bool public triggered;

    constructor() {
        partyA = payable(msg.sender);
        partyB = payable(address(0));
        triggered = false;
    }

    function fundB() public payable {
       require(partyB == address(0), "party b already funded");
        partyB = payable(msg.sender);
        depositB = msg.value;
    }

    function fundA() public payable {
        require(msg.sender == partyA, "only party a can fund");
        depositA = msg.value;
    }
    

    function triggerpayment() public {
        require(msg.sender == partyA, "only party a can trigger");
        require(triggered == false, "already triggered");

        uint256 total = depositA + depositB;
        payable(partyA).transfer(total);
        triggered = true;
    }
}
```

here are a few important notes about this first simple example:

*   this contract only allows party a to trigger the payment.
*   there is no refund functionality in this example.
*   it lacks several safety checks such as a way to prevent anyone to overwrite party b account or to prevent someone sending the ether before party b is set.

in this second example, we're adding some safeguards and a refund mechanism:

```solidity
pragma solidity ^0.8.0;

contract improvedanchor {
    address payable public partyA;
    address payable public partyB;
    uint256 public depositA;
    uint256 public depositB;
    bool public triggered;
    bool public canRefund;
    uint256 public deadline;

    event depositedA(uint256 amount);
    event depositedB(uint256 amount);
    event paymentTriggered(uint256 amount);
    event refundTriggered();

    constructor(uint256 _timeToRefund) {
        partyA = payable(msg.sender);
        partyB = payable(address(0));
        triggered = false;
        canRefund = true;
        deadline = block.timestamp + _timeToRefund;
    }

   function fundB() public payable {
       require(partyB == address(0), "party b already funded");
       require(msg.value > 0, "you have to send some amount of ether");
        partyB = payable(msg.sender);
        depositB = msg.value;
        emit depositedB(msg.value);
    }

    function fundA() public payable {
       require(msg.sender == partyA, "only party a can fund");
       require(msg.value > 0, "you have to send some amount of ether");
        depositA = msg.value;
       emit depositedA(msg.value);
    }

    function triggerpayment() public {
        require(msg.sender == partyA, "only party a can trigger");
        require(triggered == false, "already triggered");
        require(depositA > 0 && depositB > 0, "cannot be 0 for both parties");

        uint256 total = depositA + depositB;
        payable(partyA).transfer(total);
        triggered = true;
       emit paymentTriggered(total);
    }
    
     function refund() public {
        require(block.timestamp > deadline, "too early for a refund");
        require(canRefund == true, "refund already claimed");

         uint256 total = depositA + depositB;
        payable(partyA).transfer(depositA);
        payable(partyB).transfer(depositB);
        canRefund = false;
        emit refundTriggered();
    }
}
```

key improvements in the second example:

*   added a deadline, refund function to return funds if the trigger doesn't happen in time.
*   added event emission for better traceability
*   added a modifier to avoid both parties not sending any ether.
*   added require message to better understand errors in function calls.
*   added a way to cancel the refund function after it has been called.
*   this is a more robust version but it is still not perfect and can be improved further.

in the third example, we are taking a different approach with a timelock and a different logic:

```solidity
pragma solidity ^0.8.0;

contract timelockanchor {
    address payable public partyA;
    address payable public partyB;
    uint256 public depositA;
    uint256 public depositB;
    uint256 public releaseTime;

     event depositedA(uint256 amount);
     event depositedB(uint256 amount);
     event payouttriggered();

     constructor(uint256 _releaseTime) {
        partyA = payable(msg.sender);
        partyB = payable(address(0));
       releaseTime = block.timestamp + _releaseTime;
    }

    function fundB() public payable {
        require(partyB == address(0), "party b already funded");
        require(msg.value > 0, "you have to send some amount of ether");
        partyB = payable(msg.sender);
        depositB = msg.value;
         emit depositedB(msg.value);
    }

    function fundA() public payable {
        require(msg.sender == partyA, "only party a can fund");
       require(msg.value > 0, "you have to send some amount of ether");
        depositA = msg.value;
        emit depositedA(msg.value);
    }

    function withdrawfunds() public {
       require(block.timestamp >= releaseTime, "too early to withdraw");
       require(depositA > 0 && depositB > 0, "both parties need to deposit funds");

        uint256 total = depositA + depositB;
        payable(partyA).transfer(total);
        emit payouttriggered();
    }
}
```

in this third example:

*   funds are released to party a after a timelock
*    there is no way to refund the funds before timelock

a bit of advice, avoid using `transfer()` for sending ether, it has a limit of gas, which is not the best, use `call()` instead. i'm only using `transfer()` in this examples to make it simpler.

a thing i've learned from past errors is that when dealing with smart contracts for any kind of monetary exchange, always test thoroughly before deploying to mainnet and always audit your contract if you are dealing with considerable amounts. it's a lot more painful to fix problems after a contract has been deployed. i remember one time i forgot to add a check for zero value in one of my contracts, and someone sent me 0 ether, and it was just stuck there. i have no idea why someone would do it, it was just a bug in my code. the good old days, that was just a funny experience.

now, in terms of resources, i would recommend checking out "mastering ethereum" by andreas m. antonopoulos. it's a pretty good book that goes deep into the topic. also, for security best practices, the openzeppelin documentation is a good starting point, but i find that constantly reviewing common attack vectors, checking the latest exploits and trying to understand how they happen, in combination with experience is the only way to go. i would also recommend papers from academic conferences like ieee security & privacy if you want to dive deep into cryptography research papers or even look at what other people are doing and try to contribute to the community.
remember that this is just the tip of the iceberg, smart contracts can get really complex depending on what you are trying to achieve, but this should set you in the right direction. happy coding!
