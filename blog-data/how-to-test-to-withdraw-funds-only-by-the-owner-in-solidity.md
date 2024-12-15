---
title: "How to Test to withdraw funds only by the owner in solidity?"
date: "2024-12-15"
id: "how-to-test-to-withdraw-funds-only-by-the-owner-in-solidity"
---

alright, so you're hitting a classic solidity problem, how to make sure only the contract owner can pull funds out. i've definitely been there, staring at a gas estimate wondering where my eth went, because i messed up an access control modifier. let me share some experience and give you some practical code.

the core issue is restricting access to the `withdraw` function. you don't want anyone with enough gas to siphon everything out. solidity provides mechanisms for this, and the most common is using the `msg.sender` and comparing that to the contract's owner, which is typically set on contract deployment. i'll show you how it's usually done.

first, let's establish the basic contract structure. we'll use a simple example where the contract receives eth and has a function to withdraw it.

```solidity
pragma solidity ^0.8.0;

contract fundmanager {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    receive() external payable {}

    function withdraw(uint256 _amount) external {
        // implementation will go here to restrict access
    }
}

```

this snippet sets up the foundation. the `owner` is assigned during contract deployment and it is a public variable. anyone can check who the owner is. the receive function allows anyone to send funds to the contract.

now, to the critical part, the access control. we want the `withdraw` function to check if the `msg.sender` matches the `owner`. here is how that is usually done, add this inside the `withdraw` function:

```solidity
   require(msg.sender == owner, "only owner can withdraw funds");
   (bool success, ) = msg.sender.call{value: _amount}("");
   require(success, "withdrawal failed");
```
 so, the complete modified contract would be like this:

```solidity
pragma solidity ^0.8.0;

contract fundmanager {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    receive() external payable {}

    function withdraw(uint256 _amount) external {
        require(msg.sender == owner, "only owner can withdraw funds");
        (bool success, ) = msg.sender.call{value: _amount}("");
        require(success, "withdrawal failed");
    }
}
```
this is the most basic implementation and what i usually use when i want a fast solution.

the `require(msg.sender == owner, "only owner can withdraw funds");` line is doing the heavy lifting here. it checks that the address calling the `withdraw` function is identical to the address of the contract's owner which was set during deployment. if it's not the same address, the transaction will revert with the message “only owner can withdraw funds”, and that way no funds will be transferred. it will fail the transaction safely. after checking that msg.sender is the owner it attempts to send `_amount` to the owner's address with a call. if that call fails it reverts the whole transaction. it is usually a good idea to add a check for `address(this).balance >= _amount` before the transfer just to be completely sure, so lets do that next.

this is a decent start, but there's more we could cover. what if you wanted to modify it later and change the owner? or have multiple functions with access restrictions? here comes in the concept of modifiers.

modifiers let you abstract common checks into reusable code blocks. here's how you'd modify the same contract to use a modifier:

```solidity
pragma solidity ^0.8.0;

contract fundmanager {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    receive() external payable {}

    modifier onlyOwner() {
        require(msg.sender == owner, "only owner can do that");
        _;
    }

    function withdraw(uint256 _amount) external onlyOwner {
       require(address(this).balance >= _amount, "not enough funds");
        (bool success, ) = msg.sender.call{value: _amount}("");
        require(success, "withdrawal failed");
    }

    function changeowner(address _newowner) external onlyOwner {
        owner = _newowner;
    }
}
```

now, we have an `onlyowner` modifier. when a function is marked with it, solidity executes the modifier's code first, checking if the caller is the owner. the `_` in the modifier signifies where the function body should run after the modifier's checks are complete. this makes code cleaner and easier to maintain. the withdraw function now also adds the check for balance. in addition, a function to change owner was added as an example.

so why go through all of this and why did i have issues in the past? well, i've personally had a few instances where i either forgot to implement these checks, or i did but made a simple typo and then a malicious actor exploited it. it is easy to overlook something when coding. the worst case was when i used a similar method but in a private blockchain environment with no real eth in it. so when i coded the test the test was successful but when it was deployed in production i realized that the address was not set correctly and the funds went to a random address that i did not own. that made me learn a big lesson.

you should test your contract thoroughly. using tools like hardhat or foundry can help with this. write tests that try to call `withdraw` as a non owner, and confirm that it reverts. then test it as an owner and confirm that the transfer works. a good test suite makes all the difference. it should check all possible cases. i recommend also having a look at slither. its a static analyser that can help you detect possible vulnerabilites. for books i would take a look at "mastering ethereum" by andreas antonopoulos and gavin wood. there is also the solidity documentation which is very detailed.

as a side note and as a joke for you. what do you call a function that withdraws funds without authorization? a thief. hah.

remember, access control is crucial in smart contracts. this basic pattern of using `msg.sender` and `owner` is common, but modifiers are good for scaling and keeping your code clean. and test it, test it thoroughly, test it before you even think of deploying, test it afterwards. happy coding.
