---
title: "I am not able to understand how to use IVotes contract address in a GovernorVotes contract?"
date: "2024-12-15"
id: "i-am-not-able-to-understand-how-to-use-ivotes-contract-address-in-a-governorvotes-contract"
---

alright, so you're having a bit of a head-scratcher with integrating an `ivotes` contract address into a `governorvotes` contract. i've been down this road before, it can feel a bit like trying to fit a square peg into a round hole if you're not careful with the details. let me walk you through what i've learned, based on some hard-won battles with similar setups in the past.

first off, let's break down the core problem. `ivotes` usually acts as the data source for voting power – it keeps track of who has how many votes. a `governorvotes` contract, on the other hand, uses this data to enable governance proposals and voting. the trick is to make sure the `governorvotes` contract knows where to fetch the voting power from, which is your `ivotes` contract address.

the first thing i want to point out is that, at the smart contract level you're working with addresses, and that's the key part, you need to pass the `ivotes` address into your `governorvotes` contract, usually at the constructor, in a way that the governor contract has it as a state variable which can be accessed later. it's not magic, it's just smart contract code accessing another smart contract. let me show you a bit of solidity code of how it should look like.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/governance/Governor.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorVotes.sol";


contract CustomGovernor is Governor, GovernorVotes {
    
    constructor(address _votesAddress) 
    Governor("MyGovernor") 
    GovernorVotes(_votesAddress) {
    }
}
```

in the code above, we're making use of openzeppelin's contracts. i've used this library a lot in the past and i can tell you it really makes things way easier. it provides the `governor` and `governorvotes` contracts that we're extending, and as you can see the key here is the `_votesAddress`, that will be passed in the constructor. this address, as we said, is the address of the `ivotes` contract, and `governorvotes` will use this address internally to query information about votes.

you might be wondering, why do we need this `_votesAddress` variable to be an `address`? because that's how contracts interact with each other within the ethereum virtual machine, or evm. in the evm, every contract has a unique address. when you want one contract to talk to another, you have to know the address of the contract you want to interact with.

, let's talk about how to practically use that address, like inside the constructor, it's the best place in my opinion. i've had my share of mistakes in the past where i tried to update variables after the construction phase only to find out that it makes things a bit more complex that it has to be.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/governance/Governor.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorVotes.sol";


contract CustomGovernor is Governor, GovernorVotes {

    constructor(address _votesAddress)
        Governor("MyGovernor")
        GovernorVotes(_votesAddress)
    {
    }
    
    function token() public pure override returns (address) {
        //we do not need the token function here, because the votes logic is handled by the IVotes contract.
        revert("we do not use ERC20 token here");
    }
}
```

notice the `token()` function? normally you would associate a governor contract with an erc20 token. however when using governorvotes and `ivotes`, we don't need it. `governorvotes` itself handles that interaction with the `ivotes` address that we passed into the constructor. this way, we delegate the voting rights to `ivotes` contract, that could be another custom logic or simply an erc20 contract implementing the standard.

also, this is something that i've seen new developers get stuck a lot, you need to ensure that the `ivotes` contract is correctly deployed. i've personally spent hours going back and forth debugging just to realize the `ivotes` contract was not initialized properly. double and triple check that you have actually deployed both contracts before trying to make them talk to each other.

now, one thing to be aware of is the actual interface of the `ivotes` contract. usually the `governorvotes` expects certain functions to be present in the `ivotes` contract, such as a `getvotes` function that receives an address and outputs the number of votes an address has. the `governorvotes` contract expects these standard to be implemented, but you can always implement your own custom logic there as long as it exposes the same interface and parameters.

let me show a very simple example of such a case, if your `ivotes` contract does not implement the standard, but has a custom logic inside.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/governance/Governor.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorVotes.sol";


interface ICustomVotes {
    function customGetVotes(address account) external view returns (uint256);
}


contract CustomGovernor is Governor, GovernorVotes {
    
    ICustomVotes public votesContract;

    constructor(address _votesAddress) 
    Governor("MyGovernor") 
    GovernorVotes(_votesAddress) {
        votesContract = ICustomVotes(_votesAddress);

    }


    function _getVotes(address account, uint256 blockNumber)
        internal
        view
        override
        returns (uint256)
    {
        // this is the important part, we're calling the customGetVotes function
        return votesContract.customGetVotes(account);
    }
}
```

in this example, the `governorvotes` uses `_getvotes` internally, which is what we override here to call our own function `customgetvotes`. notice that we're not passing a `blocknumber` here, because that's part of the openzeppelin interface and we're not using it in our `customgetvotes` function. it can be tricky but if you follow those patterns you'll be good.

to summarize, the basic idea of using an `ivotes` address in a `governorvotes` is actually just to pass it as a parameter in the constructor and then the contract itself is doing the hard work. if the `ivotes` contract implements a custom interface, like in the example above, it is just a matter of overriding the method `_getvotes` on the governor contract. there is no particular magic, it is just smart contract logic interacting with another smart contract logic.

i also wanted to give you some further reading, if you're looking for more in-depth information, i recommend looking at "mastering ethereum" by andreas antonopoulos and gavin wood. it’s a classic for a reason and it really helps build a solid understanding of the evm, and also if you're looking for more about governance pattern, there is a good paper by the openzeppelin team titled "the openzeppelin contracts wizard: a tool for generating smart contracts" that outlines governance patterns.

also, a little advice, always, and i mean always, test your contracts thoroughly. you don't want to deploy a contract that doesn't work as expected. use a development environment like hardhat or foundry and use their test capabilities. it's much cheaper to fix a problem there than on the actual blockchain, trust me. i once deployed a contract with a small mistake and i am paying for the gas to this day, it's like having to pay monthly rent for a tiny little mistake (ok, a tiny little bug).

that's it. i hope this helps you better understand how to use `ivotes` contract addresses in `governorvotes` contract. if you have other questions, just ask, i'm happy to help, especially if it is something i already struggled in the past.
