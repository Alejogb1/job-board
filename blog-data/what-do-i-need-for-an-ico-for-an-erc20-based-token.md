---
title: "What do I need for an ICO for an ERC20 based token?"
date: "2024-12-14"
id: "what-do-i-need-for-an-ico-for-an-erc20-based-token"
---

alright, so you're diving into the wonderful world of initial coin offerings, specifically with an erc20 token. i've been there, done that, got the t-shirt – and the lingering feeling of having needed more coffee. let's break down what you’ll actually need. forget the hype, we're talking practicalities here.

first, the token itself, this is the core. you’re saying erc20 which is good, it means you’re building on ethereum. you’ll need a smart contract that implements the erc20 standard. there are tons of examples out there but it's crucial to understand it instead of blindly copying and pasting. i made that mistake once, back in '17, thought i’d just tweak some code i found on a forum. ended up with a token that would mint to the moon but not transfer properly, what a headache. look, you need to understand the nuances of `transfer`, `approve`, `allowance`… it is not just about changing the name of the variables.

here is a snippet of what a basic erc20 contract might look like using solidity, the language used for ethereum smart contracts:

```solidity
pragma solidity ^0.8.0;

contract MyToken {
    string public name = "My Cool Token";
    string public symbol = "MCT";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(uint256 initialSupply) {
        totalSupply = initialSupply * 10 ** uint256(decimals);
        balanceOf[msg.sender] = totalSupply;
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(allowance[sender][msg.sender] >= amount, "allowance insufficient");
        require(balanceOf[sender] >= amount, "sender insufficient balance");

        allowance[sender][msg.sender] -= amount;
        balanceOf[sender] -= amount;
        balanceOf[recipient] += amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }
}
```

this shows you the basic functions. but for an ico, you will need to add some things, like a function to handle minting during the sale. so, you're going to need a *sale contract*, this is very important. don’t bake it into the token contract itself, you are asking for trouble. i once helped troubleshoot an ico where the sale logic was part of the token smart contract, things got very messy when they needed to adjust the price during the sale and we had to migrate the whole thing.

here's a snippet of how a basic sale contract might look:

```solidity
pragma solidity ^0.8.0;

import "./MyToken.sol"; // Assuming your token contract is named MyToken.sol

contract MyTokenSale {
    MyToken public token;
    address public owner;
    uint256 public tokenPrice; // Price in wei per token

    constructor(address _tokenAddress, uint256 _tokenPrice) {
        token = MyToken(_tokenAddress);
        owner = msg.sender;
        tokenPrice = _tokenPrice;
    }

    function buyTokens() public payable {
        uint256 tokensToBuy = msg.value / tokenPrice;
        require(tokensToBuy > 0, "not enough ether sent");
        token.transfer(msg.sender, tokensToBuy);
    }
    //only owner can set the price
    function setTokenPrice(uint256 newPrice) public  {
         require(msg.sender == owner,"not the owner");
         tokenPrice = newPrice;
    }

    // fallback function to receive ETH
    receive() external payable {}
}
```

remember, this is just a basic example. a proper sale contract will have things like whitelist management, different tiers, vesting schedules, and more complex price structures. make sure you test these thoroughly. local testnets like ganache are your friend. i didn’t use ganache once and i had to debug live contracts, it was not fun.

now, you also need a website, a slick one with all the details about your project. whitepapers, team info, the whole shebang. this is your first impression, so don't skimp on it. also, a lot of research into the tokenomics is key, how the tokens are allocated, the circulating supply, if there are any future minting plans, token burn mechanism etc. users need to know how the token is going to perform over time. this is very crucial.

on top of that, you have the marketing piece. i mean, you can have the best technology in the world, but if nobody knows about it, you're not going to get anywhere. you’ll need a community, engage in discussions, forums, and be active in various social media platforms. be ready for the questions. some can be quite… pointed.

legal paperwork. seriously, don't ignore this part. depending on your location and the regulations, you may need to consult with a lawyer who specializes in cryptocurrency and securities. and a proper legal review of your token smart contract can save you a whole lot of problems, trust me, my team got hit with some securities issues after ignoring this step. never again.

for security audits are vital. before you go public, have your smart contracts reviewed by a reputable firm. there are quite a few now, doing some good stuff, it's worth the investment. think of them as an extra pair of eyes (and brains) looking for any vulnerabilities. there are some books out there too that are great for security in smart contracts like "Mastering Ethereum" by Andreas Antonopoulos and Gavin Wood, they cover common mistakes you can easily make.

i always keep some resources handy that i consult whenever i have doubts: "programming blockchain" by jimmy song has some fantastic examples and explanations of how things work on the evm itself. and the official documentation on the solidity website.

for an actual simple example of how to interact with this contract using web3.js here is a quick guide:

```javascript
const Web3 = require('web3');

// Replace with your own provider and contract details
const providerUrl = 'YOUR_PROVIDER_URL';
const tokenAddress = 'YOUR_TOKEN_ADDRESS';
const saleAddress = 'YOUR_SALE_ADDRESS';

const web3 = new Web3(providerUrl);

// Define the contract ABI (Application Binary Interface)
const tokenAbi = [
    // ... (your ERC20 token ABI from compilation)
];
const saleAbi = [
    //... (your MyTokenSale contract ABI from compilation)
];

const tokenContract = new web3.eth.Contract(tokenAbi, tokenAddress);
const saleContract = new web3.eth.Contract(saleAbi, saleAddress);


async function buyTokens(amountInEther) {
    try {
        const accounts = await web3.eth.getAccounts();
        const account = accounts[0]; // Using the first account
        const weiValue = web3.utils.toWei(amountInEther, 'ether');

        const tx = await saleContract.methods.buyTokens().send({
            from: account,
            value: weiValue,
        });

        console.log('Transaction successful:', tx);
    } catch (error) {
        console.error('Error during token purchase:', error);
    }
}

//Example call: buyTokens('0.1'); //to buy tokens with 0.1 ethers for instance
// set new price for the sale
async function setNewPrice(newPriceInWei){
     try {
          const accounts = await web3.eth.getAccounts();
          const account = accounts[0];
           const tx = await saleContract.methods.setTokenPrice(newPriceInWei).send({
            from: account,
        });

         console.log('Transaction successful:', tx);
     } catch (error) {
        console.error('Error during price set:', error);
    }

}

//Example call: setNewPrice('1000000000000000'); //to set new price to 0.001 ethers per token for instance
```

i had a funny experience back then, we were preparing an ico for a client and during the testing phase it was discovered by a junior developer that he set the token decimal to 0, which means only whole numbers could be used, that day i really questioned my life choices. we had to re-deploy the token contract, a day of pure stress. so, learn from our mistakes and always be careful with details.

and the last bit, don’t forget your security practices. protect your keys, use multisig wallets for your project funds, and always double check everything. this is not some game, it’s actual money, often a lot of it, and people will go after it.

it's a complex process. not going to lie. but doable if you take it step by step and learn as you go. don’t just rush into it thinking you can get rich quick. it takes effort and proper preparation.
