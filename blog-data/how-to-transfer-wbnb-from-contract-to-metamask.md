---
title: "How to transfer wbnb from contract to metamask?"
date: "2024-12-15"
id: "how-to-transfer-wbnb-from-contract-to-metamask"
---

alright, so you're looking to move wrapped bnb (wbnb) from a smart contract to a metamask wallet. this is a pretty common task when interacting with decentralized applications, and it often trips people up the first time, so let's break it down.

i've been there, trust me. early days of defi, i was moving tokens around like a drunken sailor on a trampoline. i once accidentally sent a bunch of eth to a contract address, thinking it was my own, classic rookie mistake. had to wait a few days for a dev to manually return it. it was a learning experience, to say the least. now, i’m pretty meticulous about these things.

the core of this problem boils down to understanding that contracts don't just "push" tokens to your wallet. instead, your wallet needs to "pull" the tokens from the contract, using the correct function call. it involves two crucial pieces: the contract address you are pulling the tokens from and the function you have to call.

first thing, we have to interact with the wbnb contract itself. wbnb, like other erc-20 tokens, has an interface that allows you to transfer them. think of it like the token having a specific set of functions anyone can call as long as the parameters are correct. this contract interface exposes a transfer function. let's call it `transfer`.

the `transfer` function will be what we need. now, the critical thing about `transfer` is that it requires at least two parameters: the recipient address and the amount of tokens to be transferred. so here's the deal, you’ll need your metamask wallet address as the recipient, and the amount of wbnb in wei (the smallest denomination of bnb).

to make it happen programmatically, let’s see the code. i'm going to show you how this is typically done using web3.js, which is a common library for interacting with ethereum-compatible blockchains like bsc, where wbnb exists.

here's a basic javascript example using web3.js:

```javascript
const Web3 = require('web3');

// your metamask private key (keep this secret!)
const privateKey = 'YOUR_PRIVATE_KEY';

// your metamask address, where the wbnb will be sent
const recipientAddress = 'YOUR_METAMASK_ADDRESS';

// the wbnb contract address
const wbnbContractAddress = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c'; // this is the wbnb contract on bsc mainnet

// the amount of wbnb you want to transfer, in wei, e.g. 1 wbnb = 10^18 wei
const amountToTransfer = '1000000000000000000'; // represents 1 wbnb

// setting up web3 with your node url, replace with your own.
const web3 = new Web3(new Web3.providers.HttpProvider('https://bsc-dataseed.binance.org'));

// get the address associated with your private key
const account = web3.eth.accounts.privateKeyToAccount(privateKey);

// the wbnb abi (application binary interface) for the contract
const wbnbAbi = [
    {
        "constant": false,
        "inputs": [
          {
            "name": "_to",
            "type": "address"
          },
          {
            "name": "_value",
            "type": "uint256"
          }
        ],
        "name": "transfer",
        "outputs": [
          {
            "name": "",
            "type": "bool"
          }
        ],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
      }
];

// getting an instance of the wbnb contract
const wbnbContract = new web3.eth.Contract(wbnbAbi, wbnbContractAddress);

async function transferWbnb() {
    // creating the transaction
    const tx = {
        from: account.address,
        to: wbnbContractAddress,
        gas: 200000, // adjust this based on gas estimation
        data: wbnbContract.methods.transfer(recipientAddress, amountToTransfer).encodeABI(),
    };

    // sign the transaction with your private key
    const signedTx = await web3.eth.accounts.signTransaction(tx, privateKey);

    // broadcast the transaction
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log('transaction hash:', receipt.transactionHash);
    console.log('transaction complete.');
}

transferWbnb();
```

*replace `YOUR_PRIVATE_KEY` and `YOUR_METAMASK_ADDRESS` with your actual values. keep your private key very secure*. the `amountToTransfer` is in wei, which is the smallest denomination. if you want to transfer 1 wbnb, then you would need to set the value to 1000000000000000000. if you are using a local node or testnet, make sure that the bsc node url is changed.

this script uses the `transfer` function of the wbnb contract to move the tokens. it will take some gas (bnb) to perform the transaction. the amount of gas depends on the network, but usually the default gas limits work fine, and you can change the `gas: 200000` setting to something suitable. also, note that the wbnb abi provided in the example is a very minimal abi, in a production system, we would use the complete wbnb contract abi. i have provided the `transfer` function only for illustration.

now, let's say you want to do this with ethers.js, another very common library. here's the equivalent:

```javascript
const { ethers } = require("ethers");

// your metamask private key (keep this secret!)
const privateKey = 'YOUR_PRIVATE_KEY';

// your metamask address
const recipientAddress = 'YOUR_METAMASK_ADDRESS';

// the wbnb contract address
const wbnbContractAddress = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c'; // wbnb contract on bsc

// the amount of wbnb to transfer, in wei
const amountToTransfer = ethers.parseUnits("1", "ether"); // 1 wbnb

// using bsc rpc
const provider = new ethers.JsonRpcProvider('https://bsc-dataseed.binance.org');

const wallet = new ethers.Wallet(privateKey, provider);

// abi of the transfer function
const wbnbAbi = [
  "function transfer(address _to, uint256 _value) public returns (bool)"
];

// create contract
const wbnbContract = new ethers.Contract(wbnbContractAddress, wbnbAbi, wallet);

async function transferWbnb() {
  try {
    const tx = await wbnbContract.transfer(recipientAddress, amountToTransfer);
    await tx.wait();
    console.log('transaction successful:', tx.hash);
    console.log('transaction complete.');
  } catch (error) {
    console.error('error sending transaction:', error);
  }
}
transferWbnb();

```

*replace `YOUR_PRIVATE_KEY` and `YOUR_METAMASK_ADDRESS` with your actual values. keep your private key very secure*. similar to web3.js, you need to provide a provider for the node and then set your wallet with your private key. then, you can call the `transfer` function on the contract instance. the `ethers.parseUnits` is used to format the amount of wbnb to be sent in the correct wei format.

both of these code snippets require you to have node.js and npm installed, and you'll need to install web3.js or ethers.js via npm (`npm install web3` or `npm install ethers`). they can be adapted to use in a node environment.

now, i know you might not be using javascript and might be in another environment. if that's the case, you should take the core idea of it to your own environment or programming language. there's a bunch of libraries out there in other languages that allows interacting with smart contracts. the core concept will remain the same, we are just calling the transfer function.

finally, if you're more of a visual person and prefer a ui, you could use something like metamask itself to interact with the contract. but for that you would have to have the complete contract abi for the wbnb contract. metamask only allows writing to functions with complete abi. another option is using third party blockchain explorers that has a contract write feature in order to call the `transfer` function but keep in mind that that this means revealing your private key so it's not recommended.

resources? well, i usually refer to the official documentation of web3.js and ethers.js, they are pretty good and always updated. for more general information on smart contracts, i suggest you read "mastering ethereum" by andreas antonopoulos and gavin wood, it covers the technical aspects of ethereum and smart contracts very well, it also has a lot of information on gas and how it works which might be good for you. also for specific implementation details of a given smart contract, you should always consult the official whitepaper or documentation of the project or token, they almost always include all the technical implementation details.

one last thing i always tell newcomers: always test these things on a testnet first, if you can. it saves a lot of headache and probably the loss of some real tokens. and if you end up sending it to the wrong address, at least you lost test tokens instead of the real thing, which is the equivalent of losing money. that's why we test things, right? to make sure stuff works and not for "production debugging," like we jokingly say, well i guess it's not that funny. happy coding!
