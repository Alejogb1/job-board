---
title: "How can I make BNB transfers with Nodejs?"
date: "2024-12-15"
id: "how-can-i-make-bnb-transfers-with-nodejs"
---

alright, so you're looking to fling some bnb around using nodejs, eh? i've been there, done that, got the t-shirt (and a few phantom transactions to prove it). it’s not rocket science, but there are a few gotchas that can really make things frustrating if you don't see them coming. let’s break this down.

first off, you need to interact with the binance smart chain (bsc). for that, we'll need a web3 library. `web3.js` is the usual go-to, and it’s what i'll be focusing on here. it's pretty comprehensive, and while it has its quirks, it works well enough once you get your head around it. you could also look at ethers.js, many prefer it nowadays, i personally am biased towards web3.js because it was the one i first learned.

now, before we even think about code, there's a crucial piece of the puzzle: your private key. please, for the love of everything holy, do *not* hardcode your private key into your script. i’ve made that mistake way back when i was younger and definitely less wise. i had this really cool proof of concept that i was sharing around with some fellow devs, i was so proud of it, and then one day i saw my test funds getting drained... ah, good times, good learning experience though. use environment variables, a secure vault, a hardware wallet, anything but directly embedding it in the code. you want to protect your digital goodies.

let’s get to the code. here’s a basic snippet to get you going with bsc and check the balance of an address:

```javascript
const Web3 = require('web3');

// infura or your node endpoint is a must
const bscRpcUrl = 'https://bsc-dataseed.binance.org/';
const web3 = new Web3(bscRpcUrl);

async function checkBalance(address) {
  try {
    const balanceWei = await web3.eth.getBalance(address);
    const balanceBnb = web3.utils.fromWei(balanceWei, 'ether');
    console.log(`balance for ${address}: ${balanceBnb} bnb`);
  } catch (error) {
    console.error('error fetching balance:', error);
  }
}

// put your wallet addres here
const myWalletAddress = '0xYourWalletAddressHere';
checkBalance(myWalletAddress);
```

this shows you how to connect to the bsc network and how to query a balance. note that i use `fromWei`, because the network talks in tiny units, like pennies, and we’d like to see it in actual bnb.

let's get to the meat of it. sending bnb. this is where the private key comes in. you need to sign transactions. here's how that looks:

```javascript
const Web3 = require('web3');
require('dotenv').config(); // if you choose the env variable way

// your node endpoint
const bscRpcUrl = 'https://bsc-dataseed.binance.org/';
const web3 = new Web3(bscRpcUrl);

// replace with your private key, ideally read from env
const privateKey = process.env.PRIVATE_KEY;
const senderAddress = web3.eth.accounts.privateKeyToAccount(privateKey).address;

async function sendBnb(recipientAddress, amountInBnb) {
  try {
    const amountInWei = web3.utils.toWei(amountInBnb, 'ether');

    const tx = {
      from: senderAddress,
      to: recipientAddress,
      value: amountInWei,
      gas: 21000,  // min gas required for a bnb transfer
    };

    // estimating gas (optional, but recommended)
    const gasEstimate = await web3.eth.estimateGas(tx);
    tx.gas = gasEstimate * 1.2; // let’s add a buffer

    const signedTx = await web3.eth.accounts.signTransaction(tx, privateKey);
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    console.log('transaction hash:', receipt.transactionHash);
    console.log('success, the transaction is sent!');
  } catch (error) {
    console.error('error sending bnb:', error);
  }
}

// put the recipient and amount here
const recipientAddress = '0xRecipientAddressHere';
const amountToSend = '0.001'; // in bnb

sendBnb(recipientAddress, amountToSend);
```

okay, so what is happening here? first, we build a transaction object, where we specify from, to and value, we also specify the gas limit, a fixed 21k is usually enough for a basic bnb transfer. we calculate the needed gas (estimating it), it is always a great practice, and it also makes sure we don't get errors when setting a hard limit. then, we sign this transaction using our private key (remember where you keep it, safe please). last, we send this signed transaction to the network. we get a transaction hash in return, which you can use to track the tx on a block explorer, like bscscan.

the gas thing is a common pain point. gas is what pays for the computation and it is always paid in native tokens. bsc is quite cheap, but if you’re doing something more complicated with smart contracts, gas estimation becomes super important and tricky. i remember one time i had some very complex calculations going on, in one of my smart contracts, and i wasn't estimating the gas. i lost bnb in multiple transactions before figuring out i needed to estimate it, instead of setting a static value. a pretty expensive learning session, if i am honest.

note, the `21000` is the base gas limit for simple transfers. if you're calling a smart contract, or sending more data, or if the network is congested, you might have to up the gas limit. better to estimate and add a little bit on top of the estimate so your transaction won’t be rejected.

now, one of the most common errors people experience are related to nonces. a nonce is basically a transaction count, and the network keeps track of the nonce for each account. every time you send a transaction, the nonce increments. if your local nonce is out of sync with the network, your transaction might be rejected. web3.js handles this most of the time automatically, but in specific situations you will have to handle it manually, which is a bit more complex. here's a way to ensure your nonce is correct.

```javascript
const Web3 = require('web3');
require('dotenv').config();

const bscRpcUrl = 'https://bsc-dataseed.binance.org/';
const web3 = new Web3(bscRpcUrl);

const privateKey = process.env.PRIVATE_KEY;
const senderAddress = web3.eth.accounts.privateKeyToAccount(privateKey).address;


async function sendBnbWithNonce(recipientAddress, amountInBnb) {
    try {
        const amountInWei = web3.utils.toWei(amountInBnb, 'ether');
        const nonce = await web3.eth.getTransactionCount(senderAddress);

        const tx = {
            from: senderAddress,
            to: recipientAddress,
            value: amountInWei,
            gas: 21000,
            nonce: nonce
        };
        const gasEstimate = await web3.eth.estimateGas(tx);
        tx.gas = gasEstimate * 1.2;

        const signedTx = await web3.eth.accounts.signTransaction(tx, privateKey);
        const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

        console.log('transaction hash:', receipt.transactionHash);
        console.log('success, transaction sent!');

    } catch (error) {
      console.error('error sending bnb:', error);
    }
}

const recipientAddress = '0xRecipientAddressHere';
const amountToSend = '0.001'; // in bnb
sendBnbWithNonce(recipientAddress, amountToSend);
```

see, what we did here? before sending the transaction, we used `web3.eth.getTransactionCount` to retrieve the current nonce of our address. then, we attach this value in the transaction object, ensuring that our local transaction’s nonce corresponds to the network’s expectations. there’s a subtle joke there related to blockchain consistency... if you got it. i am not explaining it though.

as for resources, while there are plenty of tutorials online, diving into the web3.js documentation directly is very helpful. it will save you tons of time in the future. also, have a look at the yellow paper of ethereum, it goes in depth into how transactions and contracts function at the lowest level. bsc works very similarly to ethereum. don’t treat it like a simple npm dependency. you must know your stuff.

and that’s mostly it. transferring bnb is not that complicated once you understand the flow of the process. remember the private key, remember the gas estimation, and remember the nonce. and always, always, double-check the address before sending the bnb. triple check if possible. you don't want to send your precious bnb to a blackhole.
