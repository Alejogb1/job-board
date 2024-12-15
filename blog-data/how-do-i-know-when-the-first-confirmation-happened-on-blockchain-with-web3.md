---
title: "How do I know when the first confirmation happened on blockchain with web3?"
date: "2024-12-15"
id: "how-do-i-know-when-the-first-confirmation-happened-on-blockchain-with-web3"
---

hey there,

so, you're asking how to figure out when a transaction gets its first confirmation on a blockchain using web3? i've been there, trust me. it's one of those things that seems straightforward at first glance but has a few gotchas that can trip you up. i remember when i was first tinkering with ethereum back in the day, i was writing a simple dapp, trying to display transaction statuses and i spent a good chunk of a friday evening trying to figure out how to know when the transaction really went through, not just broadcasted.

the basic idea is this: when you send a transaction, it gets added to the mempool (a sort of waiting area for transactions). miners then pick up these transactions, validate them, and include them in a block. once a block is added to the chain, the transaction within it is considered confirmed. the first confirmation is when it gets included in the first block. it's like your grocery order finally getting picked up by the delivery guy, first confirmation, instead of being in the store waiting in the rack and being ready to be picked.

here's how you can generally handle this using web3, focusing on javascript. this assumes you’re using a web3 library like web3.js or ethers.js, i'm going to be using `web3.js` for the examples.

**1. sending the transaction and getting the receipt:**

first off, you send your transaction using web3. you'll typically get a transaction hash back immediately. this hash is like an id for your transaction and is useful to look up the receipt:

```javascript
const Web3 = require('web3');
// assuming you have a provider set up, like infura or alchemy
const web3 = new Web3(new Web3.providers.HttpProvider('your_provider_url'));

async function sendTransaction() {
  const accounts = await web3.eth.getAccounts();
  const fromAddress = accounts[0];
  const toAddress = '0x...your recipient...';
  const value = web3.utils.toWei('0.01', 'ether');

  const tx = {
    from: fromAddress,
    to: toAddress,
    value: value,
    gas: 21000, // typical gas for a simple eth transfer
  };


  try {
    const txHash = await web3.eth.sendTransaction(tx);
    console.log('Transaction hash:', txHash.transactionHash);
    return txHash.transactionHash;

  } catch (error) {
      console.error("error sending transaction:", error)
  }
}

async function main() {
   const hash = await sendTransaction()
   if(hash) {
     await waitForFirstConfirmation(hash)
   }
}
main();
```

remember, that hash you get back is not yet confirmed. it's just the transaction *id* to be broadcasted on the network.

**2. waiting for the confirmation**

now, the tricky part. we need to wait until this transaction is actually included in a block and gets its first confirmation. web3 provides an event listener method or similar approach which you can use. this method is like hanging out by the post office and waiting to see if your package has been delivered.

```javascript
async function waitForFirstConfirmation(transactionHash) {
  console.log(`waiting for confirmation for transaction: ${transactionHash}`);
    try {
       web3.eth.getTransactionReceipt(transactionHash)
       .then(receipt => {
          if (receipt && receipt.blockNumber) {
            console.log(`transaction confirmed in block: ${receipt.blockNumber}`);
            console.log("receipt:", receipt)

            // you can get the whole block data also:
            web3.eth.getBlock(receipt.blockNumber)
              .then(block => {
                console.log("block data:", block)
              })
              .catch(err => console.error("Error retrieving block data:", err))
          } else {
            // the transaction is still pending so keep polling
            console.log("transaction not yet confirmed, still waiting for the receipt...");
            setTimeout( () => waitForFirstConfirmation(transactionHash), 5000);
          }
        })
        .catch(err => console.error('Error retrieving receipt:', err));
    } catch (err) {
      console.error("something went wrong:", err);
    }
}
```
what this code snippet does, is basically asks the network every few seconds if the transaction is part of a block yet. when we get back the receipt containing the block number we can consider that is the first confirmation. in some cases when the network is congested the waiting time for a block confirmation may be higher.

**3. some considerations about the confirmation:**
   -   **gas prices and pending transactions:** users set a gas price for transactions. the higher the price, the faster the transaction will get picked by miners. sometimes it can get stuck in the mempool for long periods and become pending (some people joke it can become like a zombie transaction). if you are going to build a production system you would be better off giving the users some way to increase the gas price or cancel the pending transaction.

  - **network congestion:** the blockchain network can get congested. this will result in longer times for transaction confirmation. it will depend on the block time of the chain you are using, for ethereum, for example, the block time is roughly 13 seconds, but the actual confirmation times might vary. if i remember correctly this was way worse years ago...

  -   **confirmation depth:** usually when we are talking about production systems we don't usually wait for the *first* confirmation only. the blockchain is immutable because there is a probabilistic consensus mechanism, meaning the more blocks that get appended to the chain *after* our block the more difficult it is to revert that block. so it is usual to wait for a specific number of confirmations like 6, or 12, or even more. the deeper it gets, the higher the level of security. there is a tradeoff between speed and safety, you choose the best parameters for your use case.

**a quick code summary:**

here is the whole thing put together as an example, and it should just work if you paste it into a node.js environment if you have web3 installed and configured:

```javascript
const Web3 = require('web3');

const web3 = new Web3(new Web3.providers.HttpProvider('your_provider_url'));


async function sendTransaction() {
    const accounts = await web3.eth.getAccounts();
    const fromAddress = accounts[0];
    const toAddress = '0x...your recipient...';
    const value = web3.utils.toWei('0.01', 'ether');

    const tx = {
      from: fromAddress,
      to: toAddress,
      value: value,
      gas: 21000,
    };

    try {
      const txHash = await web3.eth.sendTransaction(tx);
      console.log('Transaction hash:', txHash.transactionHash);
      return txHash.transactionHash;

    } catch (error) {
        console.error("error sending transaction:", error)
    }
  }

async function waitForFirstConfirmation(transactionHash) {
    console.log(`waiting for confirmation for transaction: ${transactionHash}`);
      try {
         web3.eth.getTransactionReceipt(transactionHash)
         .then(receipt => {
            if (receipt && receipt.blockNumber) {
              console.log(`transaction confirmed in block: ${receipt.blockNumber}`);
              console.log("receipt:", receipt)

              // you can get the whole block data also:
              web3.eth.getBlock(receipt.blockNumber)
                .then(block => {
                  console.log("block data:", block)
                })
                .catch(err => console.error("Error retrieving block data:", err))
            } else {
              // the transaction is still pending so keep polling
              console.log("transaction not yet confirmed, still waiting for the receipt...");
              setTimeout( () => waitForFirstConfirmation(transactionHash), 5000);
            }
          })
          .catch(err => console.error('Error retrieving receipt:', err));
      } catch (err) {
        console.error("something went wrong:", err);
      }
  }

async function main() {
     const hash = await sendTransaction()
     if(hash) {
       await waitForFirstConfirmation(hash)
     }
  }
  main();
```

**resources:**

for a more in-depth understanding of blockchain concepts, you might find "mastering bitcoin" by andreas antonopoulos a helpful read. the official web3.js or ethers.js documentation are, of course, a great resource to understand the specifics of their api. for a deeper dive into the consensus mechanism and probabilistic finality of blockchains in general, papers and resources from the distributed systems field should also be checked, like the works of nancy lynch. there are many many academic papers about it, but that subject is usually a bit outside the day-to-day development work, but it's good to know the underlying math and computer science principles for the better understanding.

hope this helps! happy coding.
