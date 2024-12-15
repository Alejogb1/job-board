---
title: "Why are NEAR tokens stuck in a staking pool - and how to release the tokens?"
date: "2024-12-15"
id: "why-are-near-tokens-stuck-in-a-staking-pool---and-how-to-release-the-tokens"
---

alright, so you're having the classic near tokens-stuck-in-staking-pool situation, i've been there, trust me. it’s not a bug; it’s more like a feature, a frustrating one. let me break it down from my experiences, and how i've usually untangled myself from it.

first off, near staking is not like just keeping your coins in a wallet. you’re essentially locking them up to help secure the network, and in return, you get staking rewards. but, this locking has some implications. when you stake, your tokens aren’t immediately available for transfer. they go into a specific smart contract associated with that staking pool.

this is where things get a bit tricky. the “stuck” feeling comes from the way near's unstaking process works. it's not instant. there’s a cool-down period, a sort of unbonding phase. this period exists to prevent malicious actors from quickly unstaking and causing instability within the network. usually, it’s around 36 to 72 hours, but it can vary a bit depending on the specifics of the pool or if the network itself has some changes or updates.

now, if your tokens are "stuck," they are most likely in this unbonding period. they're technically *not* stuck, but they are also *not* readily available. you initiated an unstake, and now the network is processing it. the funds are locked in the smart contract until the unbonding window closes, and then they get released back to your available balance. this happened to me back in 2021. i was experimenting with different pools, trying to optimize yield, and completely forgot about the unbonding period. spent a whole day wondering where my near vanished until i remembered my previous unstaking operation. since then i usually keep a spreadsheet to track these things.

let's look at how you generally initiate an unstake, and then how to actually see the pending operations. usually, you do this via the near-cli, or the near wallet web interface, or using a custom client if you're more of a hardcore coder. i'll show you the command line way first because it's the most transparent:

```bash
near call <your_staking_pool_account_id> unstake '{"amount": "<your_unstake_amount>"}' --accountId <your_account_id> --depositYocto 1
```
replace `<your_staking_pool_account_id>`, `<your_unstake_amount>`, and `<your_account_id>` with the appropriate values. `<your_unstake_amount>` is the amount you want to unstake, you need to enter the amount in yocto near. 1 near is equivalent to 10^24 yocto near.

now the tricky part, knowing that it was successful and how much time is left until your balance is updated with the unstaked tokens. you can use the command below, to check your stake status and if you have any pending unstaking operations. the important thing is the `unstake_available_epoch`

```bash
near view <your_staking_pool_account_id> get_account_staked_balance '{"account_id": "<your_account_id>"}'
```
again replace the placeholder accordingly. the output will be a json object and among others you should have a value named `unstake_available_epoch`, this indicates the epoch in which your tokens will be available. you can check the current epoch using:
```bash
near view blockchain_info
```
the response has a `epoch_id`, if the value `unstake_available_epoch` is greater than the `epoch_id` then your tokens are still in the unbonding period.

if you prefer to interact programmatically, there are near sdk libraries available. i've used the javascript one extensively for scripting some data analyses on my near staking activities.

```javascript
const { connect, keyStores, WalletConnection, Contract } = require("near-api-js");

const config = {
  networkId: 'mainnet',
  keyStore: new keyStores.InMemoryKeyStore(),
  nodeUrl: 'https://rpc.mainnet.near.org',
  walletUrl: 'https://wallet.mainnet.near.org',
};
const your_account_id = '<your_account_id>'; // replace it
const your_staking_pool_account_id = '<your_staking_pool_account_id>' // replace it

async function checkStakingStatus() {
  const near = await connect(config);
  const walletConnection = new WalletConnection(near, 'my-app');
  const account = await near.account(your_account_id);
  const contract = new Contract(account, your_staking_pool_account_id, {
    viewMethods: ["get_account_staked_balance"]
  });

  try {
    const result = await contract.get_account_staked_balance({account_id:your_account_id});
    console.log('staking status:',result)
    }
  catch (error) {
    console.error('error getting staking status:',error);
    }

  const blockchain = await near.connection.provider.status()
  console.log('blockchain status:', blockchain.sync_info)
}

checkStakingStatus();
```

in this example the code connects to the near network and uses the staking pool smart contract to check your status. of course, you'll need a `package.json` with the `near-api-js` library. you can install it with `npm i near-api-js`. you can adapt it to perform the unstake operation, however, i'll suggest the command line approach, for testing. you can use the `near-api-js` library and the official documentation to check how to accomplish this, it's a bit more advanced but is a good exercise. you need to figure out how to pass `near-api-js` the private key of your account and create an instance of a contract using `account.functionCall` to call the `unstake` method. but i'm not going into those details now.

if your near aren't released after the unbonding period has passed, *then* you might have a bigger problem. but that’s relatively rare. usually, patience is the solution. i learned that lesson the hard way, one sunday morning i was setting up some automation for staking and unstaking and i thought something was wrong with my scripts but it was just my forgetfulness. sometimes i think i'm just an interface between a chair and a computer.

now, for resources. i'd recommend digging into the official near protocol documentation. it has an extensive explanation about staking, unbonding periods and all the technicalities involved. it's available online. if you want a more in-depth dive into the underlying smart contracts, check the near protocol whitepaper, it's a bit more technical and involves more advanced concepts related to blockchain but the investment pays off. look for the staking and economics section to clarify your knowledge on the topic. in terms of books, 'mastering blockchain' by andreas antonopoulos, can be a good start to understand blockchain mechanics. and although it's focused on bitcoin you can apply a lot of the concepts to the near network. also check 'ethereum programming' from eliezer dahan and daniel hwang if you want to understand more advanced topics related to smart contracts.

remember to double-check the staking pool you're using and its policies. some pools might have slightly different rules or longer unbonding periods. if you have any doubts, it's better to reach out to the pool’s community or support channels. i usually try to avoid discord channels. in terms of information sources, usually the official ones are better because it avoids biased or outdated information. but as always, do your own research.

to wrap it up, your near isn’t probably "stuck" but rather in the process of unbonding. check the pending transactions or status, be patient, and make sure you're familiar with the time it takes to unbond. i hope this helps!
