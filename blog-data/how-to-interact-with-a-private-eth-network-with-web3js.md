---
title: "How to interact with a private eth network with web3js?"
date: "2024-12-15"
id: "how-to-interact-with-a-private-eth-network-with-web3js"
---

alright, so, interacting with a private ethereum network using web3.js, yeah i've been down that rabbit hole more times than i care to remember. it's usually not as straightforward as connecting to mainnet, especially when you're dealing with custom genesis files, specific network ids, or, god forbid, a bunch of nodes that are being temperamental. been there, done that, got the t-shirt (and the eye twitch).

first things first, you can't just throw `web3 = new Web3(Web3.givenProvider || 'http://localhost:8545');` and expect it to magically work. that's the mainnet/testnet defaults. connecting to your own private network needs a bit more explicit configuration. what i typically do is create a web3 instance pointing directly to my rpc node and that looks something like this:

```javascript
const Web3 = require('web3');
const rpcURL = 'http://<your_node_ip_or_hostname>:<your_node_port>'; // like 'http://192.168.1.100:8545' or 'http://my-node.local:8545'
const web3 = new Web3(new Web3.providers.HttpProvider(rpcURL));
```

notice that i'm using `HttpProvider` specifically. this assumes your nodes are providing an http rpc endpoint. if you're using websockets, well, that's another story, and you'll need to use `WebsocketProvider` and that looks like this:

```javascript
const Web3 = require('web3');
const wsURL = 'ws://<your_node_ip_or_hostname>:<your_node_ws_port>'; // like 'ws://192.168.1.100:8546' or 'ws://my-node.local:8546'
const web3 = new Web3(new Web3.providers.WebsocketProvider(wsURL));
```

when i first started playing with private networks, i had a weird issue where i kept getting timeouts. turned out, my geth nodes were on different machines, and i had not configured my firewall correctly for the rpc ports. i lost a whole evening diagnosing that and learned a painful but valuable lesson. always, always double check your network configurations. firewall rules can be your worst enemy and that includes your operating system firewall. i usually keep them off when i'm debugging, but i definitely don't recommend it when you're not.

after you have connected to your node, double check the connection by grabbing the network id. this confirms that your web3 instance is indeed talking to your private network and not some other random endpoint. you can do that with:

```javascript
web3.eth.net.getId()
  .then(networkId => {
    console.log(`connected to network with id: ${networkId}`);
    // you should get your custom network id here
  })
  .catch(error => {
    console.error('error fetching network id:', error);
    // print the full error object
  });
```

the key here is to know that your private network *probably* has a different network id than the mainnet (which is `1`). you should know what that is from how you initialized the network. if you're using geth, it’s usually the value you configured in your genesis file. if you are using other clients like parity or nethermind make sure you configured it on that end. i once spent hours thinking i had successfully connected to my private network only to find i was still connected to mainnet because i had not properly configured the genesis id in my client and i was trying to check against the mainnet id of one. i felt so dumb that day.

now, for the transaction part, this is where things usually get a bit hairy. with private networks, you probably are using prefunded accounts that you have on your chain. if you are using geth you might have to use the `personal` api for signing and then send the transaction from the account. here's how that can work:

```javascript
const accountAddress = '<your_account_address>'; // the account you want to use
const recipientAddress = '<recipient_address>';
const valueToSend = web3.utils.toWei('0.01', 'ether'); // send 0.01 ether
const gasLimit = 21000;

web3.eth.personal.unlockAccount(accountAddress, '<your_account_password>', 60)
  .then(() => {
    return web3.eth.sendTransaction({
      from: accountAddress,
      to: recipientAddress,
      value: valueToSend,
      gas: gasLimit
    });
  })
  .then(transactionReceipt => {
    console.log('transaction mined:', transactionReceipt);
  })
  .catch(error => {
      console.error('transaction error:', error)
      // print the error for debugging
  });
```

make sure the password here is the correct password and that your geth client exposes this api and that your account has enough funds. if you do not want to use the `personal` api you need to sign the transaction before sending it. that looks like this:

```javascript
const accountAddress = '<your_account_address>'; // the account you want to use
const privateKey = '<your_account_private_key>'; // the account's private key
const recipientAddress = '<recipient_address>';
const valueToSend = web3.utils.toWei('0.01', 'ether'); // send 0.01 ether
const gasLimit = 21000;

web3.eth.getTransactionCount(accountAddress)
    .then(nonce => {
        const transaction = {
            from: accountAddress,
            to: recipientAddress,
            value: valueToSend,
            gas: gasLimit,
            nonce: nonce,
        }
    return web3.eth.accounts.signTransaction(transaction, privateKey)
    })
    .then(signedTransaction => {
        return web3.eth.sendSignedTransaction(signedTransaction.rawTransaction);
    })
    .then(transactionReceipt => {
      console.log('transaction mined:', transactionReceipt);
    })
    .catch(error => {
        console.error('transaction error:', error)
        // print the error for debugging
    });

```

when dealing with local development, i usually use a ganache node, it creates a development network with prefunded accounts, and the web3 integration is a bit more straightforward. however, a lot of times you are not using development environments you are dealing with live private networks and you need to connect to it. sometimes i have problems with infura or alchemy when dealing with a private network. they work fine with public networks, but when it comes to private ones sometimes the connection isn't as smooth. i once was debugging why i could not connect to my private network and it turns out the vpn was blocking my client from connecting to the network and i was not thinking about it.

for learning more about private networks, especially the intricacies of genesis files and network ids i would suggest going into the official documentation of the different clients, like geth, parity, besu or nethermind, those are all excellent resources. specifically, looking into the documentation around genesis files and network configurations is key. for the web3 side of things, "mastering ethereum" by andreas antonopoulos and gavin wood is an excellent deep dive. you can skip around it as needed. also read the official web3.js documentation, it has an incredible amount of information. but don't get lost in the details, sometimes, a good old fashioned "hello world" program using web3 with your private network can be a game changer to understand what is happening with your private setup.

connecting to private ethereum networks is, without any doubt, not a five-minute task. you have to deal with network configurations, nodes, firewalls, clients, and private keys; it's like a big puzzle you have to piece together each time, but after some practice it will come naturally to you.

oh, and before i forget, a blockchain walked into a bar and ordered a drink. the bartender said "sorry, we don’t accept transactions here" it was a private network, so he couldn't pay.
