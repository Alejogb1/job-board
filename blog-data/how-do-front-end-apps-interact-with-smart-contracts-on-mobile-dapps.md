---
title: "How do front-end apps interact with smart contracts on mobile Dapps?"
date: "2024-12-16"
id: "how-do-front-end-apps-interact-with-smart-contracts-on-mobile-dapps"
---

Alright, let's break down how front-end applications, particularly on mobile dapps, interact with smart contracts. I've spent quite a bit of time in the trenches on this, most notably back in the early days of a project involving a distributed supply chain tracker. What seemed conceptually straightforward—a mobile app displaying product provenance—turned into a deeper dive into the nuances of web3 integration. The core challenge, as I’m sure you’re aware, is bridging the gap between the user interface, usually built with frameworks like react native or flutter, and the immutable code that defines the logic of your smart contract, typically written in solidity and deployed on a blockchain network like ethereum or polygon.

The primary mechanism for this interaction involves using a web3 library. These libraries are essential intermediaries which abstract away the low-level complexity of interacting directly with the blockchain. Think of them as translators; they take your javascript or dart commands and turn them into the necessary requests to the blockchain network and vice-versa. Most mobile dapps these days leverage something along the lines of `web3.js` for javascript based apps or `ethers.js` which offers more comprehensive functionalities, and a similar library might exist in dart if you're working with flutter.

On the front end, you first need to establish a provider. A provider is the gateway to the blockchain network. It's typically either an injected provider from a wallet extension or app—think metamask mobile or walletconnect—or a direct connection to an rpc node. If a user has a compatible mobile wallet installed, it can become the provider, allowing your application to securely send and receive transactions. The wallet handles the transaction signing process, so you don't have to deal with sensitive private keys within your app. This is crucial for security.

Let's take a look at a simplified example using `web3.js`. Suppose you've deployed a smart contract with a single function: `getGreeting()`, which returns a string. The solidity code might look something like this:

```solidity
// simplified contract example
pragma solidity ^0.8.0;

contract GreetingContract {
    string public greeting = "Hello, World!";

    function getGreeting() public view returns (string memory) {
        return greeting;
    }
}
```

Now, in your javascript-based front-end, you'd interact with it like this:

```javascript
// javascript code snippet
import Web3 from 'web3';

const contractAddress = '0xYourContractAddress...';
const contractAbi = [
  {
    "inputs": [],
    "name": "getGreeting",
    "outputs": [
      {
        "internalType": "string",
        "name": "",
        "type": "string"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  }
];

async function fetchGreeting() {
  if (window.ethereum) {
      const web3 = new Web3(window.ethereum);
      try{
        await window.ethereum.request({ method: "eth_requestAccounts" });
      }catch(error){
        console.error("user denied wallet connection", error);
        return;
      }


      const contractInstance = new web3.eth.Contract(contractAbi, contractAddress);
      try {
        const greeting = await contractInstance.methods.getGreeting().call();
        console.log("greeting:", greeting); // output: "Hello, World!"
      } catch (error) {
        console.error("error fetching greeting", error);
      }
  } else {
      console.log("no wallet found");
  }
}

fetchGreeting();
```

Here, we first check for `window.ethereum` which should be present if a wallet is connected to the browser. We then instantiate a `web3` object, request access to the user’s account using `eth_requestAccounts`, and finally create an instance of your contract. The `call()` function is for read-only interactions with your smart contract.

If you needed to modify state on the blockchain, let's say you had a function `setGreeting(string memory _newGreeting)` in your contract, you’d need to call the `send()` function, which triggers a transaction:

```javascript
// javascript code snippet
import Web3 from 'web3';

const contractAddress = '0xYourContractAddress...';
const contractAbi = [
  {
    "inputs": [
      {
        "internalType": "string",
        "name": "_newGreeting",
        "type": "string"
      }
    ],
    "name": "setGreeting",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  }
];

async function updateGreeting(newGreeting) {
    if (window.ethereum) {
        const web3 = new Web3(window.ethereum);
        try{
          await window.ethereum.request({ method: "eth_requestAccounts" });
        }catch(error){
          console.error("user denied wallet connection", error);
          return;
        }


        const contractInstance = new web3.eth.Contract(contractAbi, contractAddress);

        try {
            const tx = await contractInstance.methods.setGreeting(newGreeting).send({from: (await web3.eth.getAccounts())[0]});
            console.log("Transaction hash:", tx.transactionHash);
             //you would likely need a way to track the transaction, such as using web3.eth.getTransactionReceipt

        } catch (error) {
          console.error("error setting greeting", error);
        }
  } else {
      console.log("no wallet found");
  }
}

updateGreeting("A New Greeting!");
```

The key difference here is using `send()` instead of `call()`. The `send()` function requires a transaction to be signed by the user's wallet. The `from` address needs to be provided, and usually we get the first connected account. Note how the user’s wallet would then pop up to ask for a signature on this transaction. Crucially, you need to handle transaction status appropriately, likely polling using the transaction receipt to confirm whether the transaction was mined successfully, which may take a few seconds to a few minutes.

Now, if you're using flutter, the integration process will be somewhat similar, but you'll be using dart instead of javascript. You might leverage a flutter package like 'web3dart'. Let me give you a brief illustration:

```dart
// dart (flutter) code snippet
import 'package:web3dart/web3dart.dart';
import 'package:http/http.dart';

const String contractAddress = '0xYourContractAddress...';
const String contractAbi = '''
[
    {
        "inputs": [],
        "name": "getGreeting",
        "outputs": [
            {
                "internalType": "string",
                "name": "",
                "type": "string"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]
''';
Future<String?> fetchGreeting() async {
    final httpClient = Client();
    final ethClient = Web3Client("https://your-rpc-endpoint-here", httpClient);
     try{
        final contract = DeployedContract(
          ContractAbi.fromJson(contractAbi, 'GreetingContract'),
          EthereumAddress.fromHex(contractAddress),
        );

        final getGreeting = contract.function('getGreeting');
        final result = await ethClient.call(
            contract: contract, function: getGreeting, params: []);
         return result[0];
     }catch(e){
       print('Error: $e');
     } finally{
      httpClient.close();
     }

    return null;
}


void main() async {
  var greeting = await fetchGreeting();
  print('Greeting: $greeting'); //output: Greeting: Hello World!
}
```
In this dart example, we're using `web3dart`. Again, we connect to an rpc endpoint, specify the contract's address and ABI, and invoke the `getGreeting` function. Note the use of `http` client, as opposed to the injected provider of a browser-based app. This connection strategy is also useful in mobile contexts where user wallets may be a different connection strategy altogether (such as direct wallet connect via their API).

You’ll notice a recurring theme: you need the contract ABI and address, the correct web3 library, a provider, and an understanding of whether you're reading from the chain or modifying state. The details, obviously, vary based on your tools, but this interaction pattern remains fundamental.

If you're serious about diving deeper, I highly recommend looking at "Mastering Ethereum" by Andreas Antonopoulos and Gavin Wood – it covers the fundamentals. For more practical web3 javascript integration, I’d also look into the `web3.js` documentation as well as the `ethers.js` documentation, depending on which direction you're going with javascript. Also pay attention to the documentation specific to the library you use on flutter, like the `web3dart` documentation. They usually have great examples and best practices. Keep in mind, that this is a constantly evolving space, so always keep up with the latest versions of these libraries. You’ll want to understand more advanced concepts too, such as contract deployment from your application, transaction handling, and state management, as well as the implications of gas costs and transaction speed. That should give you a good start on the path to building robust and user-friendly dapps. Good luck!
