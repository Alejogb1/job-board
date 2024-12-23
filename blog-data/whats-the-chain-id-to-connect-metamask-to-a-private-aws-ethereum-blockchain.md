---
title: "What's the chain ID to connect Metamask to a private AWS Ethereum Blockchain?"
date: "2024-12-16"
id: "whats-the-chain-id-to-connect-metamask-to-a-private-aws-ethereum-blockchain"
---

,  It's a question I've seen crop up a number of times, and the details matter quite a bit, more so than what you might initially assume when thinking about just "connecting MetaMask." It's not simply a case of plugging in an arbitrary number. You're venturing into private blockchain territory, which means the usual public chain IDs won't do. In my experience, back when I was working on a supply chain traceability project, we had our own private Ethereum network hosted on AWS, and the process of connecting external wallets was a lesson in understanding the nitty-gritty of chain configuration.

The chain id, in its essence, is a unique identifier, typically an integer, that differentiates one blockchain from another. It's a crucial part of transaction signing, preventing replay attacks across different networks. When connecting MetaMask to a public chain, the process is usually seamless; the wallet app already has the common chain IDs (like 1 for the main Ethereum network, 5 for Goerli, etc.) baked in. However, with your private AWS Ethereum network, you're working outside the predefined parameters.

The crucial thing to understand here is that *you* define the chain id when you initialize your private blockchain. It's not something AWS magically generates for you. When using tools like `geth` (go-ethereum) or `parity` (now OpenEthereum) to launch your network, you specify this value in your genesis block configuration. If you’re working with Hyperledger Besu, you’d configure it there. Typically, you see developers choose chain ids that are unlikely to collide with public chains, so anything beyond 1000 is common, but ultimately any valid number will work. Remember, collisions could cause issues for your users if they try to process transactions on the wrong network.

Now, how do you actually find out what your chain id is if you didn't set it up, or perhaps, it's been a while since you configured the network? There are a few practical approaches.

First, if you have direct access to the genesis block json file, the chain id is explicitly stated within. Look for the `chainId` or `chainid` parameter, depending on the blockchain client. It's usually right near the top of the configuration. If you're unsure what this configuration looks like, I recommend reviewing the documentation of the Ethereum clients like go-ethereum and Besu. Specifically, "Configuring the Genesis Block," as it is very often documented there.

Second, and this is more practical for ongoing maintenance, you can extract the chain id by making a `net_version` JSON-RPC call to your private network's endpoint. This method is client-agnostic, and it's how MetaMask and other wallets reliably get this crucial piece of information. I’ve frequently found this to be the most resilient method.

Here are three examples demonstrating how to extract this chain id using various methods, along with code in different languages for practical application.

**Example 1: Python using web3.py**

This snippet demonstrates how to connect to your private Ethereum network using the web3 library in python and retrieve the chain id.

```python
from web3 import Web3

# Replace with your private network's RPC endpoint.
rpc_endpoint = "http://<your-private-node-ip>:<port>"

w3 = Web3(Web3.HTTPProvider(rpc_endpoint))

if w3.is_connected():
    chain_id = w3.net.version
    print(f"Chain ID: {chain_id}")
else:
    print("Failed to connect to the RPC endpoint.")

```

This Python snippet is straightforward. The `web3` library handles the rpc communication, and the `net.version` returns the chain id. Ensure you install the web3 library beforehand using `pip install web3`. Replace `<your-private-node-ip>:<port>` with the actual rpc endpoint of your private Ethereum network.

**Example 2: Javascript using ethers.js**

This example shows how to fetch the chain id with the ethers.js library. You can run this in a browser environment or using node.js.

```javascript
const ethers = require('ethers');

// Replace with your private network's RPC endpoint.
const rpcEndpoint = "http://<your-private-node-ip>:<port>";

async function getChainId() {
    const provider = new ethers.JsonRpcProvider(rpcEndpoint);
    try {
        const network = await provider.getNetwork();
        console.log(`Chain ID: ${network.chainId}`);
    } catch (error) {
        console.error("Error:", error);
    }
}

getChainId();
```

Here we are using the ethers.js library. The `ethers.JsonRpcProvider` handles the connection. We then use the `getNetwork()` method that provides an object containing the chain id. Remember to install ethers using `npm install ethers` if you choose to run this in a node.js environment. Replace `<your-private-node-ip>:<port>` with your private network endpoint.

**Example 3: Inspecting the genesis.json**

This third method is not code, but it is very important. I mentioned before that the chain id is in the genesis block. Here is an example of a typical genesis block.

```json
{
  "config": {
    "chainId": 1337,  // This is where the chain ID is defined
    "homesteadBlock": 0,
    "eip150Block": 0,
    "eip155Block": 0,
    "eip158Block": 0
  },
  "difficulty": "1",
  "gasLimit": "8000000",
  "alloc": {
        "0x1234567890123456789012345678901234567890": { "balance": "1000000000000000000000000" }
    }
}
```

Here the "chainId" parameter is configured as 1337. You'd access this file locally or through your network's configuration management system, and then extract the chain id directly.

Once you've retrieved your chain ID, you'll need to manually add the network to MetaMask. You do this through the MetaMask settings by clicking "Add Network", and entering the RPC URL (the same URL used in the python and Javascript examples), the chain id, the name for the network, and the currency symbol (usually ETH or your custom currency symbol). With this setup, MetaMask will recognize your private chain and allow you to interact with it.

To deepen your understanding, I highly recommend reading the official documentation for the Ethereum clients you are using (go-ethereum and Besu being the most common). Specifically, the sections covering genesis block configuration and JSON RPC specifications are extremely helpful. Also, the Ethereum yellow paper provides a mathematical basis for how transactions and blocks are constructed, including a good discussion of chain id. Finally, delving into the EIP-155 and EIP-159 specifications is also a good choice since these specifically detail the network identifier and transaction signing changes.

In conclusion, while connecting Metamask to a private AWS Ethereum blockchain involves a bit more configuration than a public network, the process is well-defined and reproducible once you understand the fundamental concepts of chain ids and JSON RPC. Knowing how to extract this information with practical tools like the code snippets outlined above makes integrating MetaMask with your private blockchain a much more manageable task.
