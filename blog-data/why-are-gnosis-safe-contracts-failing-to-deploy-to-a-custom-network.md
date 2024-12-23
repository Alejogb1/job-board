---
title: "Why are Gnosis Safe contracts failing to deploy to a custom network?"
date: "2024-12-23"
id: "why-are-gnosis-safe-contracts-failing-to-deploy-to-a-custom-network"
---

Okay, let's tackle this. I've certainly seen my share of gnosis safe deployment issues, and the "custom network" angle always adds a layer of intrigue. From my experience, it rarely boils down to a single, easily pinpointed cause. Usually, it's a confluence of factors that need careful examination. Let me walk you through what I've encountered and how I've approached these problems in the past.

The first, and perhaps most common, hurdle revolves around the foundational network setup itself. Often, when dealing with custom networks, we’re not leveraging the same pre-configured environment that Ethereum Mainnet or even a common testnet like Goerli provides. This means that assumptions made by the standard gnosis safe deployment scripts, specifically regarding network identifiers (chain ids) or availability of crucial precompiles, often prove inaccurate. I vividly recall one project where I was initially puzzled by deployment failures; it turned out that the custom network’s chain id was colliding with a different development network that was being used by a colleague.

We frequently see that the gas limits are not properly configured. When using custom networks, we may deal with a different throughput, potentially making the standard gas estimation functions inaccurate, even if the chain id is correctly configured. I found that a detailed review of the gas limit provided by the deployment tool, and even manual fine-tuning, often resolves this particular issue. Another time, the transaction’s gas price was simply too low, something that was overlooked because it was a private test network. Such subtle details can lead to deployments failing silently, leaving you scratching your head.

Let’s delve into the realm of crucial contract addresses. Gnosis Safe deployments rely on specific pre-deployed contracts such as the singleton libraries and the proxy factory. When you are not using a main network or common test network, you need to manually deploy these contracts first and then configure the deployment scripts to correctly recognize the addresses. Failure to do so would result in the safe contracts failing to deploy. In a past project, we spent considerable time tracking down a mismatch between the hardcoded addresses in the deployment scripts and the actual locations of the pre-requisite contracts, which were deployed to a custom dev network using a different process.

Here’s a snippet using web3.py to illustrate a simple check for network configuration issues, and how to ensure that the configuration settings are not colliding with other networks. This function should be run to check the environment in which the deployment will happen:

```python
from web3 import Web3

def check_network_config(rpc_url):
    """
    Checks the network configuration to identify common issues.
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    if not w3.is_connected():
        print("Error: Not connected to the RPC provider.")
        return False

    chain_id = w3.eth.chain_id
    block_number = w3.eth.block_number

    print(f"Connected to network with chain ID: {chain_id}")
    print(f"Current block number: {block_number}")

    # This can be further extended to check for expected precompile addresses

    if chain_id in [1, 5, 11155111]: # Check for well known mainnet and testnet ids
        print("Warning: This appears to be Mainnet or a common testnet. Ensure your intention is to deploy there.")

    return True

# Example usage
rpc_endpoint = "http://localhost:8545" # Replace with your custom network RPC
if check_network_config(rpc_endpoint):
    print("Network check complete. Proceed with deployment if everything looks correct.")
else:
    print("Network check failed. Review configurations before proceeding with deployment.")
```

Beyond network parameters, the bytecode of the deployed smart contracts can also contribute to issues. The standard deployment scripts will often assume a specific version of the solidity compiler is being used. A version mismatch might mean that the constructor arguments or the deployed bytecode is not what the scripts or the proxy is expecting. The bytecode will be different between solidity versions, and this can be hard to debug in the sense that the contract might deploy but not function correctly or revert later when executing a specific function. In one particular incident, we found that a library was compiled with a different solidity version than the primary contracts, creating subtle version conflicts that only manifested when the contracts were actively being used on the custom network. This issue required a complete recompile of the code base with a specific solidty compiler.

Here is another illustrative snippet, this time using ethers.js, showing how we can deploy the required singleton contracts and then use the correct addresses to deploy a gnosis safe:

```javascript
const { ethers } = require("ethers");
const Safe = require('@gnosis.pm/safe-contracts');

async function deploySafe(rpc_url, private_key) {
  const provider = new ethers.JsonRpcProvider(rpc_url);
  const wallet = new ethers.Wallet(private_key, provider);


    // Deploy required singleton contracts
  const singletonFactory = new ethers.ContractFactory(Safe.Singleton.abi, Safe.Singleton.bytecode, wallet);
    const singleton = await singletonFactory.deploy();
    await singleton.waitForDeployment();

  console.log(`Singleton Contract deployed to: ${singleton.target}`);


    const proxyFactory = new ethers.ContractFactory(Safe.ProxyFactory.abi, Safe.ProxyFactory.bytecode, wallet);
    const proxy = await proxyFactory.deploy();
    await proxy.waitForDeployment();

    console.log(`ProxyFactory Contract deployed to: ${proxy.target}`);

    // Now create safe
    const owners = [wallet.address];
    const threshold = 1;
    const safeFactory = new ethers.ContractFactory(Safe.Safe.abi, Safe.Safe.bytecode, wallet);

   // Initialize Safe with singleton address and proxy
     const safe = await safeFactory.deploy(
         singleton.target,
          proxy.target,
        owners,
        threshold,
        0,
        ethers.constants.AddressZero,
         ethers.constants.AddressZero,
         0,
        ethers.constants.AddressZero
     );
  await safe.waitForDeployment();


    console.log(`Safe deployed to: ${safe.target}`);
  }
    // Example usage
    const rpc_endpoint = "http://localhost:8545";
    const privateKey = "0xYOUR_PRIVATE_KEY";

   deploySafe(rpc_endpoint, privateKey)
  .then(() => console.log("Safe deployment complete."))
  .catch((error) => console.error("Safe deployment failed:", error));

```

Lastly, gas management deserves its own dedicated discussion. Custom networks often have different gas economics compared to public blockchains. The gas limits in the transaction, and the price in which you are willing to pay for gas can drastically influence the outcome of your deployment. During a past debugging process, it took quite a few trials before we determined that the gas limit for some of the initialisation transactions was way too low. Therefore, manually adjusting the gas limits during deployment can frequently solve the deployment issues.

Here's an example snippet to show how we can set the gas limit and gas price using web3.py

```python
from web3 import Web3

def deploy_with_gas_settings(w3, contract_abi, contract_bytecode, private_key, gas, gas_price):
    """
    Deploys a contract with specific gas settings.
    """

    contract = w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
    account = w3.eth.account.from_key(private_key)

    transaction = contract.constructor().build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': gas,
        'gasPrice': gas_price,
    })

    signed_tx = account.sign_transaction(transaction)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)


    if tx_receipt['status'] == 1:
        print(f"Contract deployed successfully at: {tx_receipt['contractAddress']}")
        return tx_receipt['contractAddress']
    else:
        print(f"Contract deployment failed. Transaction hash: {tx_hash.hex()}")
        return None
# Example usage
rpc_endpoint = "http://localhost:8545"
private_key = "0xYOUR_PRIVATE_KEY"

w3 = Web3(Web3.HTTPProvider(rpc_endpoint))
# Dummy ABI and bytecode (replace with your contract's)
dummy_abi = [{"inputs":[],"stateMutability":"nonpayable","type":"constructor"}]
dummy_bytecode = "0x6080604052348015600f57600080fd5b50603f80601d6000396000f3fe"

gas_limit = 4000000
gas_price = w3.eth.gas_price # use the suggested price

if w3.is_connected():
    contract_address = deploy_with_gas_settings(w3, dummy_abi, dummy_bytecode, private_key, gas_limit, gas_price)

    if contract_address:
        print("Contract successfully deployed!")
    else:
        print("Contract deployment failed.")
else:
        print("Connection error")

```

For deeper insights, I'd recommend diving into the official Ethereum documentation. "Mastering Ethereum" by Andreas Antonopoulos is also an incredibly useful resource. For a more in-depth understanding of smart contracts, consider "Solidity Programming Essentials" by Riad Attiyah. Additionally, I recommend reviewing the Gnosis Safe documentation, specifically the deployment section. These resources should provide you with more clarity on why your gnosis safe deployment might be failing.

Troubleshooting custom network deployments requires a methodical approach. I usually start by verifying the network configurations (chain id, gas limit, gas price), then meticulously validate the pre-deployed contract addresses and finally, I double check the bytecode and compiler versions. It's often a process of elimination, but understanding the common failure points can save a lot of time in the long run. Good luck, and feel free to reach out if you run into anything further; I'm always keen to help another developer navigate the sometimes-complex world of web3.
