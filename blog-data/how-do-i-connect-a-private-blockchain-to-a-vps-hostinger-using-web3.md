---
title: "How do I connect a private blockchain to a VPS (Hostinger) using web3?"
date: "2024-12-23"
id: "how-do-i-connect-a-private-blockchain-to-a-vps-hostinger-using-web3"
---

, let's unpack this. Connecting a private blockchain to a virtual private server, especially one from a provider like Hostinger, using web3 is a task I've tackled a few times over the years, and it often presents some unique considerations. It's not always as straightforward as connecting to a public network, but it's certainly achievable with a solid understanding of the components involved. Let's break down the essential steps and discuss the common hurdles.

Fundamentally, you're aiming to achieve two things: having your private blockchain node running on the VPS and then interacting with that node using web3, usually from a separate application or environment. The key here is ensuring your network is properly configured and accessible while maintaining its private nature. This usually means you'll need to manage your node's networking configuration and handle API accessibility with care, avoiding unwanted exposure. I recall one particularly tricky project where our local node worked perfectly, but transferring to a remote server revealed a subtle firewall issue that took a while to identify, so let's ensure we cover all the bases.

First, setting up the private blockchain node on the VPS is paramount. I've typically used Geth (go-ethereum) for private Ethereum networks, though the principles apply to other implementations like Parity or Hyperledger Fabric. You’ll need to generate a genesis file to define the parameters of your private blockchain. This file is crucial, as it defines your initial blocks, chain ID, and other foundational aspects. Once you have the genesis file, you can initialise your node:

```bash
geth --datadir ./mychain init ./genesis.json
```

This will set up the data directory for your node. After that, you can start the node with specifics to allow your web3 connection:

```bash
geth --datadir ./mychain --networkid 1234 --http --http.addr "0.0.0.0" --http.vhosts "*" --http.api "eth,net,web3,personal" --allow-insecure-unlock --mine --miner.threads 1 --syncmode "full" --port 30303  --http.corsdomain "*" --http.port 8545
```

Let me explain these flags, as they're fundamental:
*   `--datadir ./mychain`: Specifies the directory for storing blockchain data.
*   `--networkid 1234`: Sets the network id, this must match your genesis file, and ensure your other nodes also use this.
*   `--http`: Enables the HTTP-RPC server which will allow our web3 connection.
*  `--http.addr "0.0.0.0"`: This tells the server to listen on all available network interfaces, this makes it accessible to web3. **Use with caution on production systems and implement network security such as firewalls.**
*   `--http.vhosts "*"`: allows all vhosts. Again, **use caution in production systems and use host whitelists.**
*   `--http.api "eth,net,web3,personal"`: This defines which API modules are accessible through HTTP-RPC, it's what web3 interacts with.
*   `--allow-insecure-unlock`: Allows passwordless unlocks of accounts when developing. **Never use this for production.**
*   `--mine`: Starts the mining process. Important for testing and private chains to create new blocks.
*   `--miner.threads 1`: Specifies the number of mining threads.
*   `--syncmode "full"`: Ensures the node downloads all the blockchain data.
*    `--port 30303`: Specifies the port used for peer-to-peer node discovery on the network.
*    `--http.corsdomain "*"`: allows all domains for cors. **Again, use caution in production and use domain whitelists.**
*    `--http.port 8545`: Specifies the port used for HTTP communication, this is what we connect web3 to.

Remember, this configuration is intended for a development or test environment, and *it has several security vulnerabilities.* In a production setup, you'd severely restrict the HTTP interface, handle authentication, and implement a robust firewall.

Now, for the web3 interaction, you’ll need a web3 library (like web3.js for JavaScript/Node.js or web3.py for Python). Assuming you’re using JavaScript:

```javascript
const Web3 = require('web3');

const rpcURL = "http://<VPS_IP>:8545"; // Replace <VPS_IP>
const web3 = new Web3(rpcURL);

web3.eth.getBlockNumber()
.then((latestBlock) => {
  console.log("Latest block number:", latestBlock);
})
.catch( (err) => {
   console.error("Error connecting to blockchain:", err)
})
```

This snippet demonstrates a basic connection to your private blockchain node. Replace `<VPS_IP>` with the public IP address of your VPS. The `web3.eth.getBlockNumber()` method retrieves the latest block number, verifying your connection. It is critical you have opened port 8545 on the hostinger VPS to allow access. This can be done through the hostinger control panel for your VPS.
If you are running your node without having set the cors-domains, you may also encounter cors issues when trying to use web3 from a different domain.

Lastly, remember that private chains often require you to manually manage accounts. You may need to create new accounts on the node and fund them with some ether from the coinbase account. A key management strategy, often involving keystore files, is vital. You'll want to explore the `personal` api through the RPC interface for creating accounts. Web3 has methods to do this, which can be used to create and manage the user accounts to transact on your blockchain. Consider this example using Javascript and `web3.js`:

```javascript
const Web3 = require('web3');
const rpcURL = "http://<VPS_IP>:8545"; // Replace <VPS_IP>
const web3 = new Web3(rpcURL);

async function createAccount() {
    try{
         const account = await web3.eth.personal.newAccount('');
        console.log('New account address:', account);
        const coinbase = await web3.eth.getCoinbase();
         await web3.eth.personal.unlockAccount(coinbase, '', 0);
         let tx = await web3.eth.sendTransaction({
            from: coinbase,
            to: account,
            value: '1000000000000000000' // 1 ether (in wei)
        });
        console.log(`Sent 1 ether to new account, tx hash: ${tx.transactionHash}`)

     }
    catch(error){
        console.error('Error creating account:', error)
    }

}

createAccount()
```

This is a simple asynchronous function that demonstrates how to create a new account, and transfer funds into the account. The newly created address will be output. This could be used for the transaction from user to a contract. Remember, this uses insecure passwordless account management, avoid in production systems.
This setup should get you started but the devil is always in the details when you're working with infrastructure.

For those looking to delve deeper, I highly recommend *Mastering Ethereum* by Andreas M. Antonopoulos and Gavin Wood. It covers the intricacies of Ethereum, including private networks, in great detail. Also, the official *Ethereum Yellow Paper* by Gavin Wood provides the mathematical foundations and architecture of the Ethereum Virtual Machine, if you want a deeper dive into the low level concepts. For more on networking in distributed systems, any edition of *Computer Networking: A Top-Down Approach* by James F. Kurose and Keith W. Ross will be invaluable.

In closing, connecting a private blockchain to a VPS using web3 is absolutely doable, but it demands meticulous configuration and an understanding of networking and blockchain basics. Start with the basics, test every step, and always prioritise security, especially if the network is intended for a more permanent deployment.
