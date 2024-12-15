---
title: "What is the purpose of a new EVM node creation?"
date: "2024-12-15"
id: "what-is-the-purpose-of-a-new-evm-node-creation"
---

alright, so, you're asking about the purpose of spinning up a fresh evm node, eh? i’ve been around the block a few times with this stuff, so let me break it down from my perspective, the trenches, so to speak.

basically, a new evm (ethereum virtual machine) node is all about either joining an existing network or creating a new, isolated one. why would you want to do that? well, it depends on what you're trying to achieve. it’s not a one-size-fits-all kind of deal. i've seen this used for a whole range of things, from simple local dev environments to full-blown production deployments.

from my experience, i remember working on a project back in '18, where we were building a decentralized exchange. we initially tried developing directly on a testnet, but it was painful. the latency was inconsistent, and shared resources meant random unexpected slowdowns. debugging smart contracts became a nightmare. it's like trying to fix a car on a crowded highway at rush hour – impossible. so, we had to create our own local, private evm network for faster, more reliable development. it was a game changer, to say the least.

when you create a new node, think of it as setting up a new computer that understands the ethereum protocol. this computer can participate in the ethereum network by synchronizing with other nodes, or it can operate independently. it basically runs the same rules as other evm nodes.

one of the main reasons people create new evm nodes is for **development purposes**. having a local evm node lets you deploy and test smart contracts in a controlled environment without spending real gas money or affecting the main network. this is invaluable. you can iterate quickly, make mistakes, and debug without worrying about consequences. it's all self-contained.

here is a simple example of using `geth` a popular evm implementation in a bash command to start a local private network:

```bash
geth --datadir ./my-private-net --networkid 1337 --mine --miner.threads=1 --http --http.api "eth,web3,net,personal" --http.corsdomain "*" --http.vhosts "*" --http.addr 0.0.0.0 --http.port 8545 --allow-insecure-unlock --unlock 0 --password ./password.txt --verbosity 3
```

this command creates a local blockchain using `geth`. the `--networkid` option specifies a network id, here it's `1337` which signifies a private network. we enable mining to create blocks and `--http` exposes the node's interface which makes it available for interacting using a web3 library. you can then deploy smart contracts using something like `truffle` or `hardhat` that are frameworks for smart contract development.

another reason to create a new node is for **archival purposes**. certain entities may want to store a complete history of the blockchain for analysis or audit. this requires maintaining a full archival node, rather than relying on third-party providers. i once worked for an analytics firm that wanted to analyze all transactions, so a custom built evm node was deployed for them to do that, because using third party data proved too slow.

furthermore, **privacy and permissioned networks** are another major use case. let's say you want to build a blockchain for internal use by a company, with specific controls over who can join or access data. you don't want your transactions or data on the public chain. this requires creating a permissioned evm network, where access is limited to authorized participants only. in other words it creates a custom-tailored network that only you control and can use privately.

and you got **research purposes**. that is, researchers sometimes need to experiment with different blockchain configurations or consensus mechanisms. this often requires tweaking the client's code and creating a customized evm network. it lets you run a private network with different settings to better understand how certain aspects of the system work. i remember having to tweak a geth client for a consensus research and it was very helpful to create a new node for that.

here is a python code snippet, using `web3.py` that can help you interact with the node. this script assumes you already have a node running at `http://127.0.0.1:8545` :

```python
from web3 import Web3
import json

# connect to local node
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))

# check connection and network id
if w3.is_connected():
    print("connected to local evm node")
    print("chain id:", w3.eth.chain_id)
else:
    print("not connected to the local evm node")
    exit()

# define account addresses
account1 = w3.eth.account.from_key("0x4a739456163362ffb421c4f0c3a629a3bf3b640e36357ed1e474110771334744")
account2 = w3.eth.account.from_key("0x3d78b3af1f400b4c100189b517e97d87d6a66e2b71e298c8752609a166d65e5a")
# define sample amount
amount_wei = w3.to_wei(10, 'ether')
# send some funds
transaction_hash = w3.eth.send_transaction({'to': account2.address,'from':account1.address, 'value':amount_wei})
print(f"transaction sent with hash: {transaction_hash.hex()}")
# get balance of account 2
balance = w3.eth.get_balance(account2.address)
print(f"account {account2.address} balance: {w3.from_wei(balance, 'ether')}")

```

this script uses `web3.py` to connect to a local node, it checks its connection and then sends some funds between two accounts. this showcases how you interact with the node.

in production environments, you might set up **load-balanced nodes** for improved performance and reliability. you could have multiple nodes serving the same data, distributing the load and preventing a single point of failure. the complexity can range greatly between a simple personal node running from your computer and a production system that serves an actual application used by many people.

and let's not forget, sometimes, someone might just want to run an evm node for **the sheer nerd factor**. i mean, it’s fun to have your own personal blockchain running on your machine. it's like having your own little digital universe. i remember doing that once, just because i was bored on a saturday afternoon.

when choosing an evm implementation, you have different options like `geth`, `parity`, `erigon` and so on. each client has their own strengths and weaknesses, and the best choice really depends on your requirements and needs. i have found that `geth` and `erigon` are the most flexible for the majority of situations and use cases.

the process of creating a node is relatively straight forward depending on the chosen client implementation. most of the time, the steps usually include downloading the client, configuring the node, starting it and letting the client sync with other nodes.

another piece of code you might find useful is this simple node management using docker, using `docker-compose` it becomes very handy to manage nodes, for instance:

```yaml
version: "3.9"
services:
  geth-node:
    image: ethereum/client-go:latest
    container_name: geth-local-node
    ports:
      - "8545:8545"
      - "30303:30303"
    volumes:
      - ./geth-data:/root/.ethereum
    command:
      - "--http"
      - "--http.api=eth,web3,net,personal"
      - "--http.corsdomain=*"
      - "--http.vhosts=*"
      - "--http.addr=0.0.0.0"
      - "--http.port=8545"
      - "--allow-insecure-unlock"
      - "--mine"
      - "--miner.threads=1"
      - "--networkid=1337"
      - "--unlock=0"
      - "--password=/root/.ethereum/password.txt"

    restart: always
```

this docker configuration sets up a `geth` node using `docker`. it exposes the ports, sets up the data volume and passes the arguments to the `geth` executable.

now, if you're new to all this, you might be feeling a bit overwhelmed, i understand, we have all been there. but don't worry, start with small steps and simple use cases. create a simple private network and experiment.

for diving deeper into the topic, you can check out the official ethereum yellow paper, it is the bible of the ethereum protocol, and you should be familiar with it. and books like "mastering ethereum" by andreas antonopoulos are also great starting points.

so, in short, creating a new evm node is a versatile tool that you can use for development, privacy, archival, and research purposes, it's more than just running a program on your computer, it’s also about participating in the very foundation of web3. it’s important to understand it if you want to be proficient in this field. it is not as scary as it looks, once you do it a few times you will be doing it with eyes closed... ok not really, but you get the idea.
