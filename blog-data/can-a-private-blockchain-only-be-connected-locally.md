---
title: "Can a private blockchain only be connected locally？"
date: "2024-12-14"
id: "can-a-private-blockchain-only-be-connected-locally"
---

here's my take on it:

so, the question about private blockchains and local connectivity, i've definitely been down that rabbit hole a few times. it’s not a simple yes or no, but more of an “it depends” situation with a lot of practical nuances. the core of the question is if a private blockchain is inherently limited to a local network, and the quick answer is absolutely not. private just means restricted access, not necessarily restricted *physical* location.

i remember this one time, back when i was working on a supply chain tracking project for a small startup, we were initially thinking of keeping the whole thing super local. we had this idea of running everything on a cluster of machines in our office. basically, it was a bunch of beefy servers sitting in a corner with a dedicated network switch. we had a basic ethereum private network setup. it worked alright for internal testing, but the moment we needed to share data with the external factories and distributors, things got complicated real quick. we needed to give these other entities access to the chain, but not everyone was going to physically connect to our office network. 

the key thing to grasp is that a blockchain network, private or public, relies on peer-to-peer communication between nodes. these nodes don't need to be on the same local area network (lan). they just need to be able to communicate with each other through whatever network infrastructure is available. in our case, it meant using public ip addresses for some nodes, securing their communication using certificates and firewall rules, and creating a vpn to reach the nodes on the factory floor.

let's break it down a bit more technically. a local blockchain, like the one we initially used, typically uses a simple network configuration where all nodes are on the same subnet. nodes can discover each other easily through broadcast messages and simple network protocols. you can achieve that with a configuration file like this (for a geth client, for example):

```
{
  "networkid": 1234,
  "genesis": {
    "alloc": {},
    "config": {
      "chainId": 1234,
      "homesteadBlock": 0,
      "eip150Block": 0,
      "eip155Block": 0,
      "eip158Block": 0,
      "byzantiumBlock": 0,
      "constantinopleBlock": 0,
      "petersburgBlock": 0,
      "istanbulBlock": 0,
      "muirGlacierBlock": 0,
      "berlinBlock": 0,
      "londonBlock": 0,
      "parisBlock": 0
    },
    "difficulty": "0x20000",
    "gasLimit": "0x8000000",
    "extraData": "",
    "nonce": "0x0000000000000042",
    "mixhash": "0x0000000000000000000000000000000000000000000000000000000000000000",
    "coinbase": "0x0000000000000000000000000000000000000000",
    "timestamp": "0x00",
    "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000"
  },
  "datadir": "/home/user/my-eth-node",
  "port": 30303,
  "verbosity": 3,
  "rpc": {
      "enabled": true,
      "api": ["eth","net","web3","personal","admin"],
      "port": 8545,
      "corsdomains": ["*"],
      "vhosts": ["*"]
  }
}

```

the configuration above sets up a private ethereum network locally, it can be access in the same lan network, but it lacks configuration to allow external connections.

but what if we need to expand beyond the lan? well, we need to modify the node's configuration. that's where we start looking into the network layer settings. for nodes to communicate over a wider network (say, the internet), they need to be reachable using their ip addresses. this means potentially opening ports on firewalls, setting up static ips, or using domain names. and of course, securing these connections with tls or vpn is crucial. this configuration allows the nodes to communicate with each other, it is needed to make your blockchain usable outside the lan network:

```
{
  "networkid": 1234,
  "genesis": {
    "alloc": {},
    "config": {
      "chainId": 1234,
      "homesteadBlock": 0,
      "eip150Block": 0,
      "eip155Block": 0,
      "eip158Block": 0,
      "byzantiumBlock": 0,
      "constantinopleBlock": 0,
      "petersburgBlock": 0,
      "istanbulBlock": 0,
      "muirGlacierBlock": 0,
      "berlinBlock": 0,
      "londonBlock": 0,
      "parisBlock": 0
    },
    "difficulty": "0x20000",
    "gasLimit": "0x8000000",
    "extraData": "",
    "nonce": "0x0000000000000042",
    "mixhash": "0x0000000000000000000000000000000000000000000000000000000000000000",
    "coinbase": "0x0000000000000000000000000000000000000000",
    "timestamp": "0x00",
    "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000"
  },
  "datadir": "/home/user/my-eth-node",
  "port": 30303,
  "verbosity": 3,
  "bootnodes": [
    "enode://your_node_id@your_public_ip:30303"
  ],
  "rpc": {
      "enabled": true,
      "api": ["eth","net","web3","personal","admin"],
      "port": 8545,
      "corsdomains": ["*"],
      "vhosts": ["*"]
  }
}

```

the new configuration includes the `bootnodes` array. this array is where you include one or more nodes that the new node can initially connect to. `your_node_id` is the id of the node that you want to use as a bootnode, and the `your_public_ip` is the public ip where that bootnode is running. this allows nodes outside the lan network to join the private blockchain.

the key thing here is that the "private" aspect of a private blockchain is about *permissioning*. it's about controlling *who* can participate in the network, not *where* those participants are located. you can enforce this using a variety of techniques, such as node whitelisting, permissioned smart contracts, or identity management solutions, but not by limiting the location of the nodes. you can have a completely private network with nodes spread across different continents as long as their communication is properly configured and secured. this can be done with vpn networks, or even with direct public ip access, although, if you want to have a more professional set up, you should include a load balancer in the middle for availability and to avoid single points of failure.

now, it isn't just about getting the nodes connected; you also have to consider how users/applications interact with the chain. for instance, if an application needs to interact with a smart contract, you don't want every single user connecting directly to a node. that's where things like load balancers, apis and gateways come into play. those solutions are really important for building usable applications on top of a private chain.

one more example of a node configuration. this time is a simplified example using docker. the important thing here is that you need to map the port where the node runs inside the docker to a port exposed to the host machine. this way you can create a cluster of nodes, and those nodes can be running in different machines with different ips and so on. you can connect different nodes using the `bootnodes` property that we talked about before.

```yaml
version: "3.9"

services:
  eth_node:
    image: ethereum/client-go:v1.13.13
    container_name: eth-node
    ports:
      - "30303:30303"
      - "8545:8545"
    volumes:
      - ./my_data:/root
    command:
      - "--datadir=/root"
      - "--networkid=1234"
      - "--http"
      - "--http.api=eth,net,web3,personal,admin"
      - "--http.addr=0.0.0.0"
      - "--http.port=8545"
      - "--http.corsdomain=*"
      - "--http.vhosts=*"
      - "--allow-insecure-unlock"
      - "--bootnodes=enode://your_node_id@your_public_ip:30303"
```
a docker compose like this one is very useful for building clusters of nodes quickly, and deploy nodes to any machine with docker installed. you will need to generate a genesis file as well with the previous configuration.

so, tl;dr: a private blockchain absolutely doesn't have to be confined to a local network. it’s about controlling access, not location, and you can connect nodes across various networks as long as you configure networking, security, and access controls correctly. and yeah, setting up a network can be tricky, sometimes it feels like you are trying to solve an old puzzle from the 80's, and only after hours and hours of checking configurations you find a small typo that makes everything work again.

for diving deeper into the technical nitty-gritty, i would recommend checking out books like “mastering ethereum” by andreas antonopoulos and gavin wood, or the documentation of specific blockchain implementations. the ibm blockchain documentation is also a pretty good resource. it's all out there, just gotta know where to look. also, the ethereum yellow paper is also a must read for understanding how ethereum works.
