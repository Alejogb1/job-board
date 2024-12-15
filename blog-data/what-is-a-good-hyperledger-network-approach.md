---
title: "What is a good Hyperledger network approach?"
date: "2024-12-15"
id: "what-is-a-good-hyperledger-network-approach"
---

alright, so you're asking about a good hyperledger network approach. it's a broad topic, and there isn't one single "good" way, it really depends on your specific use case, but let's break down how i've approached this in the past, what i’ve seen work well, and some gotchas to look out for.

i've spent the last decade or so knee-deep in distributed ledger tech, starting way back when everyone was still trying to figure out exactly what the heck 'blockchain' really meant. hyperledger fabric, specifically, has been a big part of that journey, and i’ve learned a few things the hard way. i've built networks for everything from supply chain tracking to digital identity management, and each time it’s a different beast.

first off, thinking about network topology is crucial. a flat network where all peers connect to each other is usually not the way to go, especially if you're aiming for any kind of scale. you quickly run into bottlenecks with gossip and transaction processing, the bigger it gets.

what i usually do, and recommend, is a tiered approach. i split the network into a few different types of peers:

*   **orderers:** these guys are the gatekeepers, they're responsible for ordering transactions into blocks. i usually deploy them in a cluster using raft consensus. it's more resilient than solo ordering, and it’s relatively straightforward to set up. you could go with kafka, but raft is generally more straightforward for most cases.
*   **endorsing peers:** these are the peers that simulate and endorse transactions based on the chaincode (smart contract) logic. this is where most of the actual business logic lives. you should have at least two endorsing peers per organization for redundancy.
*   **committing peers:** these are peers that commit the validated blocks from the orderers. these can be the same as the endorsing peers but sometimes you might want to separate them if you have a huge volume.
*   **anchoring peers:** these peers handle cross-organization discovery and gossip, these are important for multi-organization network setups and they serve as the entry point of your org in your network setup, they gossip with the anchoring peers from the other orgs to share information about the network.

here's a high-level picture of how that typically looks, without diagrams, but written out:

*   client applications send transaction proposals to endorsing peers.
*   endorsing peers execute the chaincode and send back endorsement responses.
*   the client collects enough endorsements based on the endorsement policy.
*   the client sends a transaction to the orderer service.
*   the orderer creates a block of transactions.
*   the orderer broadcasts the block to all committing peers.
*   committing peers validate and commit the block to their ledgers.

one key lesson i learned the painful way, is not to skimp on the resources of your peers. especially the orderers. undersized orderers become a performance choke point quite quickly. memory, cpu and i/o is your friend here. it's not "one size fits all", but monitoring during load testing is key to know your limits, i typically use prometheus for monitoring.

when it comes to channel management, i recommend keeping your channels specific to logical data and the business entities you work with, avoid having a single "master channel" where all the data goes. this helps with controlling access and overall network performance. for example, if you have two teams that work with different parts of a product, give them separate channels, that way you can isolate the chaincode of each team and avoid collisions.

here's a basic example of a peer config yaml snippet to illustrate the separation of the endorsement and commitment process, you can have this under `core.yaml` inside the peer's configuration folder:

```yaml
peer:
  gossip:
    useLeaderElection: true
    orgLeader: false
    membershipTrackerInterval: 5s
    maxBlockCountToStore: 100
    pullInterval: 4s
    pullPeerNum: 3
  # for endorsing peers, enable this block.
  # chaincode:
  #  startuptimeout: 300s
  #  executeTimeout: 300s
  #  installTimeout: 300s
  # for committing peers, avoid this configuration block.
  
  events:
    address: 0.0.0.0:7053
    buffersize: 100
    timeout: 30s
  # ... more peer config params
```

see that `chaincode` block in the comment section? if you want a peer to endorse chaincode transactions uncomment it if you want only to commit then avoid it.

regarding chaincode, i prefer writing them in go. it’s not everyone’s first choice, but the performance is a big benefit. i've had my share of headaches debugging javascript chaincode, especially with asynchronous calls. go just feels more predictable and performs better, also keep your chaincode logic simple and modular. don't try to cram everything into one chaincode, it makes updates and testing harder. small and focused chaincodes are easier to manage.

here's a snippet of a simple chaincode example in go that gets a value from the ledger:

```go
package main

import (
	"fmt"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	pb "github.com/hyperledger/fabric-protos-go/peer"
)


type SimpleChaincode struct {
}


func (t *SimpleChaincode) Init(stub shim.ChaincodeStubInterface) pb.Response {
	return shim.Success(nil)
}


func (t *SimpleChaincode) Invoke(stub shim.ChaincodeStubInterface) pb.Response {
	function, args := stub.GetFunctionAndParameters()
	if function == "getValue" {
        return t.getValue(stub,args)
	}
    return shim.Error("invalid function")
}


func (t *SimpleChaincode) getValue(stub shim.ChaincodeStubInterface, args []string) pb.Response {
	if len(args) != 1 {
        return shim.Error("incorrect number of arguments, expect key")
    }

    key := args[0]

    valueBytes, err := stub.GetState(key)
    if err != nil {
        return shim.Error(fmt.Sprintf("failed to get state for key %s",key))
    }

    if valueBytes == nil{
        return shim.Error(fmt.Sprintf("value not found for key %s",key))
    }
    
    return shim.Success(valueBytes)
}

func main() {
	err := shim.Start(new(SimpleChaincode))
	if err != nil {
		fmt.Printf("Error starting Simple chaincode: %s", err)
	}
}

```

this is an example of a really simple chaincode that retrieves a value for a given key using the `GetState` method of the `stub` object.

security is another area to be mindful of. remember that hyperledger fabric by default requires mutual tls authentication between all peers. a poorly configured network is an open invitation for trouble, so pay extra attention to setting up your certificate authorities and managing your crypto material. rotate them periodically. do not reuse the same root ca for multiple networks. i also encourage you to look into hardware security modules for managing your private keys.

here's a very simple example on how you can specify the location of the key material:

```yaml
  client:
    #...
    cryptoconfig:
      path: /opt/msp/
```

inside `/opt/msp/` you should have the following structure `keystore` for the private key of the current peer, and `signcerts` for the certificate of this same peer. this is used by the client to interact with the network.

for resources, i wouldn't point you to some "quick start guide" online. instead look at the official hyperledger documentation, specifically pay attention to the tutorials, especially the ones regarding the operation part. i'd also recommend looking into the "mastering hyperledger fabric" book by angus young, it provides a complete overview of fabric and all of the moving parts. another book i would recommend is the "blockchain in action" by bina ramamurthy which although not hyperledger specific gives good insights on building and designing distributed ledger applications.

finally, remember to monitor your network's performance, and test your setup in various scenarios, and always be prepared to tweak configurations and deploy changes as the network grows, because it will. there isn't a single perfect blueprint here, it's a constant iteration to adapt to your use case. building a solid hyperledger network is much like building a house, if you don't lay a strong foundation you will end up with some major headaches later.

oh, and one last thing, never ever `rm -rf` the peer's chaincode directory, it deletes it permanently from your file system... (yes, i did that once, i deserve all the shame i got for that one).

that is a summary of how i usually approach hyperledger network design and configuration. hope it helps you with your quest.
