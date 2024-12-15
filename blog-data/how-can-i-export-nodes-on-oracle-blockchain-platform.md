---
title: "How can I export Nodes on Oracle Blockchain Platform?"
date: "2024-12-15"
id: "how-can-i-export-nodes-on-oracle-blockchain-platform"
---

ah, exporting nodes from oracle blockchain platform, been there, done that, got the t-shirt… and probably a few sleepless nights too. it’s one of those things that sounds straightforward on paper but quickly becomes a bit of a rabbit hole once you get into the weeds. i’ve wrestled with this particular beast quite a few times, so let me share what i’ve picked up along the way.

first off, we’re not talking about some neat "export node" button in the console. oracle blockchain platform isn't exactly designed for easily moving individual nodes around like lego bricks. what you're really dealing with is more about extracting the *configuration* and *data* associated with a node, rather than the node itself as a self-contained unit. it's crucial to understand that distinction.

i remember my first time trying to migrate a test environment. i thought it was going to be a matter of clicking around and downloading a zip file. boy, was i wrong. i spent a solid three hours fumbling around with the apis before i even got the first real data point out of that thing. the docs, while comprehensive, aren’t always exactly ‘beginner friendly’ in this area. anyway, lesson learned: get comfortable with the oracle blockchain platform api and cli tools.

the core issue, from what i've gathered, usually boils down to what exactly you need to export. are we talking just the node configuration? or are we talking about the entire ledger data too? each scenario requires a different approach. let's break it down.

**node configuration export**

this usually involves gathering information about the node’s identity, its peer configuration (including endpoint urls, tls certificates, etc) and any custom chaincode deployed on it. here’s where the oracle blockchain platform cli comes into play. we can use it to get details about the network and the specific nodes. the trick is to script it so you’re not copy-pasting outputs all day. for instance:

```bash
export obp_user="your_obp_user"
export obp_password="your_obp_password"
export obp_url="your_obp_url"
export obp_instance="your_obp_instance_id"

obp --user $obp_user --password $obp_password --url $obp_url network get --instance $obp_instance --output json > network_config.json
```

this grabs all the network config including all the nodes' details and stores in `network_config.json` as json which is fairly easy to read and parse with python, for example. you will likely need to dig through this output to extract the relevant node-specific config. things like the `peerurl`, `eventurl`, and the various `tls` configurations are really key.

what we are actually doing is to get a json file with a list of all the nodes of a given instance, so you can extract your node's data from that big bunch of data.

remember, the cli is your friend here, don’t get stuck clicking around the web interface for this kind of task. it's way more efficient once you get used to it.

**ledger data export**

this is where things get trickier. oracle blockchain platform doesn’t offer a direct way to just dump the entire ledger to a file (and i doubt you'd want to even if it did!). what we usually need to do is to use the chaincode apis and write a custom chaincode to query the ledger based on the data and then we export that data as a json or a csv by doing external calls to a rest api endpoint or something else. this approach allows a controlled and structured export of the required data. let me give you an example of how to do this in a chaincode with go:

```go
package main

import (
        "fmt"
        "encoding/json"
        "github.com/hyperledger/fabric/core/chaincode/shim"
        pb "github.com/hyperledger/fabric/protos/peer"
)

type SimpleChaincode struct {
}

func (t *SimpleChaincode) Init(stub shim.ChaincodeStubInterface) pb.Response {
        return shim.Success(nil)
}

func (t *SimpleChaincode) Invoke(stub shim.ChaincodeStubInterface) pb.Response {
        function, args := stub.GetFunctionAndParameters()

        if function == "exportLedgerData" {
                return t.exportLedgerData(stub, args)
        }
        return shim.Error("Invalid invoke function name.")
}

func (t *SimpleChaincode) exportLedgerData(stub shim.ChaincodeStubInterface, args []string) pb.Response {
    if len(args) != 1 {
       return shim.Error("Incorrect number of arguments. Expecting a key prefix.")
    }
    keyPrefix := args[0]

    queryIterator, err := stub.GetStateByRange(keyPrefix, keyPrefix + "\uffff")
    if err != nil {
      return shim.Error(fmt.Sprintf("Error querying ledger range: %s", err.Error()))
    }
    defer queryIterator.Close()

    var results []map[string]interface{}
    for queryIterator.HasNext() {
            queryResponse, err := queryIterator.Next()
            if err != nil {
                return shim.Error(fmt.Sprintf("Error getting next element: %s", err.Error()))
            }

            var value map[string]interface{}
            if err := json.Unmarshal(queryResponse.Value, &value); err != nil {
                return shim.Error(fmt.Sprintf("Error unmarshalling value: %s", err.Error()))
            }

            results = append(results, value)
    }

    resultsBytes, _ := json.Marshal(results)
    return shim.Success(resultsBytes)
}

func main() {
    err := shim.Start(new(SimpleChaincode))
        if err != nil {
            fmt.Printf("error starting Simple chaincode: %s", err)
        }
}
```

this chaincode allows you to query ledger entries with a given prefix for the key. you can then query this chaincode by passing the prefix as the argument and return the data as a json. of course you would need to iterate through each prefix or use a more sophisticated approach according to your data and schema. the important part here is that you’re pulling the data *out* of the ledger. this isn’t a snapshot, but rather an extraction based on the current state.

**important caveats and gotchas**

*   **certificates:** exporting nodes often involves dealing with lots of x509 certificates. you'll need to carefully manage these, as they're fundamental to how the blockchain nodes authenticate each other. usually, you will need to extract those from the `network_config.json` we just mentioned and use the keys and certs with your new node. you'll need to know how to use `openssl` to manage the certs and convert the pem files to other formats according to what the oracle platform expects.

*   **chaincode management:** the chaincode used is part of the nodes configuration as well and needs to be deployed as the same version in any new nodes and this process might need extra care when you export the node. ensure you have your chaincode sources and deployment configurations handy. i have made the mistake a couple of times of using a wrong chaincode version and then had all kind of unexpected errors.

*   **network configuration:** exporting a node often implies importing it somewhere else, so carefully check if your new network configuration is compatible with your current exported configuration. things like network ids, channel names, etc. must be all aligned, otherwise, it will be a mess.

*   **data volume:** if you have large ledger, extracting it using the approach i showed above can take a while. i've seen some instances with millions of transactions, and extracting that data to a json was not a quick task. in these cases, consider incremental export strategies or more efficient query mechanisms within your chaincode.

**what resources to check out**

for learning the ropes, i highly recommend:

*   **oracle blockchain platform documentation:** the official docs are your first point of call. search for terms like "cli" and "api" to find the relevant parts and practice a lot with the cli commands.
*   **the hyperledger fabric documentation:** even if you are using the oracle platform, understanding the basic concepts from hyperledger fabric is key to grasp what is going on under the hood. look for the fabric peer command reference, specially when you are working with keys and certs. i spent hours reading this, and i can't tell you how helpful it is.
*   **programming hyperledger fabric by mahmoud abdel-kader:** this is a good deep dive into fabric and the concepts used in blockchain nodes if you want to understand more behind the scenes.
*   **mastering blockchain, third edition by lorne lantz:** also an amazing resource to master blockchain concepts with examples.

**final thoughts**

exporting nodes from oracle blockchain platform isn't a task for the faint of heart. it requires a good understanding of the underlying technology, the apis, and a bit of coding, plus you need to understand how to use certificate management tools, so prepare for some trial and error. treat it as a learning experience, and hopefully this response saves you some of the headaches i’ve been through. and remember, when life gives you lemons, write a chaincode to export them. that's the kind of problem solving this platform pushes you towards, or is it more like 'when life gives you a blockchain, write a chaincode for it'? i am not very good at jokes.

good luck out there!
