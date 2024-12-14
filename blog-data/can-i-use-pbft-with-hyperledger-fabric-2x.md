---
title: "Can I use pbft with Hyperledger Fabric 2.x?"
date: "2024-12-14"
id: "can-i-use-pbft-with-hyperledger-fabric-2x"
---

no, you can't directly use pbft with hyperledger fabric 2.x. fabric's architecture is designed around a different consensus model. let me explain what's going on under the hood.

fabric, as you’re probably aware, utilizes a pluggable consensus architecture. this means you can choose from different ordering services, the core component responsible for ordering transactions into blocks. by default, it ships with raft, a crash fault-tolerant (cft) ordering service. raft is efficient and well-suited for permissioned blockchain networks like fabric. pbft, practical byzantine fault tolerance, is a completely different beast. it’s designed to handle byzantine faults where nodes can act maliciously. raft, on the other hand, assumes that most nodes will behave correctly, or at least fail benignly.

i've personally spent countless hours debugging consensus-related issues on different platforms, and the distinction between cft and bft protocols is something that has always been in my mind. early in my career, i was tasked to develop a toy distributed ledger from scratch. thinking i could just plug in pbft because "it's stronger", proved very very wrong. i ended up with a horribly slow and incredibly complicated mess. the performance hit was astronomical! the point is that these things are not interchangeable.

now, fabric's ordering service is the component where this decision is baked in. it's not just a matter of swapping out some configuration parameters. raft is implemented at the ordering service level. pbft, if it were to be implemented, would also require major changes at that level. the entire architecture and message flow patterns are different.

let’s think about this conceptually, imagine you’ve built a car with a specific type of engine (raft) and someone asks “can i just swap the engine with a completely different type of engine(pbft) and expect it to work?”. no right? the chassis, the transmission, the whole system is designed around that engine type.

it's a bit like trying to fit a square peg into a round hole. the pbft algorithm requires different messaging and validation mechanisms that fabric’s current ordering service isn’t designed to handle directly.

that said, don't get me wrong, the need for byzantine fault tolerance is a valid concern, especially in scenarios where you might not fully trust all the participating organizations. the way to approach this is to acknowledge that fabric at its core does not have a native pbft, but we can explore other means.

a theoretical solution would involve crafting a custom ordering service plugin for fabric that implements the pbft protocol. this would be a substantial undertaking, requiring deep understanding of fabric’s internals and the subtleties of pbft. honestly, it’s not something i would recommend doing unless you're tackling a very specific, high-security use case. plus, the performance implications of bft are something to seriously consider given the communication overhead.

while a fully functional pluggable pbft for fabric is not readily available, there are some things that can help mitigate potential malicious behaviours, like:

*   access control: fabric's identity and access management system (msp) can be used to restrict access to the network to only trusted participants. this limits the potential attack surface.
*   endorsement policies: endorsement policies define which organizations need to approve a transaction before it's committed to the ledger. this adds layers of security.
*   channel architecture: the concept of channels in fabric allows you to compartmentalize network activity, restricting the impact of potential attacks to specific channel(s). this can reduce the blast radius of a security incident.
*   hardware security modules (hsms): these can be integrated to protect cryptographic keys, which are important to the operation of the fabric network.

these can help, but they are not a complete replacement of a real byzantine fault tolerant protocol.

now, regarding resources. for a solid understanding of pbft, i recommend reading the original paper “practical byzantine fault tolerance” by miguel castro and barbara liskov. it's a dense read, but it provides the foundational knowledge needed. the 'distributed systems' book from tanenbaum is also an excellent general resource about distributed consensus protocols in general.

if you want to explore the intricacies of hyperledger fabric's consensus model, the official hyperledger fabric documentation is the best starting point. pay particular attention to sections related to the ordering service and raft. it will help you understand what is and what is not possible.

also, i've seen discussions and attempts to integrate pbft-like mechanisms into fabric, such as exploring the potential for a "bft-lite" ordering service. however, these are typically complex proofs of concepts and not ready for production. the open source community is always exploring ways to improve fabric, it's worth keeping an eye on the latest development in the fabric code repositories, and community mailing lists.

here are some code snippets to show the configuration of orderer and how it works with raft. note that this is just an example of raft, not how pbft would work:

```yaml
# example of raft configuration in fabric orderer
orderer:
  ordererType: raft
  raft:
    consensus:
      snapshottype: file
      options:
        tick-interval: 500ms
        election-tick: 10
        heartbeat-tick: 1
        max-inflight-blocks: 5
        snapshot-interval: 1000
        
    
```

and an example of a network configuration to configure orderers:

```yaml
    orderers:
        - url: orderer0.example.com:7050
          grpcOptions:
             ssl-target-name-override: orderer0.example.com
             grpc.keepalive_time_ms: 120000
          tlsCACerts:
                path: <path to orderer0 cert>
        - url: orderer1.example.com:7050
          grpcOptions:
             ssl-target-name-override: orderer1.example.com
             grpc.keepalive_time_ms: 120000
          tlsCACerts:
               path: <path to orderer1 cert>
```

and here is an example of a raft channel configuration:

```yaml
capabilities:
    channel: &ChannelCapabilities
      V1_3: true
    orderer: &OrdererCapabilities
      V1_3: true
      
application: &ApplicationCapabilities
      V1_3: true
      V1_2: false

orderer:
    ordererType: raft
    addresses:
        - orderer0.example.com:7050
        - orderer1.example.com:7050
    options:
        batchsize: 1000
    organization: OrdererOrg
    capabilities: *OrdererCapabilities
    raft:
      options:
          tick-interval: 500ms
          election-tick: 10
          heartbeat-tick: 1
          max-inflight-blocks: 5
          snapshot-interval: 1000

```

these snippets illustrate how raft parameters are configured within fabric's configuration files. this further highlights that consensus algorithms like raft are deeply integrated with the fabric architecture, so swapping them with a completely different approach like pbft is not straightforward. the complexity involved in implementing something completely different like pbft can be significant.

in short, no, you can't just plug in pbft and call it a day. if it was that easy i think i would be writing this from my yacht. you would need to write it from scratch. it's a hard thing, but not impossible. just don't start without being aware of the difficulties.
