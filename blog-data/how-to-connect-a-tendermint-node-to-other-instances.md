---
title: "How to connect a Tendermint node to other instances?"
date: "2024-12-14"
id: "how-to-connect-a-tendermint-node-to-other-instances"
---

alright, so you're looking to get your tendermint nodes talking to each other. i've been down that rabbit hole a few times, and it can be a bit tricky, especially when you’re first getting started. let's break down how to get those tendermint instances playing nice.

first off, tendermint relies on a gossip protocol for peer discovery. this means nodes essentially broadcast their presence and learn about other nodes on the network. it’s not a centralized thing. think of it like a bunch of people shouting in a room, each hearing the others. to make this work, each node needs a few key pieces of information: the ip address and the listening port, and the node id.

the most important thing is the persistent_peers configuration. this is where you tell the node explicitly who to connect to. it’s a list of ‘node_id@ip_address:port’ strings. if you dont do this the node tries to get peers by itself but it may not be what you expect if they are isolated. i've seen a lot of setups that go south because of this missing setting. once, i spent an entire afternoon scratching my head because a new node refused to connect, only to realize i'd forgotten to update the persistent_peers list on the others. rookie mistake, i'll grant, but we all make 'em.

lets start with some code snippets. you'll find these invaluable in getting your setup going. we need to see the actual configurations, no hand-waving allowed.

```toml
# config.toml for node 1
###################################################
#     P2P Configuration Options                  #
###################################################
[p2p]
  laddr = "tcp://0.0.0.0:26656"
  seeds = ""
  persistent_peers = "node_id_2@192.168.1.101:26656,node_id_3@192.168.1.102:26656"
  private_peer_ids = ""
  unconditional_peer_ids = ""
  upnp = false
  pex = true
  seed_mode = false
  flush_throttle_ms = 100
  max_packet_msg_payload_size = 1024
```

```toml
# config.toml for node 2
###################################################
#     P2P Configuration Options                  #
###################################################
[p2p]
  laddr = "tcp://0.0.0.0:26656"
  seeds = ""
  persistent_peers = "node_id_1@192.168.1.100:26656,node_id_3@192.168.1.102:26656"
  private_peer_ids = ""
  unconditional_peer_ids = ""
  upnp = false
  pex = true
  seed_mode = false
  flush_throttle_ms = 100
  max_packet_msg_payload_size = 1024
```

```toml
# config.toml for node 3
###################################################
#     P2P Configuration Options                  #
###################################################
[p2p]
  laddr = "tcp://0.0.0.0:26656"
  seeds = ""
  persistent_peers = "node_id_1@192.168.1.100:26656,node_id_2@192.168.1.101:26656"
  private_peer_ids = ""
  unconditional_peer_ids = ""
  upnp = false
  pex = true
  seed_mode = false
  flush_throttle_ms = 100
  max_packet_msg_payload_size = 1024
```

replace `192.168.1.100`, `192.168.1.101` and `192.168.1.102` with the actual ip addresses of your machines, naturally. the port `26656` is the standard tendermint p2p port. notice how each node is configured to point to the other two, creating a small mesh network. if these were virtual machines you will have to make sure the network interface is bridged so they can connect directly to each other.

now, how do we find the node_id? that's tucked away inside the node’s configuration directory, usually named `config/node_key.json`.

the node key file looks like this:

```json
{
  "priv_key": "long_private_key_here",
  "pub_key": "long_public_key_here",
  "address": "node_id_here"
}
```
take the `address` field. this is the node id you need for the `persistent_peers` configuration. copy the `address` value for each node and add to the persistent_peers configuration. these are really long hexadecimal strings. make absolutely sure you have the correct `node_id` for each node. there's no magic in this, just careful copy and paste and double check. i've spent more than my share of time debugging connection issues only to find out i copied the wrong node id. it's the sort of error that makes you question all of your life choices.

another setting to watch is `laddr` in the `[p2p]` section of your `config.toml`. it specifies which address and port the node listens on for incoming connections. `tcp://0.0.0.0:26656` means that the node will accept connections on port 26656 from any network interface. you might want to restrict that based on your network configuration. for a local network, `tcp://<ip_address>:26656` is normally enough to ensure the node will accept connections only from that specific address. when running inside a docker container you will need to use the internal ip of the container not `0.0.0.0`, so something like `tcp://172.17.0.2:26656`.

there are other ways to get nodes to connect. if your nodes are in the same network, you can use a seed node. a seed node acts as a rendezvous point for other nodes. they don’t need to have a full list of every single node in the network, just the seed's address. you configure `seeds` variable in `config.toml` in the same way as `persistent_peers` but this is for bootstrapping. its useful when you are dealing with hundreds of nodes but not in the case of the question here.

there is also something called 'pex' or peer exchange which is also enabled by default. it means when you peer to a node it will give you the list of other nodes it knows about. you can also try this to bootstrap your network but its not as stable as using persistent peers. usually when i want more stability i set both persistent peers and pex.

if you do not specify any peer configuration the node will try to connect to public seed nodes or random peers by itself and you probably dont want this on a private network. it may not connect to other nodes in your local setup if they are not on the same internet facing network.

remember, tendermint’s gossip protocol isn’t instant. it might take a few seconds or a minute for nodes to fully establish connections after you start them. so, don’t get anxious immediately if you don’t see connections appear at once.

a quick note: firewalls are usually the culprit when things go pear-shaped. make sure the port (default 26656) is open on your firewall for both tcp and udp. i have been on the other side of this one and its a real head-scratcher until the aha moment. we had a new junior dev who had their laptop firewall turned on and it was blocking communication for everyone on the private test network, it took 2 hours and a lot of debugging to realise the firewall was the issue (he has learned his lesson).

when it comes to debugging the connection, check the logs. tendermint provides very informative logging. you're going to want to grep for "peer" or "connection" to see what’s happening on your node. also, check the `INFO` or `DEBUG` logs and you may see messages like `New peer connected`, or `dialing peer` and also log the peer id and network address to see what is going on. in general, you have to be in the correct directory before you run the node, otherwise it will generate new files in the current directory. the default log folder is in the same directory where the `tendermint` command is run.

as for resources, i can’t recommend enough the official tendermint documentation (look for the p2p section, as it contains pretty much all of the necessary information). also, i'd suggest reading the paper "the flood dissemination protocol" which is a seminal paper in this domain. if you can get your hands on a copy of “understanding distributed systems” by roberto vitillo, it explains all the fundamentals, which, in my experience, helps you get past the easy gotchas of this sort of networking. if you really want to understand blockchain and networking deeply i also suggest reading "mastering bitcoin" by andreas antonopoulos, although it is not tendermint specific it helps you understand all the concepts that are behind tendermint as well.

the core message here is configuration is key. get your node ids, ips, ports, right and remember the firewalls. with some careful setup and attention to detail, you should have your tendermint instances connected in no time. if they still dont connect make sure they are on the same subnet. if you are using cloud providers, the way you are networking them is very dependent on that provider. each one does its networking differently. it’s a process, it gets easier with time and experience. remember, coding is like using chopsticks, the first time its difficult and then becomes second nature. just dont try to make sushi while on a moving train.
