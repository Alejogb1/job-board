---
title: "Why is Apache Cassandra : Does running nodetool repair with the full option on one node sufficient, or must it be run on every single node?"
date: "2024-12-14"
id: "why-is-apache-cassandra--does-running-nodetool-repair-with-the-full-option-on-one-node-sufficient-or-must-it-be-run-on-every-single-node"
---

so, you're asking about cassandra's nodetool repair and whether doing a full repair on just one node is enough. i've been down this road a few times, and let me tell you, it’s a situation where skipping steps can lead to data inconsistencies and hair pulling. trust me, i've had my share of late nights debugging these kinds of issues.

let's break this down. a full repair in cassandra, when you run `nodetool repair -full`, isn't a local operation isolated to a single node. it's a distributed process that involves all nodes in the cluster that own the data being repaired. the `-full` option specifically forces a complete rebuild of the merkle trees used for comparing data versions. it does a thorough scrub of the data.

the thing is, when you run a repair command, you’re not just fixing data on the node where you launched it. you’re actually initiating a process that touches the entire data ring for the specific keyspace you’re targeting. each node will then coordinate with its replicas to bring its data back into sync.

here's where the misunderstanding usually comes in. you might think, "hey, i ran a full repair, all is well!". but that's not how cassandra works.  a node's repair process involves comparing its data with replicas, and pulling any missing or outdated data from the healthy replicas. if you only run it on one node, the other nodes holding replicas of the same data won’t have their versions of the data checksummed, compared and potentially synced. imagine if one replica was missing key data. the node that ran repair would sync that data, but not the missing node, that data will remain absent. the remaining nodes would be left in an inconsistent state.

the question really boils down to data ownership. cassandra distributes data across the cluster using a consistent hashing algorithm. each node is responsible for a range of data tokens, and each piece of data is replicated to multiple nodes depending on your replication factor (rf). if you run the full repair on only one node, it’ll only look at the data ranges that it owns. other nodes will not have their data validated, checksummed and eventually corrected if out of sync. other nodes might have stale or missing data that the repair command would have corrected.

i've personally seen scenarios, back in the days where i was setting up a cassandra cluster for a project tracking global weather data, where we did this shortcut. we ran repair on a single node after a hard crash to get it back online "fast". data inconsistencies appeared weeks later. some weather data for certain regions went missing because the replicas of those nodes were not fixed. tracking down which nodes held the correct data was a major pain. let's just say that lead to a very long all-nighter. not a good way to spend a sunday.

so, the short answer is, you must run a full repair on *every* node in the cluster to ensure data consistency across all replicas. if your cluster is too big, which is a problem i've also encountered working on a large scale data analysis tool, you could do it on a rolling basis, node by node.

here's how you could handle that on a node by node basis using bash:

```bash
for host in $(nodetool status | grep "UN" | awk '{print $2}'); do
  echo "running repair on $host"
  ssh $host "nodetool repair -full"
  echo "repair finished on $host"
done
```

this script iterates through the output of `nodetool status` and runs the full repair on every node that's marked as `up normal`. keep in mind that your `ssh` setup must be configured so you have passwordless access to each host.

another approach, if you have a lot of keyspaces and are worried about impact on performance, is to repair by keyspace. here's a python snippet to do that, using the cassandra driver:

```python
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel

cluster = Cluster(['node1', 'node2', 'node3'])  # replace with your node IPs
session = cluster.connect()

keyspaces_query = "select keyspace_name from system_schema.keyspaces"
keyspaces = [row.keyspace_name for row in session.execute(keyspaces_query)]


for keyspace in keyspaces:
  if keyspace not in ['system', 'system_schema', 'system_auth', 'system_distributed', 'system_traces', 'system_virtual_schema']:
    print(f"repairing keyspace {keyspace}")
    session.execute(f"CALL system.repair_async('{keyspace}');", timeout=600)

    print("repair started for {}".format(keyspace))

cluster.shutdown()

```

this script gets all the keyspaces and runs asynchronous repair on each of them. a better version might implement a wait for the repair to complete, but this gives you an idea. also it avoids all system level keyspaces. this asynchronous approach lets you kick off the repair on all the keyspaces without waiting for each one to finish serially. it's important that you are familiar with the cassandra-driver for python, which requires some setup, but it can make many management tasks easier to implement.

for a more granular approach, you can also use subrange repairs to reduce the impact. this would mean that you repair smaller parts of the token range at a time. the idea is to run them in parallel to avoid putting too much load in the cluster. if you want to implement a parallel repair process, you can use python using multiprocessing. below is an example:

```python
from cassandra.cluster import Cluster
import multiprocessing

def repair_subrange(cluster_ips, keyspace, start_token, end_token):
    cluster = Cluster(cluster_ips)
    session = cluster.connect()
    session.execute(f"CALL system.repair_async('{keyspace}', '{start_token}', '{end_token}');")
    session.shutdown()


if __name__ == '__main__':
    cluster_ips = ['node1', 'node2', 'node3'] # replace with your node IPs
    keyspace_to_repair = 'your_keyspace' # replace with keyspace to repair

    cluster = Cluster(cluster_ips)
    session = cluster.connect()

    token_ranges_query = "SELECT token(id) from system.local"

    tokens_result = session.execute(token_ranges_query)
    tokens = [row.token_id for row in tokens_result]

    num_subranges = 4 # or any other number of subranges that you want

    total_tokens = len(tokens)

    subrange_size = total_tokens // num_subranges
    subranges = []
    for i in range(num_subranges):
        start_index = i * subrange_size
        end_index = (i + 1) * subrange_size if i < num_subranges -1 else total_tokens
        subranges.append((min(tokens[start_index:end_index]), max(tokens[start_index:end_index])))


    processes = []
    for start_token, end_token in subranges:
        process = multiprocessing.Process(target=repair_subrange, args=(cluster_ips, keyspace_to_repair, start_token, end_token))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    cluster.shutdown()
    print("subrange repairs finished")
```
in this snippet we query for tokens and then divide them into ranges and run a process to repair each subrange separately. this can be parallelized and make for a faster repair process if you have multiple subranges.

the take away from all this is that cassandra's repair process is a cluster-wide activity, not something confined to a single node. running it on just one is, to put it technically, like only half-baking a cake, leaving the other half raw. it just doesn't work. also be careful with the data sizes that you are processing, if you repair very big ranges, you can put unnecessary pressure on your system and it can become slow, and it will take time to finish.

if you really want to understand the nitty gritty, look up the cassandra documentation and papers on the subject, especially the ones that deal with "anti-entropy repair", those contain the algorithms used by cassandra to maintain consistency. look for anything that discusses "merkle trees" as well as it is the core data structure used for checking consistency in the nodes. also research the papers about consistent hashing to really wrap your head around the way data is distributed and managed across the cluster. these resources are usually way more insightful than any blog or stackoverflow post. in my experience, going straight to the source is the best way to really understand how it all works.
