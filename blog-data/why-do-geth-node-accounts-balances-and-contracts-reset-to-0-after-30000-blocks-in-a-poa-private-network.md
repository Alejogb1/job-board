---
title: "Why do Geth node accounts balances and contracts reset to 0 after 30,000 Blocks in a POA private network?"
date: "2024-12-15"
id: "why-do-geth-node-accounts-balances-and-contracts-reset-to-0-after-30000-blocks-in-a-poa-private-network"
---

alright, so you're seeing account balances and contract storage seemingly zero out after roughly 30,000 blocks on your geth-based proof-of-authority (poa) private network. yeah, i've totally been there, it's a pain, and feels like you've stumbled into some weird edge case. this is not a bug, its more of a misunderstood configuration option.

i recall this happening back when i was first spinning up a local dev environment for some dapp prototyping, and it took me a while to isolate the issue. i thought i had broken something during a node restart or deployment script. at first it looked like my code was just completely broken, but after days of code review, logs analysis and even recreating the environment from scratch the problem continued. the data was there for a period and then *poof* everything reset.

what's happening is that geth, by default and when configured for poa networks uses a feature called "periodic snapshotting" or sometimes referred as "epoch transition". in poa systems there are no miners constantly solving puzzle to add blocks, so the blocks are created by the authorities and consensus is achieved by signature verification of those blocks by the authorities. so, blocks are periodically created. if the snapshotting isn't handled properly you end up with a reset of the state.

the whole idea behind snapshotting is to periodically prune the state trie, a persistent data structure, that records the current state, so to speak the balances and storage at any given point in time. the trie is important for data integrity, however, it's a mutable state, and without pruning it can grow very large, very quickly. the snapshotting helps manage this. instead of storing every single change to the state, geth creates these snapshots at regular intervals (the 'epoch' in poa terms). this interval is what’s causing your 'reset' at 30,000 blocks. the idea is to keep the state manageable and prevent a huge state trie that makes everything slow. in poa networks this parameter is not tied to anything related to difficulty. it's defined by your chain configuration.

the default value for `epoch` for a poa network is 30,000 blocks, meaning that every 30,000 blocks geth will create a snapshot. and if not configured, it will remove the old state before the new snapshot. this is why your data is reset. if you want to avoid the reset you have to explicitly configure the *full sync*.

to fix this issue, you have a couple of ways forward. the first way, is configuring the snapshotting to retain the previous state. the second is to use the full sync configuration.

*   **adjusting the epoch configuration:**

    if you want to keep the snapshotting behavior (that is the prune the trie), but just want to keep your data between snapshots, you will need to configure your genesis file appropriately. this is how you can do this.

    here is a sample `genesis.json` file that has the `config` and the `consensus` set to poa, and the `epoch` is defined.

    ```json
    {
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
            "berlinBlock": 0,
            "londonBlock": 0,
            "shanghaiBlock": 0,
            "cancunBlock": 0,
            "clique": {
                "period": 15,
                "epoch": 30000
            }
        },
        "consensus": "clique",
        "difficulty": "1",
        "gasLimit": "8000000",
        "alloc": {
            "0x...": {
                "balance": "1000000000000000000000"
            },
        },
        "extraData": "0x0000000000000000000000000000000000000000000000000000000000000000abcdef...12345678...",
        "mixHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "nonce": "0x0000000000000000",
        "coinbase": "0x0000000000000000000000000000000000000000",
        "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000"
    }
    ```

    the key here is this:
    ```json
    "clique": {
        "period": 15,
        "epoch": 30000
     }
    ```
    the `epoch` parameter here determines when a snapshot occurs. if you dont include this, the default value is used (30000). in your case, you need to adjust this if you want to change when these snapshots happen.

    remember that in poa there is no *difficulty* so it's a static value, also the nonce is static.

*   **full sync:**

    if you want to have the full historical data you will need to start the geth node with the `--syncmode full` flag (you might also need to resync from zero). this will make geth synchronize and keep all the history of state changes and not discard anything.

    this can be done when starting your node as a parameter flag:
    ```bash
    geth --datadir ./your_datadir --networkid 1234 --syncmode full  --http --http.api web3,eth,net,personal,debug --allow-insecure-unlock  --http.addr "0.0.0.0" --http.vhosts="*" --http.corsdomain="*"  --port 30303 --mine --miner.threads=1  --unlock "0x..." --password "password.txt" --verbosity 4 --authrpc.vhosts="*" --authrpc.port 8545 --authrpc.addr "0.0.0.0"
    ```
    or using command line flags:

    ```bash
    geth --datadir ./your_datadir --networkid 1234 --syncmode full  --http --http.api web3,eth,net,personal,debug --allow-insecure-unlock  --http.addr "0.0.0.0" --http.vhosts="*" --http.corsdomain="*"  --port 30303 --mine --miner.threads=1  --unlock "0x..." --password "password.txt" --verbosity 4 --authrpc.vhosts="*" --authrpc.port 8545 --authrpc.addr "0.0.0.0"
    ```

    notice the flag `--syncmode full`. that ensures you keep all history without discard anything.

    also notice the port options, those are for local access and for personal access on the `0.0.0.0` address.

    if you want to do the opposite use `light` instead of `full`. you can do `fast`, but it will eventually become `full`.

*   **resync from zero:**

    if you changed the sync mode for a running node, most likely you will need to *resync from zero*. this can be achieved deleting the datadir folder and starting again from the beginning. that process is going to take time, so be aware.

    if you created a folder using `--datadir` parameter, you have to delete the content of that folder, and restart the node as explained above.
    be aware that you may have to delete the chaindata files or the whole directory as the cache mechanism may be broken with this kind of sync mode change.

    i have to tell you, that when i first dealt with this problem, i spent like a full weekend and i though i went crazy. i remember spending so much time looking at the code trying to understand what i broke. and it turned out i just needed a configuration change. lesson learned. always look at the configuration before trying to debug code that isn't broken.

    just to give you an idea, during this problem time, it was before the merge, and we used `ethash` to generate blocks for testnets (but we used poa) and the issue was the same. it was a little bit funny that poa network was behaving like that, but it was just because of the state snapshotting.

    a good place to start to dive deeper into how these things work internally is to read up on the yellow paper and other resources that delve into the ethereum virtual machine (evm) and trie structures, check the book “Mastering Ethereum” by Andreas Antonopoulos and Gavin Wood for an extensive technical view. also, the eth docs itself has a full explanation of how these configurations are handled. you can find more information about consensus algorithms also in “ethereum: principles, technologies, applications” by shroff and gaur. those are some places to start. the geth docs also is a good place for command line flags and configurations.

    remember that the trie is the fundamental persistent data structure that maintains the current state of the ethereum network, so a proper understanding is fundamental to be able to debug such problems.

    anyway, hopefully this clears things up for you and you can get your private network running smoothly.
