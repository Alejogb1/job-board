---
title: "What is the unknown task type 'ETHTX' in a Chainlink node operator job creation?"
date: "2024-12-23"
id: "what-is-the-unknown-task-type-ethtx-in-a-chainlink-node-operator-job-creation"
---

Alright, let's dissect this "ETHTX" task type you've stumbled upon in Chainlink node operator job configurations. It’s not something you'll see documented directly in the highest-level Chainlink resources, and I’ve definitely seen developers scratch their heads over it, sometimes for good reason. I recall one particularly intense week on a project last year where we were optimizing our oracle network’s performance and suddenly encountered a very similar challenge – unraveling some low-level task types.

At its core, the "ETHTX" task type within a Chainlink job definition is essentially about generating and submitting Ethereum transactions. It's a fundamental building block, though often hidden behind the more abstract interfaces like `RunLog` or `Bridge`. While those tasks manage higher-level logic and external communication, `ETHTX` puts us directly in control of transaction construction. Think of it as accessing a lower level API. In my experience, it's used when other, more pre-built Chainlink tasks don’t quite fit the requirements of a specialized interaction with the Ethereum blockchain. It's particularly handy for scenarios that need very fine-grained control over the transaction details.

Now, why would a job need to generate Ethereum transactions directly? Several reasons crop up: perhaps you need to send data to a smart contract, issue contract calls, or even directly move ETH between addresses without relying on a data feed and its associated off-chain consensus. It’s crucial when dealing with complex multi-step smart contracts that have conditional logic or require gas optimization or specific nonce handling.

Here’s a breakdown of how it generally works. The job definition will include key parameters within the `ETHTX` task, notably the `to` address (the destination contract or address), the `data` (the encoded function call, if any, including function selector and arguments), and the `value` (the amount of ETH to send). Additionally, it might specify details like the gas limit, nonce, and the from address - although Chainlink generally handles the `from` and nonce automatically unless you specifically need to configure those parameters.

What’s most critical to understand is that the `ETHTX` task does *not* automatically fetch data from an external source. It only executes the transaction that you’ve meticulously defined within your job specifications. This is important - if you need to extract data and *then* form a transaction, you’ll usually combine `ETHTX` with other Chainlink tasks such as `httpGet`, `jsonparse`, or a custom adapter.

Let's move onto some illustrative code snippets to highlight this. These examples use the Chainlink job DSL format:

**Example 1: Simple Transfer of ETH**

This snippet demonstrates sending a minimal amount of ETH to an address. It is a trivial, but illustrative example. This may be useful for on-chain state transitions that need to be triggered via ETH.

```json
{
  "initiators": [
    {
      "type": "cron",
      "schedule": "0 0 0 * * *"
    }
  ],
  "tasks": [
    {
      "type": "ethtx",
      "params": {
        "to": "0xabcdef1234567890abcdef1234567890abcdef12",
        "value": "1000000000000000"
      }
    }
  ]
}
```

In this example, the `cron` initiator dictates that the job runs at midnight every day (UTC). The `ethtx` task then constructs a transaction to send 0.001 ETH (10^15 wei) to the specified recipient address. Important: I am deliberately keeping it simple and avoiding a lot of error handling here to focus on the `ETHTX` task. In a real-world setting, you’d add additional parameters and tasks to handle failures and ensure data integrity. Also, notice we're sending wei, not ETH.

**Example 2: Calling a Smart Contract Function**

This next example illustrates a scenario where you need to interact with a smart contract by calling a specific function. Let's assume there is a contract at address `0x1234567890abcdef1234567890abcdef12345678` with a function `incrementCounter(uint256 value)`.

```json
{
  "initiators": [
    {
      "type": "runlog",
      "params": {
        "address": "0xfeedfeedfeedfeedfeedfeedfeedfeedfeedfeed"
      }
    }
  ],
  "tasks": [
      {
        "type": "jsonparse",
        "params": {
          "path": ["value"]
        }
      },
      {
          "type": "ethabiencode",
          "params": {
            "abi": "function incrementCounter(uint256 value)",
            "data": {
                "value": "$(jsonparse)"
            }
          }
       },
    {
      "type": "ethtx",
      "params": {
        "to": "0x1234567890abcdef1234567890abcdef12345678",
          "data": "$(ethabiencode)"
      }
    }
  ]
}
```

Here, we now introduce two more tasks.  The `runlog` task triggers the job when a log event occurs on the contract at address `0xfeedfeedfeedfeedfeedfeedfeedfeedfeedfeed`. It extracts the data from a `json` object that is present in the log event, which is specified via the `path` parameter of the `jsonparse` task. Then, we utilize the `ethabiencode` task to create a properly encoded function call to `incrementCounter` on our contract using that extracted data and the specified function signature. The final `ethtx` task submits this encoded function call to the specified smart contract.

**Example 3:  Adding Custom Gas Limit**

Often, you will need to adjust gas limits, especially if dealing with complex smart contract function calls.

```json
{
  "initiators": [
      {
      "type": "cron",
      "schedule": "0 0 0 * * *"
    }
  ],
  "tasks": [
      {
        "type": "ethabiencode",
        "params": {
            "abi": "function setMaxVal(uint256 value)",
             "data": {
                "value": 500
             }
        }
      },
    {
      "type": "ethtx",
      "params": {
        "to": "0x9876543210fedcba9876543210fedcba98765432",
          "data": "$(ethabiencode)",
          "gasLimit": 500000
      }
    }
  ]
}
```

In this snippet, I've added a `gasLimit` parameter, which explicitly sets the gas limit of the transaction to `500000`. The `ethabiencode` task here encodes a call to a function that takes a single uint256 argument. The gasLimit parameter is critical if you want to avoid failed transactions due to the default gas limits assigned by the node.

Now, for some recommended further reading: for a deep dive into Ethereum transaction structures and encoding, I recommend "Mastering Ethereum" by Andreas Antonopoulos, and Gavin Wood. This should clarify a lot of the lower-level workings that are essential to understanding the `ETHTX` task. Another useful book is "Programming Ethereum" by Lucena et al.

You'll find the Chainlink official documentation a good starting point, of course, but as you've observed, it might not directly detail every nuanced task type, especially those used at lower-levels such as `ETHTX`. You’ll need to use it in conjunction with reading the source code. Look particularly into their job spec design documentation on Github, if it is publicly available. Often, exploring the codebase is necessary to understand certain features that are not highlighted in official resources.

In conclusion, the "ETHTX" task provides a crucial way to directly interact with the Ethereum blockchain, enabling the construction and submission of raw transactions. It's often paired with other tasks to create complex workflows. Understanding the specifics of your project's requirements will be critical for deciding if this task type is necessary. Knowing this will ensure you can use this task effectively when other, higher-level interfaces do not suffice. Hopefully, this clarifies things for you.
