---
title: "What is the relationship between Near Protocol and smart contracts?"
date: "2024-12-23"
id: "what-is-the-relationship-between-near-protocol-and-smart-contracts"
---

, let's unpack the interaction between Near Protocol and smart contracts. From my experience, particularly back in the early days of blockchain interoperability experiments where I was heavily involved in a multi-chain project, the nuances of each platform’s smart contract execution model became incredibly apparent. It's not just about the abstract notion of 'smart contracts'; it's about how each blockchain structures its environment to enable them, and how that impacts the developers using them. Near Protocol offers a unique take on this, and I'd say its approach is worth understanding in detail, especially if you're coming from platforms with different architectures.

Near, at its core, executes smart contracts using WebAssembly (wasm). This is significantly different from, say, Ethereum’s use of the Ethereum Virtual Machine (EVM) with its solidity language, which is a custom-built bytecode execution environment. The fact that Near leverages wasm is crucial because it opens up a broader range of development languages. You're not locked into a specific language; you can compile contracts written in Rust, AssemblyScript, and others, directly into wasm and deploy them. This approach has several practical advantages. Firstly, it's typically faster and more efficient compared to EVM execution. Secondly, it lowers the barrier to entry by allowing developers to use familiar languages, fostering a more diverse ecosystem.

The interaction isn't just about execution, though. It's fundamentally about how contracts are stored and managed on the Near blockchain. Contracts aren't just blobs of wasm code floating around. On Near, they’re deployed to specific accounts. Think of it as each contract having its own unique namespace where it resides. When a transaction calls a contract, the relevant code from that account is loaded into the execution environment. This allows for a clean and organized method of contract management, reducing the risk of name clashes, a common challenge when contracts are more loosely identified. The account system also supports a sophisticated access control mechanism that is critical for ensuring security when composing multiple smart contracts.

To illustrate this practically, let's consider a simple contract written in Rust, a common language for Near development due to its performance and safety features. This contract will have a single function to increment a counter:

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{env, near_bindgen};

#[near_bindgen]
#[derive(Default, BorshDeserialize, BorshSerialize)]
pub struct Counter {
    count: u64,
}

#[near_bindgen]
impl Counter {
    #[init]
    pub fn new() -> Self {
        Self { count: 0 }
    }

    pub fn increment(&mut self) {
        self.count += 1;
        env::log(format!("Current count: {}", self.count).as_bytes());
    }

    pub fn get_count(&self) -> u64 {
        self.count
    }
}
```

This snippet demonstrates a basic Near contract. The `near_bindgen` attributes are specific to the Near SDK and automate the necessary boilerplate to interface with the Near runtime. The `BorshDeserialize` and `BorshSerialize` traits enable the contract's state to be stored on the blockchain. The `increment` function increases the internal counter and logs its value. This highlights the fundamental aspects of contract interaction on Near: specific account storage, structured contract calls, and integration with the Near SDK.

Another key facet of how Near smart contracts interact is through cross-contract calls. In some other platforms, these inter-contract communications can be expensive or complicated. Near introduces a mechanism known as 'promise-based calls'. This asynchronous model for inter-contract interaction allows contracts to call other contracts and continue processing their own logic without having to wait for the called contract to fully execute. This enables parallel execution of contract calls and vastly enhances the efficiency of complex, multi-contract systems.

To exemplify this, consider an example where one contract calls another to retrieve data. Here's the first contract, a ‘DataStore’, which exposes a getter:

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{near_bindgen, PromiseResult};

#[near_bindgen]
#[derive(Default, BorshDeserialize, BorshSerialize)]
pub struct DataStore {
    data: String,
}

#[near_bindgen]
impl DataStore {
    #[init]
    pub fn new(initial_data: String) -> Self {
        Self { data: initial_data }
    }

    pub fn get_data(&self) -> String {
        self.data.clone()
    }
}
```

Here’s the calling contract, a `DataConsumer`, which gets the data from `DataStore`:

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{env, near_bindgen, Promise, PromiseResult};

#[near_bindgen]
#[derive(Default, BorshDeserialize, BorshSerialize)]
pub struct DataConsumer {
    data_retrieved: String,
}

#[near_bindgen]
impl DataConsumer {
    #[init]
    pub fn new() -> Self {
        Self {data_retrieved: String::new()}
    }

     pub fn retrieve_data(&mut self, store_account_id: String) -> Promise{
         Promise::new(store_account_id.parse().unwrap())
              .function_call(
               "get_data".to_string(),
                "".into(),
                0,
                env::prepaid_gas() - 5_000_000_000_000
            )
    }

    #[private]
    pub fn retrieve_data_callback(&mut self, #[callback] result: PromiseResult<String>) {
           match result {
                PromiseResult::Successful(value) => {
                    if let Ok(data) = near_sdk::serde_json::from_slice::<String>(&value){
                         self.data_retrieved = data;
                    }
                }
                PromiseResult::Failed => {
                     env::log(b"Failed to get data.");
                }
        }
    }

    pub fn get_retrieved_data(&self) -> String {
      self.data_retrieved.clone()
    }

}
```
In the `DataConsumer`, `retrieve_data` initiates the asynchronous call to `DataStore`, and the result is processed by the `retrieve_data_callback`. This shows how Near handles inter-contract calls; the `DataConsumer` doesn't halt execution but continues processing after the `Promise` is fulfilled.  The critical aspect here is the utilization of the callback function via the `@private` attribute and the `#[callback]` attribute, which is essential when dealing with asynchronous operations.

Finally, the way Near handles transaction fees related to smart contract execution also plays an important part. Rather than an indiscriminate, all-encompassing gas system, Near has a concept of ‘gas limits’ that are set on a per-transaction basis. This allows for a more predictable fee model, and can help avoid the gas spikes that occur in other blockchain platforms.

In terms of resources, I’d suggest looking into the official Near Protocol documentation, which is quite comprehensive and includes in-depth explanations of the gas model, contract interactions, and the WebAssembly execution environment. The "Programming NEAR" book by the Near core team would be another outstanding resource for a deeper dive, offering both theoretical and practical information for working with Near. I would also recommend reading some research papers regarding the design and architecture of other high-performance blockchains such as Solana or Algorand. Understanding the choices made in those systems will give additional context for appreciating the design of Near. Further investigation into wasm itself is also invaluable. The WebAssembly official documentation and specification are available, offering very detailed explanations of the standard.

In conclusion, the relationship between Near and smart contracts is not just about supporting their execution; it's about doing so in a way that leverages the benefits of WebAssembly, enhances the developer experience through readily accessible programming languages, provides a structured approach to account management and provides a robust mechanism for inter-contract communication. The unique promise-based asynchronous model for contract calls makes it more powerful than other similar systems in terms of speed and capability. The deliberate design choices made by the team behind Near ultimately set it apart, allowing for efficient and scalable smart contract development.
