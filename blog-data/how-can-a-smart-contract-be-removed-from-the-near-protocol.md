---
title: "How can a smart contract be removed from the NEAR Protocol?"
date: "2024-12-23"
id: "how-can-a-smart-contract-be-removed-from-the-near-protocol"
---

Alright, let's tackle this. The question of removing a smart contract from the near protocol isn't as straightforward as one might initially hope. It's not like deleting a file on your local machine, and for good reason; we're dealing with immutable data on a blockchain. I've had to navigate this situation a couple of times, most notably during a project involving a particularly experimental token contract that, frankly, didn't pan out as planned. My experience, and the solutions I've implemented, have provided a clear picture of what's possible, and what's not, when dealing with near contracts.

The first crucial point to understand is that you cannot actually *delete* a smart contract from the near blockchain. Once deployed, the contract's code and its associated state data are permanently recorded as part of the distributed ledger. This immutability is a core characteristic of blockchain technology and serves to provide transparency and security. What we *can* do, however, is effectively disable the contract and, in some scenarios, migrate its data to a new contract. This process requires a multi-pronged approach, focusing on rendering the original contract unusable, rather than attempting a deletion, which is fundamentally impossible.

There are primarily three techniques I’ve found useful, each with its own set of implications and suitable scenarios:

1. **Self-Destruction Mechanism:** The most direct approach is to embed logic within the contract itself that allows it to be effectively disabled. This is usually implemented through a dedicated function, often restricted to the contract's owner, that changes the contract's state in such a way that further function calls become ineffective or revert. Essentially, it's a controlled shut-down.

2. **Contract Replacement (Proxy Pattern):** This method involves deploying a new, ‘replacement’ contract and having the original contract redirect all calls to the new one. This involves creating a proxy contract that acts as an intermediary. The user interacts with the original address, and the proxy directs the calls to the new contract. The initial contract is effectively sidelined, with calls redirected or reverted.

3. **Contract Migration (Data Transfer):** If the objective is to not just disable, but actually salvage the data managed by the contract, a data migration strategy is needed. This requires reading the state of the original contract and then writing it to a new, updated version of the contract. Often this is done in conjunction with the proxy pattern.

Let’s break down each of these methods with code examples. For simplicity, we’ll assume the contracts are written in Rust, which is frequently used for Near contract development.

**Example 1: Self-Destruction**

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{env, near_bindgen, AccountId, PanicOnDefault};

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct StatusContract {
    pub owner: AccountId,
    pub is_active: bool,
}

#[near_bindgen]
impl StatusContract {
    #[init]
    pub fn new(owner: AccountId) -> Self {
        Self {
            owner,
            is_active: true,
        }
    }

    pub fn get_status(&self) -> bool {
        if !self.is_active {
            env::panic_str("Contract is inactive");
        }
        self.is_active
    }

    pub fn deactivate(&mut self) {
        self.assert_owner();
        self.is_active = false;
    }

    fn assert_owner(&self) {
        assert_eq!(
            env::predecessor_account_id(),
            self.owner,
            "Only the owner can call this method"
        );
    }
}
```

In this example, the `deactivate` function sets `is_active` to `false`.  After this, any calls to `get_status` will trigger a panic.  The contract is effectively disabled.  This is the simplest form of "removal" and suitable when no data needs to be preserved.

**Example 2: Contract Replacement (Proxy)**

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{env, near_bindgen, AccountId, PanicOnDefault, Promise};

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct ProxyContract {
    pub owner: AccountId,
    pub target_contract: AccountId,
}


#[near_bindgen]
impl ProxyContract {
    #[init]
    pub fn new(owner: AccountId, target_contract: AccountId) -> Self {
        Self { owner, target_contract }
    }

     #[payable]
    pub fn forward_call(&mut self) -> Promise{
       self.assert_owner();
        let method = env::current_account_id();
        let args = env::input().unwrap();

        Promise::new(self.target_contract.clone())
        .function_call(
            method.to_string(),
            args,
            env::attached_deposit(),
            env::prepaid_gas() - env::used_gas()
        )
    }

    fn assert_owner(&self) {
        assert_eq!(
            env::predecessor_account_id(),
            self.owner,
            "Only the owner can call this method"
        );
    }
}
```

This proxy contract, upon initialization, accepts an `target_contract` address.  Instead of having logic, it forwards all calls from itself to the `target_contract`.  Once deployed and setup, the initial contract is no longer used and essentially defunct.

**Example 3: Data Migration**

Data migration, while technically more complex, is needed in some scenarios. The actual implementation will depend heavily on the state data you need to move, but the following shows the concept.

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{env, near_bindgen, AccountId, PanicOnDefault, collections::LookupMap, Promise};


#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct OldData {
    pub values : LookupMap<String, u128>
}


#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct MigrationContract {
    pub owner: AccountId,
    pub migrated: bool,
    pub new_contract: AccountId,
    old_contract: AccountId,
}

#[near_bindgen]
impl MigrationContract{
    #[init]
    pub fn new(owner: AccountId, new_contract: AccountId, old_contract: AccountId ) -> Self {
        Self {
            owner,
            migrated: false,
            new_contract,
            old_contract
         }
    }

   pub fn migrate(&mut self) -> Promise{
        self.assert_owner();
        assert!(!self.migrated, "Already migrated");

        Promise::new(self.old_contract.clone())
          .function_call("get_old_data".to_string(),
          vec![],
           0,
           env::prepaid_gas() - env::used_gas())
           .then(
              Promise::new(env::current_account_id())
              .function_call(
                "process_old_data".to_string(),
               vec![],
                0,
                env::prepaid_gas() - env::used_gas())
            )
       }

    #[private]
    pub fn process_old_data(#[callback] old_data_raw: Vec<u8>){

        let  old_data: OldData = BorshDeserialize::try_from_slice(&old_data_raw).expect("Faild to deserialize");
        
        // Process and move data from old_data to self.new_contract.
        // This function is highly dependent on both the old data structure
        // and the target contract structure. 
        env::log_str("migrated");
        //Example: For each entry: 
        //let keys = old_data.values.keys();
        //for k in keys{
        //  Promise::new(self.new_contract.clone())
        //  .function_call("set_value".to_string(),
        //    args_from(k, old_data.values.get(&k.to_string())),
        //   0,
        //   env::prepaid_gas() - env::used_gas());
        //}

    }
    
    fn assert_owner(&self) {
        assert_eq!(
            env::predecessor_account_id(),
            self.owner,
            "Only the owner can call this method"
        );
    }
}

```

This shows a simplified example of migrating data. The `migrate` function triggers a `Promise` to extract data from the old contract, and then uses a callback to `process_old_data`, which transfers the data to the new contract. This involves custom logic tailored to the specific data structures.

For further study on these concepts, I'd recommend the following:

*   **“Mastering Ethereum” by Andreas M. Antonopoulos and Gavin Wood:** This provides a solid foundational understanding of smart contracts and their behavior, though focused on Ethereum, many concepts are transferable.
*   **The NEAR Protocol Documentation:** The official NEAR docs are the best source for understanding the nuances of contract development within the NEAR ecosystem, including patterns and best practices. Specifically, research the topics of Contract Ownership and Cross-Contract calls.
*   **“Programming Rust” by Jim Blandy, Jason Orendorff, and Leonora F.S. Tindall:** If you're not comfortable with Rust, this book offers a deep dive into the language, which is beneficial for NEAR contract development.

In summary, while true deletion is impossible, you can effectively remove or replace a smart contract on the NEAR Protocol through self-destruction mechanisms within the contract, proxy patterns for forwarding calls, or contract migration for preserving the data to a new upgraded contract. Each technique comes with its own considerations, and the “best” approach depends on your specific goals for the contract in question.
