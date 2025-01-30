---
title: "Can an Oracle Node be implemented as a NEAR Protocol contract in Rust?"
date: "2025-01-30"
id: "can-an-oracle-node-be-implemented-as-a"
---
The inherent challenge with integrating external data into a deterministic blockchain like NEAR lies in maintaining consensus. A smart contract, by design, must produce the same output given the same input across all validator nodes. External data, however, is not static and can vary. Consequently, a direct call from a NEAR contract to an external API is impossible; hence the need for an Oracle. The Oracle Node itself, which fetches this external data, cannot exist as a *direct* part of the contract execution environment. Instead, we must model a trust framework between an external agent and the contract. We achieve this by leveraging a combination of events and specific contract methods.

In my previous work developing decentralized finance (DeFi) protocols on NEAR, I consistently encountered this requirement for off-chain data integration. Building a fully trustless, decentralized oracle is complex, often exceeding the development scope for specific application contracts. Therefore, I typically structure oracle systems using an off-chain Oracle Node that publishes data on-chain, with the contract acting as a data consumer. The NEAR contract then verifies the published data against parameters set by the contract's owner or, in more advanced scenarios, by an on-chain governance mechanism.

The implementation hinges upon several key aspects of NEAR’s smart contract architecture. Primarily, we need a method to store the off-chain data, a mechanism for an external node to submit the data, and a means to authenticate this submission. Storing the data is straightforward: the contract will use persistent state variables. The submission, however, must be carefully managed to prevent malicious actors from arbitrarily altering the values. Thus, we will use a `set_data` method requiring pre-approved access. This will be secured using the `env::predecessor_account_id()`, which can verify whether the caller is the pre-designated oracle node account.

Here's the breakdown:

1.  **State Variables:** The contract will store the external data as a state variable. This variable will hold the data's most recent value and potentially a timestamp of when it was last updated.
2.  **Data Submission Method:** The `set_data` method will be the core of data ingestion. It takes the new data and a timestamp as parameters, verifies that the sender is the approved Oracle Node, and updates the state variables.
3.  **Oracle Node Authentication:** The contract will have a specific variable containing the account ID of the permitted Oracle Node. This provides authorization for the `set_data` method.
4.  **Access Control:** Only the owner of the contract can modify the allowed Oracle Node account ID. This ensures control over which external agent is authorized to submit data.
5.  **Data Retrieval:** Methods will be included to return the stored data value and the last updated timestamp. This allows other contract functionalities to operate using the external information.

Let's examine the code snippets:

**Example 1: Contract Storage and Initialization**

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::near_bindgen;
use near_sdk::env;
use near_sdk::AccountId;

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize)]
pub struct OracleContract {
    pub oracle_account: AccountId,
    pub data_value: u64, // Example with a numerical data type
    pub last_updated: u64, // Timestamp
    owner_account: AccountId,
}

impl Default for OracleContract {
    fn default() -> Self {
        panic!("Contract is not initialized")
    }
}


#[near_bindgen]
impl OracleContract {

  #[init]
    pub fn new(oracle_account: AccountId) -> Self {
        assert!(env::state_read::<Self>().is_none(), "Already initialized");
        Self {
            oracle_account,
            data_value: 0,
            last_updated: 0,
            owner_account: env::predecessor_account_id(),

        }
    }

     /// only owner can update the oracle account id
  pub fn change_oracle_account(&mut self, new_oracle_account: AccountId) {
      self.assert_owner();
        self.oracle_account = new_oracle_account;
    }
    fn assert_owner(&self) {
       assert_eq!(
        env::predecessor_account_id(),
        self.owner_account,
            "Only owner can call this method"
        );
    }
}
```

This example establishes the contract’s core state. The `oracle_account` stores the ID of the authorized Oracle Node, `data_value` will hold the actual external data, `last_updated` holds the update time, and `owner_account` the deployer of the contract which can alter the `oracle_account` setting. A default implementation is explicitly avoided to enforce proper initialization. The `new` function initializes these variables. Additionally, we have an implementation for changing the oracle account, only the contract owner is authorized to make such change.

**Example 2: Data Submission and Verification**

```rust
#[near_bindgen]
impl OracleContract {
    #[payable]
    pub fn set_data(&mut self, new_value: u64, timestamp: u64) {
        self.assert_oracle(); //Verify caller
        self.data_value = new_value;
        self.last_updated = timestamp;
    }

    fn assert_oracle(&self) {
        assert_eq!(
            env::predecessor_account_id(),
            self.oracle_account,
            "Only oracle can call this method"
        );
    }

    // Getters
    pub fn get_data_value(&self) -> u64 {
      self.data_value
    }
    pub fn get_last_updated(&self) -> u64 {
       self.last_updated
    }
}
```

This section focuses on the core interaction between the Oracle Node and the smart contract. The `set_data` method receives the new data and a timestamp. Crucially, it employs the `assert_oracle` function to ensure that the caller’s ID matches the stored `oracle_account`. This prevents unauthorized access. `get_data_value` and `get_last_updated` are methods to retrieve the oracle data after it has been set.

**Example 3: Usage Example**

Consider a scenario where a decentralized exchange (DEX) requires the price of NEAR to determine the exchange rate for other assets.

1.  The owner deploys the `OracleContract`, specifying an approved oracle node account during deployment using `new` method.
2.  The owner, or a governance mechanism if implemented, can at any point use `change_oracle_account` method to update the trusted oracle.
3.  An off-chain Oracle Node will retrieve the current NEAR price from an exchange API and create a transaction that calls the `set_data` method on the contract, passing in the price and the corresponding timestamp.
4.  The DEX contract can then call `get_data_value` and `get_last_updated` on this Oracle contract to get the information and use it as input for its transactions.

The Oracle Node itself can be implemented in any programming language, but will require an off-chain service to interface with NEAR's RPC. It will need to be aware of both the deployed smart contract address and the approved account ID. The oracle should regularly retrieve external data and submit updates to the contract through NEAR transactions. Note, a timestamp should always be part of the data pushed to the smart contract as a basic replay-attack deterrent.

This approach allows the NEAR contract to remain deterministic while still utilizing external data. However, it's crucial to emphasize that this system depends on the trustworthiness of the Oracle Node itself. A malicious Oracle Node could supply incorrect data, potentially compromising the consuming smart contracts.

For further study of best practices, I recommend exploring the NEAR documentation on smart contracts, specifically the sections related to contract state and cross-contract calls. Moreover, delving into the official Rust documentation, specifically the `near_sdk` crate, will prove valuable.  Understanding the security implications of external data handling in smart contracts is of paramount importance. Researching how established oracles work, with an emphasis on decentralization, data verification, and incentive models, will provide key insights. Examining code examples of other NEAR projects that incorporate oracles will be beneficial. Also, study the core NEAR concepts such as transaction structure, account management, and the NEAR RPC interface will be helpful when creating the off-chain oracle components.
