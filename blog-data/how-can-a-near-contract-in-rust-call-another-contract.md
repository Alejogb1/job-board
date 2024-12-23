---
title: "How can a NEAR contract (in Rust) call another contract?"
date: "2024-12-23"
id: "how-can-a-near-contract-in-rust-call-another-contract"
---

Alright, let’s tackle this. Been there, done that – more times than I care to recall. Cross-contract calls in near can be a little… intricate, to put it mildly. It's certainly not a simple function invocation, but it’s absolutely crucial for building complex, modular applications. We’re basically orchestrating communication between separate, independent smart contracts, and that introduces both opportunities and challenges.

The core mechanism for this is asynchronous callbacks. Instead of a direct synchronous call, a contract initiates a promise to another contract, and the result of that promise is delivered via a callback function. It’s this asynchronous nature that differentiates cross-contract calls from regular function calls within the same contract. The NEAR runtime, being what it is, demands this promise-based paradigm for security and atomicity.

Here’s the breakdown of the process: first, you define an interface for the external contract you want to interact with. This interface describes the functions you intend to call. Next, you use the `near_sdk::Promise` to generate the call to the other contract. Importantly, you also need to register a callback function that near will execute *after* the external contract’s function completes. This callback receives the result from the external contract, and you can then process it within the original contract. It's crucial to handle potential errors during the external call within the callback.

Now, for some specifics. We will see this all in action with the following examples. In our case, we have two example contracts, 'caller\_contract' which makes a call and 'callee\_contract' which will receive and process it.

Let’s dive into some code examples. First, the 'callee\_contract':

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{near_bindgen, AccountId, env, PanicOnDefault};

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct CalleeContract {
    value: u32,
}

#[near_bindgen]
impl CalleeContract {
    #[init]
    pub fn new(initial_value: u32) -> Self {
        Self { value: initial_value }
    }

    pub fn get_value(&self) -> u32 {
        self.value
    }

    pub fn set_value(&mut self, new_value: u32) {
        self.value = new_value;
        env::log_str(format!("Value updated to: {}", self.value).as_str());
    }

    pub fn increment_value(&mut self, increment_by: u32) -> u32 {
        self.value += increment_by;
        env::log_str(format!("Value incremented by: {}, new value: {}", increment_by, self.value).as_str());
        self.value
    }
}
```

This 'callee\_contract' is straightforward. It has basic get, set and increment methods, and serves as the recipient of the cross-contract call.

Now, let’s look at the 'caller\_contract' that will invoke functions on the `callee\_contract`:

```rust
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{near_bindgen, AccountId, env, Promise, PromiseResult, PanicOnDefault};

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct CallerContract {
    callee_id: AccountId
}

#[near_bindgen]
impl CallerContract {
    #[init]
    pub fn new(callee_id: AccountId) -> Self {
        Self { callee_id }
    }

    pub fn call_callee_increment(&self, increment_by: u32) -> Promise {
      Promise::new(self.callee_id.clone())
        .function_call(
          "increment_value".to_string(),
          near_sdk::serde_json::to_vec(&increment_by).unwrap(),
          0,
          5_000_000_000_000
        )
        .then(Self::ext(env::current_account_id()).callback_increment())
    }

    #[private]
    pub fn callback_increment(&self, #[callback_result] call_result: Result<u32, near_sdk::PromiseError>) -> u32 {
         match call_result {
            Ok(result) => {
               env::log_str(format!("Callback result: {}", result).as_str());
               result
            },
            Err(err) => {
               env::log_str(format!("Error during callback: {}", err).as_str());
               0
            },
        }
    }

    pub fn call_callee_set(&self, new_value: u32) -> Promise {
      Promise::new(self.callee_id.clone())
          .function_call(
            "set_value".to_string(),
            near_sdk::serde_json::to_vec(&new_value).unwrap(),
            0,
            5_000_000_000_000
            )
            .then(Self::ext(env::current_account_id()).callback_set())
    }

    #[private]
    pub fn callback_set(&self, #[callback_result] call_result: Result<(), near_sdk::PromiseError>) -> bool {
       match call_result {
         Ok(_) => {
           env::log_str("Callback result: Set value successful".to_string().as_str());
           true
         },
         Err(err) => {
           env::log_str(format!("Error during callback: {}", err).as_str());
           false
          },
      }
    }

    pub fn call_callee_get(&self) -> Promise {
       Promise::new(self.callee_id.clone())
       .function_call(
          "get_value".to_string(),
          vec![],
          0,
          5_000_000_000_000
        )
        .then(Self::ext(env::current_account_id()).callback_get())
    }

    #[private]
    pub fn callback_get(&self, #[callback_result] call_result: Result<u32, near_sdk::PromiseError>) -> u32 {
      match call_result {
        Ok(result) => {
           env::log_str(format!("Callback result: Get value: {}", result).as_str());
          result
        },
        Err(err) => {
          env::log_str(format!("Error during callback: {}", err).as_str());
          0
        },
      }
    }

}
```

In `CallerContract`, you’ll notice several key components. First, the `call_callee_increment`, `call_callee_set` and `call_callee_get` are the functions that make calls to the `callee_contract`, and there are callbacks of `callback_increment`, `callback_set` and `callback_get`, each handle the different responses of the callee. The `Promise::new` method is used to build the request, along with method names as strings, arguments encoded using `near_sdk::serde_json::to_vec`, and gas limits. The `then` method associates each promise with the specific callback. The `@callback_result` attribute on the callback argument allows the callback to have access to the result of the promise.

It is imperative to handle the `Result` from the callback to check if there was an error on the cross-contract call. If an error occurs, the `Err` variant will contain the error, which you can then process or log.

For additional reading, I would highly recommend starting with the official NEAR documentation, particularly the sections on promises and cross-contract calls. The NEAR SDK GitHub repository is also a goldmine for examples. Beyond that, diving deep into the source code of the `near-sdk-rs` crate is often useful to understand the internals. The “Mastering Rust” book by Jim Blandy and Jason Orendorff is excellent to sharpen your understanding of the Rust language, which is essential for near development. And, of course, the NEAR whitepaper itself, while not directly about cross-contract calls, provides critical insights into the overall architecture.

These examples are somewhat simplified, of course. In production environments, you’ll need to be extra careful about error handling, potential security vulnerabilities, and efficient gas management. But with practice and a solid understanding of these principles, cross-contract calls in NEAR become a powerful tool for building robust and scalable decentralized applications. Remember, thorough testing is paramount before deploying anything to mainnet. Been burned more than once not paying enough attention to that!
