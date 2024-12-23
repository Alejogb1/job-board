---
title: "Why does a Rust ink cross-contract call return ContractTrapped?"
date: "2024-12-23"
id: "why-does-a-rust-ink-cross-contract-call-return-contracttrapped"
---

, let's unpack this. *ContractTrapped*, that old chestnut, especially when dealing with ink!'s cross-contract calls in Rust. I've seen my fair share of head-scratching moments tracing that particular error back to its roots. It's not always immediately apparent, but usually, it boils down to a few common culprits in the interplay between your smart contracts.

Frankly, encountering a `ContractTrapped` result in a cross-contract call within ink! usually points to an unrecoverable error occurring within the *target* contract during the execution of the called function. Think of it less as your calling contract doing something wrong directly, and more as the target contract experiencing a catastrophic failure that propagates back. The important detail is that `ContractTrapped` is a deliberately vague, broad-spectrum error on the *caller's* side because you can't always know exactly what went wrong within the confines of another contract's execution context.

Let’s explore the underlying reasons for this. The primary mechanisms for such failures can be categorized into a few key areas:

1.  **Panics and Unhandled Exceptions:** The most straightforward reason. If the target contract's function encounters a panic – perhaps due to an assertion failure, an out-of-bounds access, a division by zero, or any other condition that triggers a `panic!` macro or an unhandled `Result::Err` case – it unwinds the call stack. This unwinding propagates through the contract call machinery as an unrecoverable state, leading to `ContractTrapped` being returned to the caller. The target contract itself doesn't return a clean, handled error, but effectively 'crashes' within the call.

2.  **Insufficient Gas:** Smart contracts on blockchain platforms operate with strict gas limits. If the target contract's execution requires more gas than the caller provides or is available in the context of execution, the execution is halted. This out-of-gas condition results in a `ContractTrapped`. This doesn't mean the caller necessarily *sent* insufficient gas, but that during the entire call chain, there wasn’t sufficient gas *available* to finish the called contract’s operation. Sub-calls and memory allocation all consume gas.

3.  **Contract State Corruption:** This one is less common but more insidious. If the target contract, through some logical flaw, corrupts its own state during execution (e.g., writing to incorrect memory locations or not correctly storing results that future calls rely upon), a subsequent call might encounter issues that also lead to a panic or unrecoverable error, hence trapping the transaction. This isn’t direct corruption caused by the caller, but induced by its execution within that target. It can lead to subtle, hard-to-diagnose situations over time.

4.  **Contract Version Mismatch or Incorrect Contract Instantiation:** When calling a contract, you’re referencing its contract address. If the contract at that address isn't what you expect (wrong version, not initialized correctly, the code has been swapped out), unexpected behavior or a panic can ensue, leading to the `ContractTrapped` result. This points to a setup or deployment issue, not necessarily a code issue.

To illustrate with some examples, let’s start with a basic case that demonstrates a target contract panicking. I'll present the snippets in simplified form for clarity, highlighting the crucial parts:

```rust
// Target contract - let's call it contract_b
#[ink::contract]
mod contract_b {
    #[ink(storage)]
    pub struct ContractB {
    }

    impl ContractB {
        #[ink(constructor)]
        pub fn new() -> Self {
           Self {}
        }

        #[ink(message)]
        pub fn failing_function(&self) -> u32 {
           panic!("Intentional panic in B");
           0 // This line is unreachable due to panic above, but rustc requires this to be present.
        }
    }
}
```

Now, the calling contract (let’s call this one `contract_a`) would call `failing_function` using ink!'s cross-contract machinery:

```rust
// Calling contract - contract_a
#[ink::contract]
mod contract_a {
   use contract_b::ContractB;
    #[ink(storage)]
    pub struct ContractA {
        contract_b: ContractRef
    }

    type ContractRef = contract_b::ContractRef;

    impl ContractA {
       #[ink(constructor)]
       pub fn new(contract_b_code_hash: Hash) -> Self {
          let contract_b = Self::env()
          .instantiate_contract(
             contract_b_code_hash,
             0,
             ContractB::new(),
          )
          .unwrap_or_else(|error| {
             panic!("Failed to instantiate contract B: {:?}", error);
           });

          Self {
             contract_b,
          }
       }
       #[ink(message)]
        pub fn call_failing_contract_b(&self) -> Result<u32, ink::LangError> {
            self.contract_b.failing_function()
        }
    }
}
```

When `call_failing_contract_b` is executed, `failing_function` panics, causing `call_failing_contract_b` to return an `Err(LangError::ContractTrapped)` result. We haven't handled anything wrong on `contract_a`, but the effect of a panic in `contract_b` propagates.

Secondly, let's demonstrate an out-of-gas situation. Gas calculation is inherently complex and tied to the runtime environment of the chain you're on, but here's a simulation where we perform some arbitrary computation within the target contract:

```rust
// Target contract - contract_c
#[ink::contract]
mod contract_c {
    #[ink(storage)]
    pub struct ContractC {
    }

    impl ContractC {
        #[ink(constructor)]
        pub fn new() -> Self {
           Self {}
        }

        #[ink(message)]
        pub fn resource_intensive_function(&self, n: u32) -> u32 {
           let mut result = 0;
           for i in 0..n {
               for j in 0..n {
                result = result.wrapping_add(i.wrapping_mul(j));
               }
           }
            result
        }
    }
}
```

Here, if the value of `n` passed from the calling contract is too large for the gas provided within the cross-contract call context, the transaction will terminate due to out-of-gas, thus the result is `ContractTrapped`:

```rust
// Calling contract - contract_d
#[ink::contract]
mod contract_d {
    use contract_c::ContractC;
    #[ink(storage)]
    pub struct ContractD {
       contract_c: ContractRef
    }

    type ContractRef = contract_c::ContractRef;

    impl ContractD {
        #[ink(constructor)]
        pub fn new(contract_c_code_hash: Hash) -> Self {
          let contract_c = Self::env()
          .instantiate_contract(
              contract_c_code_hash,
              0,
              ContractC::new(),
          )
          .unwrap_or_else(|error| {
              panic!("Failed to instantiate contract C: {:?}", error);
          });
          Self {
              contract_c,
          }
        }

        #[ink(message)]
        pub fn call_intensive_contract_c(&self, input: u32) -> Result<u32, ink::LangError> {
           let gas_limit = 100000; // Assuming this is insufficient gas
           self.contract_c.call()
              .gas_limit(gas_limit)
              .resource_intensive_function(input)
        }
    }
}
```

Again, calling `call_intensive_contract_c` will frequently result in a `ContractTrapped`, even if the input number is not excessively large, depending on what gas costs the runtime calculates based on the operations performed within `resource_intensive_function`.

Lastly, let’s illustrate a scenario where contract state corruption might lead to the error, albeit this is more contrived for illustrative purposes, as it's tricky to simulate random memory corruption directly in a controlled setting. This is more a case of a bug within contract_e itself and its state than direct interaction from outside

```rust
// Target contract - contract_e
#[ink::contract]
mod contract_e {
    #[ink(storage)]
    pub struct ContractE {
        data: Vec<u32>,
    }

    impl ContractE {
       #[ink(constructor)]
       pub fn new() -> Self {
          Self { data: Vec::new() }
       }

      #[ink(message)]
      pub fn add_data(&mut self, value: u32) {
         let len = self.data.len();
         // Intentionally writing to an index past the end of the vector.
         self.data[len] = value;
      }
    }
}

```

In this highly contrived case, the `add_data` function attempts to write past the bounds of the `data` vector. Since Rust vector access is bounds checked, it will lead to an out-of-bounds panic and thus lead to `ContractTrapped` return on any caller. This is an error not directly caused by an outside caller, but caused due to logic and state corruption inside of `contract_e`, that makes the contract itself unable to continue running.

For a deeper understanding, I recommend studying the ink! documentation thoroughly, specifically the sections on cross-contract calls and error handling. Additionally, exploring the source code of the `pallet-contracts` runtime in Substrate will offer further insight into how contract calls are executed at a lower level. Also, a great resource is the book "Programming Substrate" by Bryan Chen. Look into sections on contract instantiation and execution lifecycles. Finally, exploring the Rust standard library documentation on error handling using the Result type is fundamental, particularly in how to handle the returned errors from the cross-contract calls.

In summary, `ContractTrapped` isn't a failure on the caller’s side directly but an indication of a catastrophic, unhandled error within the target contract. Careful debugging and testing of your target contracts, alongside understanding gas limits, are crucial to resolving this issue. It’s often a sign that one of your contracts needs a little more attention.
