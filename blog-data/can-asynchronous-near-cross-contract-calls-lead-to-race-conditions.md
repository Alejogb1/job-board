---
title: "Can asynchronous NEAR cross-contract calls lead to race conditions?"
date: "2024-12-23"
id: "can-asynchronous-near-cross-contract-calls-lead-to-race-conditions"
---

, let's dive into this. I've seen my share of tricky situations with asynchronous cross-contract calls on NEAR, and believe me, race conditions are a genuine concern if not handled carefully. It's not as straightforward as a simple synchronous call, and that's where the complexity – and potential problems – lie.

My experience with a decentralized finance (defi) project a couple of years back really solidified my understanding of this. We were building a complex protocol involving multiple smart contracts, and we relied heavily on asynchronous cross-contract calls for efficient fund management and state updates. Initially, we weren't fully aware of the subtle traps we were setting. Specifically, we had a core contract that would initiate asynchronous calls to multiple auxiliary contracts. These auxiliary contracts, in turn, would update their own state based on the responses. We weren't thinking deeply enough about the possibility of these calls and their corresponding state changes occurring in a non-deterministic order. We discovered, through some rather expensive testing on the testnet, that we were susceptible to race conditions. The unpredictable nature of asynchronous execution meant that we could sometimes end up with inconsistent or incorrect state, depending on which callback fired first. That experience was a painful but valuable lesson.

So, the core issue here is that when you make an asynchronous call on NEAR, the current contract execution doesn't halt and wait for the call to return. Instead, the contract fires off the call and continues its operation, and at some later point, *if the called contract successfully completes*, a callback is triggered. The problem arises when multiple asynchronous calls are initiated from the same contract to either the same or different contracts, *all potentially modifying shared state*, whether that be state in the original contract or state in the recipient contract. Because the order in which these callbacks are executed isn't guaranteed, you can enter the territory of race conditions. The final state of your system can depend on the relative timing of these asynchronous events, and that timing can be influenced by things outside your control, such as gas costs and network conditions. This makes debugging and predicting the outcome significantly harder.

Let’s illustrate this with some hypothetical examples using pseudo-code, focusing on the conceptual issue rather than syntax:

**Example 1: Race Condition in a Simple Counter Contract**

Imagine two contracts: `counter_contract` and `incrementer_contract`. `counter_contract` holds a simple counter and has a function to update it. `incrementer_contract` makes asynchronous calls to `counter_contract` to increment it.

```pseudo
// counter_contract.near
state {
  counter: u64 = 0;
}

function increment_counter() {
  this.counter += 1;
  return true; // For callback success
}

function get_counter() -> u64 {
    return this.counter;
}

// incrementer_contract.near
function initiate_increments() {
    // asynchronous calls
    counter_contract.increment_counter(callback="handle_increment_result");
    counter_contract.increment_counter(callback="handle_increment_result");
}
function handle_increment_result() {
    // this method is not modifying counter_contract, just here as an example
    // imagine each call will eventually trigger this, but it's not guaranteed which callback comes first
    log("increment result received");

}
```

Here's the problem: if `incrementer_contract` initiates two asynchronous calls to increment `counter_contract`, and the calls aren’t processed in order, there’s a potential race. If the asynchronous calls are fired and the callback processing takes different amounts of time, there's no guarantee they will increment the counter sequentially. For example, if one call's increment updates the `counter` to 1, and before the second callback executes the same action, it may read the original state of the `counter`, also incrementing from 0, ending up at 1, instead of 2. The expected outcome is 2, but because of the race condition, the outcome is 1. This scenario occurs because both increment calls, in this simplified example, are not reading and updating the state atomically.

**Example 2: Race Condition with Shared State Modifications**

Let's get slightly more complicated. Suppose a `vault_contract` manages users' balances, and a `deposit_contract` initiates deposits using asynchronous calls.

```pseudo
// vault_contract.near
state {
    balances: HashMap<AccountId, u128>;
}

function deposit(account_id: AccountId, amount: u128) -> u128 {
   let current_balance = this.balances.get(account_id).unwrap_or(0);
   this.balances.insert(account_id, current_balance + amount);
   return this.balances.get(account_id).unwrap(); // Return new balance after the operation
}

function get_balance(account_id: AccountId) -> u128{
    return this.balances.get(account_id).unwrap_or(0);
}


// deposit_contract.near
function make_deposits(account_id: AccountId, amount: u128) {
    // asynchronous calls, both targeting the same account
    vault_contract.deposit(account_id, amount, callback="handle_deposit_result");
    vault_contract.deposit(account_id, amount, callback="handle_deposit_result");
}

function handle_deposit_result(result: u128){
        log ("new balance " + result.to_string());
}
```

Here, if `deposit_contract` makes two asynchronous `deposit` calls for the same `account_id` to the `vault_contract`, we face a similar issue. The callbacks to `vault_contract` could modify `balances` state in an unexpected order. One deposit may overwrite the other's balance increase if they happen to read and update the state concurrently without proper locking or atomicity, potentially losing funds.

**Example 3: Data Corruption Due to Multiple Callbacks**

Let's expand the example further to demonstrate how a race can corrupt derived state and introduce further complexity. Consider now `vault_contract` also calculates and stores total tokens deposited, and makes the token count available through another method.

```pseudo
// vault_contract.near
state {
    balances: HashMap<AccountId, u128>;
    total_tokens: u128 = 0;
}

function deposit(account_id: AccountId, amount: u128) -> u128 {
   let current_balance = this.balances.get(account_id).unwrap_or(0);
   this.balances.insert(account_id, current_balance + amount);
   this.total_tokens += amount;
   return this.balances.get(account_id).unwrap();
}

function get_balance(account_id: AccountId) -> u128{
    return this.balances.get(account_id).unwrap_or(0);
}

function get_total_tokens() -> u128{
    return this.total_tokens;
}

// deposit_contract.near (same as before)
function make_deposits(account_id: AccountId, amount: u128) {
    // asynchronous calls, both targeting the same account
    vault_contract.deposit(account_id, amount, callback="handle_deposit_result");
    vault_contract.deposit(account_id, amount, callback="handle_deposit_result");
}

function handle_deposit_result(result: u128){
        log ("new balance " + result.to_string());
}
```

In this slightly modified version of the previous example, if the race condition happens we could also end up with incorrect total token balance, as both calls read `total_tokens` state, update it, and persist. If the updates happen non-atomically, `total_tokens` could not accurately reflect the sum of all the deposits, and this would be a difficult error to trace, as `get_balance` method could be correct for individual users, but `get_total_tokens` will show discrepancies. This underscores that these issues can cascade and become more complex.

**Mitigation Strategies**

, so how do we deal with this? There isn't a single, silver bullet. It’s usually a combination of techniques. A crucial strategy is to **design your contracts and state updates to be idempotent** whenever possible. This means that executing the same operation multiple times should yield the same result as executing it once. In the counter example above, instead of incrementing, we could set the counter to a value computed from previous stored state and incoming parameters using a deterministic algorithm.

Another essential tool is to **use locking mechanisms** within your smart contracts, which NEAR provides through its storage access patterns. This helps prevent concurrent modifications to shared state. The *Contract Storage* section of the NEAR documentation is particularly useful for understanding how to leverage NEAR’s storage access patterns for atomic operations. I'd recommend spending time there. Also, check out the paper on *Formal Verification of Smart Contracts* if you want to get into more rigorous methods. There are several such papers you can find with a search, focusing on formally specifying contract logic and using tools to prove properties about its execution.

Another approach, which we used extensively in my previous project, is to **implement a queueing system** to manage asynchronous calls. Rather than initiating calls directly, we'd add them to a queue and process them sequentially, which eliminates the chance of race conditions arising from concurrent callbacks. You can also consider employing **optimistic concurrency control**, where you assume that conflicts will be infrequent and handle them when they occur. The basic principle is that the contract first reads state from the storage, then performs computations, and then commits the changes in a single transaction (similar to the atomic methods described previously). If the state was modified between the read and the commit, the whole process is rolled back, and the method is retried.

In conclusion, asynchronous cross-contract calls on NEAR *can* absolutely lead to race conditions if not handled correctly. It's crucial to understand the execution flow, consider potential concurrency issues, and apply the appropriate mitigation techniques. There's no one-size-fits-all answer; it depends on the specifics of your contract logic and interactions. My advice is to think thoroughly about how asynchronous operations interact, rigorously test, and take advantage of the atomicity mechanisms offered by the platform, and you will avoid most headaches.
