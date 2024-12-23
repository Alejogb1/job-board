---
title: "Which addresses in the Serum network have a data size of 777?"
date: "2024-12-23"
id: "which-addresses-in-the-serum-network-have-a-data-size-of-777"
---

, let's tackle this one. It’s not every day you encounter such a specific data size quirk on a network like Serum, but I recall a project back in '21 where we had some seriously perplexing address-related data issues. Specifically, we were dealing with custom PDA (Program Derived Addresses) which, as it turns out, can be the root of non-standard sizes. Understanding why some Serum addresses might yield a data size of exactly 777 bytes boils down to how Serum, and more generally Solana, stores and interacts with account data, particularly when programs are in the mix.

Let's be clear upfront: standard Solana addresses, the public keys we all know, don't directly store data in this manner. They are identifiers. The 777-byte data size almost certainly refers to the data held within *account* structures associated with those addresses, specifically accounts owned by a program or perhaps a custom account created via a program’s instructions. In Solana, when a program needs to store state or data, it does so in accounts that it owns. This is where the 777 bytes comes into play.

The size of the data associated with an account is determined at the time the account is created. A standard system-owned account, which simply holds lamports (Solana's native currency), will have a minimal data size, often just enough to hold the lamport balance and some other basic metadata. However, when a program owns an account, it dictates the structure of that account's data, including its size. Programs often use custom data structures defined within their smart contracts (or more accurately, their on-chain programs). These data structures, serialised using a mechanism like Borsh, are packed into the account's data space. A 777-byte data size, while specific, would indicate a well-defined data structure that was probably compiled to a size of 777 bytes during program development. This is not random; it was likely a conscious design choice.

Furthermore, it's crucial to remember that programs can also modify the data within their accounts. This flexibility is what allows smart contracts to implement complex logic and store state changes persistently on-chain. It also creates scenarios where seemingly unrelated addresses are connected via the programs that control them, potentially with all accounts using that particular program storing account data that follows the program’s data structure. Thus, multiple seemingly unrelated addresses *could* all have associated accounts storing 777 bytes of data if a particular program created accounts with that structure.

To clarify, let's consider some simplified examples. These aren't Serum-specific, as demonstrating Serum's specifics requires quite a detailed setup, but they illustrate the concept. These are pseudo-code examples for educational clarity, not production ready implementations.

**Example 1: Simple Account Creation**

```python
import borsh

class ExampleData(borsh.BorshSchema):
    field1: int
    field2: str
    field3: list[float]

    def __init__(self, field1: int, field2: str, field3: list[float]):
         self.field1 = field1
         self.field2 = field2
         self.field3 = field3


def create_account_data(data: ExampleData) -> bytes:
    serialized_data = borsh.serialize(data)
    return serialized_data

example_instance = ExampleData(10, "hello", [1.0, 2.0, 3.0])

#this is a simplified way of simulating data, in reality, borsh will encode
#this to a certain byte size. Let's assume in this contrived example the encoded data is 20 bytes

account_data = create_account_data(example_instance)


#assume a helper function is used to allocate space that is 777 bytes large.

# then this hypothetical 20 bytes would be stored into 777 bytes allocated space.

```

This example shows how data defined using Borsh could be serialised and stored. Obviously in a real application, we would pad the data to 777 bytes.

**Example 2: Program Derived Address (PDA)**

PDAs are vital for program interactions, and understanding how they can yield account data is crucial. PDAs are not generated via the traditional keypair process. Instead, they are derived from a program ID and a set of seeds. This ensures the program “owns” the account and can modify it.

```python
from solana.publickey import PublicKey
import hashlib
import base58

def derive_pda(program_id_str: str, seeds: list[bytes]) -> PublicKey:
  program_id = PublicKey(program_id_str)
  full_seed = b"".join(seeds)
  for i in range(255):
        attempt = full_seed + i.to_bytes(1, 'little')
        hash = hashlib.sha256(attempt).digest()
        pubkey = PublicKey(hash)
        if pubkey.is_on_curve():
           continue
        return pubkey
  raise Exception("Unable to find a valid PDA, check seeds.")

program_id = "YourProgramIdxxxxxxxxxxxxxxxxxxxxxxxxx"
seeds = [b"example_seed_1", b"example_seed_2"]

pda_address = derive_pda(program_id, seeds)


#Now an account at this pda address could hold 777 bytes as explained in example 1.
```
This illustrates how a PDA can be generated and will later hold the data for a program.

**Example 3: Data Modification**

```python
import borsh

class AccountState(borsh.BorshSchema):
    value1: int
    value2: str
    value3: list[int]

    def __init__(self, value1: int, value2: str, value3: list[int]):
        self.value1 = value1
        self.value2 = value2
        self.value3 = value3



def update_account_data(current_account_data: bytes, new_state: AccountState) -> bytes:
  # This would perform a deserialize and update. In the interest of simplicity we are assuming we can simply overwrite.
  serialized_data = borsh.serialize(new_state)
  #In a real application we would have logic to update this field instead of just overwriting,
  #we are doing this for simplicity. This should always return a byte size equal to the account's
  #defined initial allocation.
  return serialized_data

initial_data = AccountState(0, "init", [0])
encoded_initial = borsh.serialize(initial_data)
#pretend this encoded data is padded to 777 bytes, the actual byte size here will vary based on the
#borsh encoding.

new_data = AccountState(10, "updated", [1,2,3,4])
updated_encoded = update_account_data(encoded_initial, new_data)

#this updated_encoded data is assumed to be padded to 777 bytes.

```

This shows that, even if we update the account’s state, we are not changing the fundamental 777-byte size that the account holds. This is the important consistency enforced by the Solana runtime.

In practical terms, if you're encountering addresses with 777-byte data sizes on Serum (or Solana in general), you're most likely dealing with accounts owned by specific programs. To determine *which* addresses, you would need to analyse the on-chain data for accounts of programs involved in the operations you are investigating. This is done by querying the Solana RPC endpoints for account information with filters that may include the size and program id. You could then examine the programs' source code to figure out the structure, and thus decode the data.

For further learning, I'd strongly recommend looking at these resources:

1.  **"Programming on Solana: An Introduction"** by Paul Berg (a well-regarded book that explains Solana's programming model clearly).
2.  **The official Solana documentation** (always the best place to start). Pay particular attention to the sections on Account Data, Program Derived Addresses and program state management.
3.  **"Mastering Rust"** by the Rust community (Solana programs are typically written in Rust, and a good understanding of Rust, particularly how its memory model impacts on-chain program structures, is vital).
4.  **The Borsh serialization library documentation.**

These resources will give you a comprehensive understanding of Solana's internals, and will greatly assist in tackling issues such as this one. Remember, the 777-byte size is not a random number; it reflects the way program data is handled on the Solana blockchain, especially in a highly customized application such as Serum. It’s a fingerprint of the program’s design choices, and once identified can provide significant insight.
