---
title: "How can I retrieve Solana NFT holders using Python?"
date: "2024-12-23"
id: "how-can-i-retrieve-solana-nft-holders-using-python"
---

Okay, let's tackle this. It's a request that surfaces often, and having personally worked on several NFT marketplace integrations, I've encountered various challenges in getting this information reliably. Retrieving Solana NFT holders with Python involves interacting with the Solana blockchain, which, while efficient, doesn't provide a single, straightforward call to get a list of all holders of a given NFT mint. Instead, we need to employ a combination of methods and understand the underlying data structures.

The core challenge stems from how Solana handles NFTs. Unlike some other blockchains, Solana doesn't maintain an explicit 'list of holders' for each NFT. Ownership is determined by the state of accounts on the blockchain. An NFT is essentially a token, and the token account that holds it defines ownership. Thus, our approach must revolve around finding these token accounts.

The process generally follows these steps:

1.  **Identify the NFT's Mint Address:** This is the unique identifier of the NFT.
2.  **Retrieve all Token Accounts:** We need to query the blockchain for *all* token accounts associated with the specified mint address.
3.  **Filter for Non-Zero Balances:** Token accounts can exist with zero balance, indicating a past holder. We must filter out such accounts, keeping only those with a balance greater than zero.
4.  **Extract Owner Addresses:** For each valid token account, we extract its owner, which is the address of the holder.

This requires us to leverage a Solana RPC (Remote Procedure Call) endpoint, and a Python library suitable for interacting with it. I generally prefer `solana-py`, a robust Python library that provides excellent bindings for Solana's RPC. You can install it with `pip install solana`.

Now, let's dive into a practical implementation with a series of code examples. These examples showcase the essential steps, incorporating error handling and optimizations based on my experience.

**Example 1: Basic Retrieval**

This first example demonstrates the fundamental approach, aiming for clarity and minimal complexity. It will fetch and display holders of a given NFT mint address. Note that this version does not perform any advanced pagination or rate-limiting.

```python
from solana.rpc.api import Client
from solana.rpc.types import MemcmpOpts
from solana.publickey import PublicKey

def get_nft_holders(mint_address: str, rpc_endpoint: str) -> list:
    client = Client(rpc_endpoint)
    mint_pubkey = PublicKey(mint_address)
    holders = []

    # create memcmp filter to find token accounts for given mint
    memcmp_opts = [MemcmpOpts(
      offset=0,
      bytes=str(mint_pubkey)
      )]

    try:
      resp = client.get_program_accounts(
          PublicKey("TokenkegQfeZyiNwAJjE1mRiK23oGf6Xf1zGDnqVrZkb"),
          encoding = "jsonParsed",
          filters=[{"memcmp": memcmp_opts[0]}],
      )
      if resp and resp.value:
        for account in resp.value:
          if account.account.data.parsed.info.tokenAmount.uiAmount > 0:
            holders.append(account.account.data.parsed.info.owner)
    except Exception as e:
      print(f"Error fetching token accounts: {e}")
      return []
    return holders

if __name__ == '__main__':
    rpc_url = "https://api.mainnet-beta.solana.com" # Use Mainnet RPC
    mint_address_example = "EPjFWdd5AufqssvQTvTsdoug7jLmfKAgk2czay1n31h" # Example mint address for USDC
    holders_list = get_nft_holders(mint_address_example, rpc_url)
    if holders_list:
      print(f"Holders for {mint_address_example}:")
      for holder in holders_list:
        print(holder)
    else:
        print(f"Could not fetch holders for {mint_address_example}")
```

This snippet establishes a connection to the Solana RPC using `solana.rpc.api.Client`, defines the mint address, and then uses `get_program_accounts` to fetch token accounts matching the mint address. The `memcmp` filter is essential, allowing us to efficiently query only token accounts related to the specific mint. The result is then filtered to keep only those with a positive balance, and their owner addresses are added to a list.

**Example 2: Pagination Handling**

Real-world scenarios often involve NFTs with a large number of holders, exceeding the limit of a single `get_program_accounts` call. To address this, we need to implement pagination using the `until` and `before` parameters of the RPC call. This example shows how to handle a paginated response.

```python
from solana.rpc.api import Client
from solana.rpc.types import MemcmpOpts
from solana.publickey import PublicKey
from typing import List, Optional

def get_nft_holders_paginated(mint_address: str, rpc_endpoint: str) -> list:
  client = Client(rpc_endpoint)
  mint_pubkey = PublicKey(mint_address)
  holders = []
  last_pubkey = None
  while True:
      memcmp_opts = [MemcmpOpts(
        offset=0,
        bytes=str(mint_pubkey)
      )]

      try:
          filters = [{"memcmp": memcmp_opts[0]}]
          if last_pubkey:
            filters.append({"dataSize":165}) #Token accounts have a fixed size
            resp = client.get_program_accounts(
                  PublicKey("TokenkegQfeZyiNwAJjE1mRiK23oGf6Xf1zGDnqVrZkb"),
                  encoding = "jsonParsed",
                  filters=filters,
                  before = str(last_pubkey)
              )
          else:
             resp = client.get_program_accounts(
                PublicKey("TokenkegQfeZyiNwAJjE1mRiK23oGf6Xf1zGDnqVrZkb"),
                encoding = "jsonParsed",
                filters=filters,
                )

          if not resp or not resp.value:
            break

          for account in resp.value:
              if account.account.data.parsed.info.tokenAmount.uiAmount > 0:
                  holders.append(account.account.data.parsed.info.owner)
              last_pubkey = PublicKey(account.pubkey)

      except Exception as e:
          print(f"Error fetching token accounts: {e}")
          return holders
  return holders


if __name__ == '__main__':
    rpc_url = "https://api.mainnet-beta.solana.com"  # Use Mainnet RPC
    mint_address_example = "EPjFWdd5AufqssvQTvTsdoug7jLmfKAgk2czay1n31h"  # Example mint address
    holders_list = get_nft_holders_paginated(mint_address_example, rpc_url)
    if holders_list:
        print(f"Holders for {mint_address_example} (paginated):")
        for holder in holders_list:
          print(holder)
    else:
        print(f"Could not fetch holders for {mint_address_example}")

```

In this version, a `while` loop is used to iteratively fetch token accounts. After each call, the last retrieved `pubkey` is used as the `before` cursor for the next call, enabling us to paginate over the entire list. This ensures we can retrieve all holders, regardless of the number.

**Example 3: Advanced Rate Limiting**

While not included in the core examples, be aware that frequent requests to public RPC endpoints can result in rate limiting or even IP bans. Implementing a proper rate-limiting strategy is crucial. This may involve techniques such as exponential backoff, using multiple RPC endpoints, or leveraging a dedicated infrastructure provider such as QuickNode. Rate limiting isn't something that's demonstrated directly here in this text, but is a critical aspect of integrating with Solana in production environments. You’ll want to consider libraries like `tenacity` for this purpose.

**Recommended Resources:**

For a deeper understanding of Solana and its programming model, I strongly recommend the following resources:

*   **Solana Documentation:** The official documentation is invaluable: [docs.solana.com](https://docs.solana.com). It provides in-depth information about the protocol, APIs, and programming concepts.
*   **Programming on Solana by Brian Friel:** This book serves as an excellent guide, walking through the fundamentals of Solana development, with detailed explanations of account models and on-chain programs.
*   **Anchor Framework Documentation:** For those interested in developing smart contracts on Solana, this documentation is very good. It is a framework for building secure and efficient programs on the Solana blockchain.
*   **The Solana Cookbook:** The Solana Cookbook offers many recipes and tutorials with practical applications and code examples. It's perfect for hands-on learning.

In conclusion, retrieving Solana NFT holders using Python requires a careful approach to working with the blockchain. The examples above demonstrate the core functionality and crucial considerations like pagination. Remember to use appropriate rate limiting strategies in production scenarios, and use recommended resources for a solid understanding of Solana's architecture. As with all things on-chain, things are evolving, so always refer back to the main reference documentation for any new adjustments or nuances.
