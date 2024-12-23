---
title: "How are all obligations in the SPL token lending program identified?"
date: "2024-12-23"
id: "how-are-all-obligations-in-the-spl-token-lending-program-identified"
---

, let’s dissect how obligations are tracked within an spl token lending program. From my experience building a similar system, this isn't just about a simple ledger; it's about establishing a robust framework to accurately represent each participant’s position within the lending pool. We're essentially creating a system where we can, at any given point, definitively identify who owes what, and who is owed what. The complexity comes from managing changing loan balances, interest accrual, and potential liquidations.

Fundamentally, an obligation within an spl token lending program is a data structure, typically a program-owned account on the blockchain, that encapsulates all relevant details pertaining to a specific lender or borrower’s position. Each obligation is uniquely identified, usually using a combination of the associated user’s public key, a specific market identifier for the lending pool, and an optional seed to ensure uniqueness, even if the user participates multiple times in the same market. This structure forms the core of the lending program’s ability to enforce its rules.

Let’s consider the key elements stored within this obligation structure. Firstly, we have the associated user's address – this identifies *who* holds the obligation. Next, we must store information about the principal balances; we typically have separate fields for borrowed amounts and deposited collateral amounts for each token. We often use associated token accounts addresses that represent where the lender deposited his assets, and the borrower received them. These token accounts are also uniquely associated with the given obligation, thereby providing a direct linkage. Another critical element is tracking interest. Rather than storing the interest due directly (which would require constant updates), most implementations store an index and a timestamp representing the last time interest was calculated. This approach reduces computation burden during normal operations, as interest is calculated only when an interaction happens with the obligation. The program also stores information that the user borrowed against the supplied collateral to calculate the health factor, which we’ll get to later. Finally, the obligation includes various flags and constants to determine the state of the obligation itself, including potential liquidation status.

To understand this practically, consider these examples. These examples will not include actual program instructions, but rather focus on the data structure layout to showcase the points mentioned above.

**Example 1: Basic Obligation Data Structure (Rust-like struct)**

```rust
struct Obligation {
    owner: Pubkey,            // The address of the user owning the obligation.
    market: Pubkey,           // The lending market the obligation is tied to.
    deposited_tokens: Vec<TokenBalance>, // Vector of deposits, associated token account, and balances of each deposited token.
    borrowed_tokens: Vec<TokenBalance>, // Vector of borrowings, associated token account, and balances of each borrowed token
    last_update_time: i64,    // Timestamp of last interest calculation (unix timestamp).
    liquidation_flag: bool, // flag to indicate the state of the liquidation.
    borrow_health: u64,      // Health factor, representing the state of borrow vs. collateral.
    seed: [u8; 32],             // Unique seed for the obligation.
}

struct TokenBalance {
    token_mint: Pubkey,    // Token type.
    balance: u64,           // Balance of the token.
    token_account: Pubkey, // Associated token account.
}
```

In this basic structure, `owner` represents the user who created the obligation, `market` specifies the pool, `deposited_tokens` and `borrowed_tokens` are vectors that contain the balances of all tokens deposited as collateral or borrowed with their associated token accounts, `last_update_time` is the last time interest was accrued, `liquidation_flag` shows the current state of the liquidation, `borrow_health` indicates if the obligation is above or below the borrowable threshold, and `seed` provides a unique ID.

Now, let’s illustrate how interest calculation integrates with this obligation. Instead of continuously updating interest fields, we calculate accrued interest only when interacting with the obligation.

**Example 2: Interest Accrual Logic (pseudo-code)**

```python
def calculate_interest_due(obligation: Obligation, current_time: int, market_rate: float) -> dict:
    time_elapsed = current_time - obligation.last_update_time
    if time_elapsed <= 0:
        return {"borrowed": obligation.borrowed_tokens, "deposited": obligation.deposited_tokens} # No interest since last update

    new_borrowed = []
    new_deposited = []
    for borrowed_token in obligation.borrowed_tokens:
        interest_amount = borrowed_token.balance * market_rate * time_elapsed / (365*24*60*60) # Assuming an annual rate.
        new_borrowed.append(
             {
                "token_mint": borrowed_token.token_mint,
                "balance": borrowed_token.balance + interest_amount,
                "token_account": borrowed_token.token_account
            }
        )
    for deposited_token in obligation.deposited_tokens:
        interest_amount = deposited_token.balance * market_rate * time_elapsed / (365*24*60*60) # Assuming an annual rate.
        new_deposited.append(
            {
                "token_mint": deposited_token.token_mint,
                "balance": deposited_token.balance + interest_amount,
                "token_account": deposited_token.token_account
            }
        )


    return {"borrowed": new_borrowed, "deposited": new_deposited}


def update_obligation_with_interest(obligation: Obligation, new_borrowed_tokens: list, new_deposited_tokens: list, current_time: int) -> Obligation:
   return Obligation(
        owner = obligation.owner,
        market = obligation.market,
        borrowed_tokens = new_borrowed_tokens,
        deposited_tokens = new_deposited_tokens,
        last_update_time = current_time,
        liquidation_flag = obligation.liquidation_flag,
        borrow_health = obligation.borrow_health,
        seed = obligation.seed
   )
```

This pseudo-code demonstrates that we read the previous obligation data, calculate the new borrowed and deposit balances based on a current timestamp, then return a new obligation with the updated data. We'd then save this updated `Obligation` to the blockchain via a program instruction. Crucially, we are not continuously re-evaluating the interest, which reduces the load.

Finally, let’s look at how this data structure can identify a user’s outstanding obligations and collateral. Using the unique identifier for the obligations (combinations of the user public key, market id, and seed) we can fetch an obligation based on the relevant address and then evaluate a user's balances. The program maintains indexes that allow easy lookups of obligations based on the associated users and the markets they participated in.

**Example 3: Fetching obligations (pseudo-code)**

```python
def get_user_obligations(user_address: str, lending_market_id: str, all_obligations: dict) -> list:
    user_obligations = []
    for obligation_id, obligation in all_obligations.items():
        if obligation.owner == user_address and obligation.market == lending_market_id:
            user_obligations.append(obligation)
    return user_obligations

def display_balances(user_obligations: list):
    for obligation in user_obligations:
        print(f"Obligation Owner: {obligation.owner}, Market: {obligation.market}")
        for token_balance in obligation.deposited_tokens:
              print(f"  Deposited {token_balance.token_mint}: {token_balance.balance}")
        for token_balance in obligation.borrowed_tokens:
            print(f"  Borrowed {token_balance.token_mint}: {token_balance.balance}")
        print(f"  Health Factor: {obligation.borrow_health}")
        print("---")

# Example Usage:
all_obligations = {
    "obligation1": Obligation(
        owner = "user_1_address",
        market = "market_a",
        deposited_tokens=[{"token_mint": "tokenA", "balance": 100, "token_account":"account1"}],
        borrowed_tokens = [{"token_mint": "tokenB", "balance": 20, "token_account":"account2"}],
        last_update_time=1678886400,
        liquidation_flag = False,
        borrow_health = 80,
        seed = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
     ),
     "obligation2": Obligation(
          owner="user_1_address",
          market="market_b",
          deposited_tokens = [{"token_mint": "tokenC", "balance": 50, "token_account": "account3"}],
          borrowed_tokens= [{"token_mint": "tokenD", "balance": 10, "token_account": "account4"}],
          last_update_time=1678886400,
          liquidation_flag=False,
          borrow_health=90,
          seed= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ),
    "obligation3": Obligation(
        owner="user_2_address",
         market="market_a",
         deposited_tokens = [{"token_mint": "tokenA", "balance": 200, "token_account": "account5"}],
         borrowed_tokens=[{"token_mint": "tokenB", "balance": 100, "token_account": "account6"}],
         last_update_time=1678886400,
         liquidation_flag=False,
         borrow_health=50,
         seed = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

    )
}

user_obligations_a = get_user_obligations("user_1_address", "market_a", all_obligations)
display_balances(user_obligations_a)
# Output:
# Obligation Owner: user_1_address, Market: market_a
#   Deposited tokenA: 100
#   Borrowed tokenB: 20
#   Health Factor: 80
# ---
user_obligations_b = get_user_obligations("user_1_address", "market_b", all_obligations)
display_balances(user_obligations_b)
# Output:
# Obligation Owner: user_1_address, Market: market_b
#  Deposited tokenC: 50
#  Borrowed tokenD: 10
#  Health Factor: 90
# ---

```

This code shows how to query all the obligations and then filter down to a specific user. I have included example output to ensure the purpose of the code is clear.

For further reading, I'd suggest reviewing the documentation for existing Solana lending programs like Mango and Solend, which often provide detailed explanations of their data structures. Further, research papers on decentralized finance and on-chain lending protocols can also provide invaluable context, particularly for concepts related to interest rate models and risk assessment. Also, consider looking into the Solana Program Library (SPL) documentation, which outlines the lower level structure of these kinds of programs. The Solana cookbook is also a useful resource.

In conclusion, obligation management within an spl token lending program isn't a singular task but a combination of data structure design, efficient interest calculation, and effective indexing. It’s a nuanced design that ensures proper tracking of each user's interactions within the lending pool. The key is a system which clearly delineates and stores each obligation and the details that come with it, and allows easy retrieval and management.
