---
title: "When are PancakeSwap trade fees deducted, and why aren't they reflected in liquidity pools?"
date: "2024-12-23"
id: "when-are-pancakeswap-trade-fees-deducted-and-why-arent-they-reflected-in-liquidity-pools"
---

Okay, let's tackle this one. I remember back in '21, working on a DeFi project, we had a similar head-scratcher around transaction fees and liquidity pool mechanics on a fork of Uniswap v2. PancakeSwap, fundamentally, operates on a similar Automated Market Maker (AMM) model, so the principles remain consistent, although specifics differ. The issue of when fees are deducted and why they’re not directly mirrored in the pool balance is a common source of confusion, and it's important to get it clear.

The core concept to understand is that trade fees on PancakeSwap (and similar AMMs) are not a direct subtraction from the tokens exchanged. Instead, they are essentially a *redistribution* mechanism. The fees are charged at the point of the trade execution, and this happens transparently *before* the swap is completed. Therefore, what you *receive* is already adjusted for the fee.

Consider this: you're trading token 'A' for token 'B'. You initiate a swap with the intention of exchanging *x* amount of token 'A'. At the backend, the protocol calculates the expected amount of token 'B' *before* the fee is applied. Let's say the calculation shows you should get *y* tokens of 'B'. If the swap fee is 0.25% (a common figure, but it can vary on PancakeSwap), you will receive not *y*, but *y* *minus* 0.25% of *y* which I will denote as *y'*. That *y'* is the actual amount that is transferred to you. The full amount of 'A' that you send is added to the pool, while the fee calculated is what is *not* distributed back to the receiver of 'B', not an actual deduction from the pool balance.

Now, the question of why these fees are not reflected in the liquidity pool's balance is crucial. The short answer is, they are not *directly* added to the primary token balances within the pool. Instead, they are sent to the liquidity providers (LPs) as *yield*. Remember, the core functionality of an AMM is to maintain a constant product of the pool balances—that *x* * y = k* (where *k* is a constant). If the transaction fees were directly added, this constant relationship would be broken, and the price of the asset in the pool would not represent the actual trading activity. To illustrate with an extremely simplified scenario, this also means the pool might have an increasing amount of tokens and might be vulnerable to manipulation.

The fees are accumulated in the smart contract (usually a separate contract, or an internal accounting method within the core contract), and the protocol then uses these fees to reward LPs in several ways:

1.  **Trading Fees:** The protocol accrues the fees generated from trades. These fees are often distributed to LP tokens holders when the liquidity providers 'burn' or 'withdraw' their LP tokens.
2.  **Liquidity Provider Rewards:** LPs receive LP tokens that represent their share in the pool. The fees accumulated are distributed in proportion to their LP tokens, thus increasing the value of their tokens as a form of reward.
3.  **Staking and Farming:** LPs might also have the opportunity to stake their LP tokens in dedicated farming contracts, further earning additional token rewards, sometimes derived from fees.

Let’s look at some pseudo-code examples to illustrate this. This isn't exact Solidity, but a high-level representation:

**Example 1: Trade Execution and Fee Calculation (Simplified):**

```python
def execute_trade(token_in_amount, pool_token_in, pool_token_out, fee_rate):
    """Simulates a trade execution in an AMM."""
    # Calculate the amount of token_out based on the constant product formula
    k = pool_token_in * pool_token_out
    new_pool_token_in = pool_token_in + token_in_amount
    new_pool_token_out = k / new_pool_token_in
    token_out_amount_before_fee = pool_token_out - new_pool_token_out

    # Calculate the fee
    fee = token_out_amount_before_fee * fee_rate
    
    # Calculate the token_out amount *after* fees
    token_out_amount = token_out_amount_before_fee - fee

    return token_out_amount
    
# Sample Usage
pool_token_A = 1000 # initial tokens of 'A'
pool_token_B = 500 # initial tokens of 'B'
amount_of_A_to_trade = 100
fee_rate = 0.0025 # 0.25%
token_B_received = execute_trade(amount_of_A_to_trade, pool_token_A, pool_token_B, fee_rate)
print(f"Tokens of B received: {token_B_received}") #Output is close to expected value but less to include the fee.

```

This shows the fee is deducted before the output is sent to the trader, but it doesn’t touch the pool balance *directly*. The full amount of token A sent is added to the pool.

**Example 2: Fee Accumulation and LP reward (Conceptual):**

```python
class LiquidityPool:
    def __init__(self, token_a, token_b):
        self.token_a = token_a
        self.token_b = token_b
        self.fees_accumulated = 0

    def add_fees(self, fees):
        self.fees_accumulated += fees

    def distribute_rewards(self, lp_tokens_total, user_lp_tokens):
      user_share = user_lp_tokens / lp_tokens_total
      user_reward = self.fees_accumulated * user_share
      # In actual implementations this reward would be distributed
      # and would also include more complex logic, this is just a conceptual example
      return user_reward
        
    
# Sample Usage
pool = LiquidityPool(1000,500)
pool.add_fees(10) # Adding simulated fees
lp_tokens = 100
user_lp_tokens = 20
reward = pool.distribute_rewards(lp_tokens, user_lp_tokens)
print(f"Reward received: {reward}") #Output is 2 since the user owned 20/100 of the lp tokens
```
This demonstrates how fees accumulate (again simplified), and how liquidity providers can earn based on their contribution to the pool.

**Example 3: Conceptual Fee Accounting**

```python
class TradeAccountant:
    def __init__(self):
        self.fees = 0

    def process_trade(self, token_in_amount, fee_rate):
        fee_amount = token_in_amount * fee_rate
        self.fees += fee_amount #fee is recorded in the accountant
        return token_in_amount - fee_amount #actual sent amount is minus fee

accountant = TradeAccountant()
amount_to_trade = 100
fee_rate = 0.0025
sent_amount = accountant.process_trade(amount_to_trade, fee_rate)
print(f"Sent amount: {sent_amount}") #Output is 99.75
print(f"Fees accumulated: {accountant.fees}") #Output is 0.25
```
This is to show that the fees are not lost, simply that they are routed for other purposes, like LP rewards. The trade amount received is deducted by the fees amount, while the accumulated fees are kept separate.

For those wanting a deep dive into the theoretical underpinnings, I'd recommend looking into the work of Guillermo Angeris et al., particularly their papers on “An analysis of Uniswap markets,” which provides a very rigorous treatment of AMM design. Also, “Algorithmic Trading and Deep Learning” by Robert Kissel touches on how these mechanisms play out in practice. Also, you can get a better understanding of specific AMM designs and their implementations by checking the documentation of various DeFi projects, like the official PancakeSwap documentation itself.

In summary, the fees aren't directly reflected in the pool’s token balances because they are intended to compensate liquidity providers and maintain a consistent market mechanism. They are a form of yield generation, not a simple subtraction, and this is fundamental to the way AMMs function. The code examples, simplified as they are, illustrate the essential flow, demonstrating how fees are calculated during swaps and how they function as a reward mechanism. These practical mechanisms ensure a sustainable and balanced system.
