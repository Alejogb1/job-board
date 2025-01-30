---
title: "Why isn't my sushi swap returning?"
date: "2025-01-30"
id: "why-isnt-my-sushi-swap-returning"
---
The failure of a sushi swap to return anticipated output is frequently rooted in a misunderstanding of the underlying Automated Market Maker (AMM) mechanics, especially within environments with low liquidity or significant price volatility. I've seen this happen countless times when debugging DeFi applications, particularly with decentralized exchanges (DEXs) like those emulating SushiSwap.

Let’s unpack the common causes for a ‘stuck’ or unexpectedly failed swap. The crucial principle at play here is the constant product formula, often expressed as *x* * *y* = *k*, where *x* represents the quantity of token A, *y* represents the quantity of token B, and *k* is a constant. The swap mechanism fundamentally relies on preserving this constant within a pool. When you attempt a swap, you’re essentially adding your input token, which increases its pool quantity, while simultaneously decreasing the quantity of the output token you desire. The algorithm determines the amount of output tokens based on ensuring the *k* value remains consistent (or as near as possible, considering fees).

One common reason for a swap failing is slippage exceeding tolerance. Users often configure a maximum slippage percentage during the transaction initiation. Slippage refers to the difference between the initially quoted swap price and the actual execution price. When a large swap is executed in a low liquidity pool or during periods of high volatility, the price can shift significantly before your transaction is finalized on the blockchain. If this price shift exceeds your specified tolerance, the transaction will fail, preventing a swap from occurring, and usually returning your initial tokens.

Gas fees present another obstacle. Insufficient gas settings can cause your transaction to remain unconfirmed. Every operation on the blockchain, including a swap, requires the payment of gas to the miners who validate and add the transaction to the ledger. If the gas limit is too low, the transaction may fail mid-process, or sit pending indefinitely, ultimately preventing the successful completion of the swap. Furthermore, during periods of high network congestion, the gas price can spike dramatically, leading to transaction failures if the initially set gas price is no longer competitive with the prevailing network conditions.

Finally, the code implementation itself may contain logical errors. I've encountered poorly coded smart contracts that don't properly account for edge cases, such as extremely small or large swaps, rounding errors in price calculations, or incorrect handling of fees. Such errors, though less common in verified contracts of popular exchanges, can nonetheless be a source of swap failure. The following example code snippets highlight areas where these problems can manifest.

**Example 1: Simplified Slippage Check**

```python
def calculate_output_amount(input_amount, pool_x, pool_y):
    k = pool_x * pool_y
    new_pool_x = pool_x + input_amount
    new_pool_y = k / new_pool_x
    output_amount = pool_y - new_pool_y
    return output_amount

def check_slippage(expected_output, actual_output, max_slippage):
    slippage = (expected_output - actual_output) / expected_output
    if slippage > max_slippage:
        return False
    return True


initial_pool_x = 10000 #token x
initial_pool_y = 5000 #token y
input_amount = 1000   # amount of token x to swap
max_slippage = 0.01 # 1% max slippage

expected_output = calculate_output_amount(input_amount, initial_pool_x, initial_pool_y)

# Assume a change in pool_y during execution, this emulates volatility
current_pool_x = initial_pool_x + input_amount
current_pool_y = 4000 # token y has decreased during process
actual_output = calculate_output_amount(0, current_pool_x, current_pool_y) #output is now calculated based on the current pool_y

if check_slippage(expected_output, actual_output, max_slippage):
    print("Swap Allowed.")
else:
    print("Slippage Too High. Transaction failed.")
```

This Python snippet illustrates a simplified slippage check. The `calculate_output_amount` function simulates the constant product formula. The `check_slippage` function then checks if the difference between the initial predicted output and the post-transaction output exceeds the defined `max_slippage`. The change in `current_pool_y` highlights how price volatility can trigger a slippage rejection. A contract must include this check (or a more robust variant) to ensure users aren't getting a significantly worse price than expected. Failing this check within the contract will prevent the swap from being completed and return tokens.

**Example 2: Basic Gas Calculation and Insufficient Gas**

```javascript
async function executeSwap(web3, contract, inputTokenAmount, gasLimit, gasPrice) {
    try {
    const estimatedGas = await contract.methods.swap(inputTokenAmount).estimateGas({from: web3.eth.defaultAccount}); // estimate gas

    if(estimatedGas > gasLimit){
        console.error("Insufficient Gas Limit Provided.")
        return false // transaction will fail
    }
        
        const tx = await contract.methods.swap(inputTokenAmount).send({
            from: web3.eth.defaultAccount,
            gas: gasLimit,
            gasPrice: gasPrice
        });
        console.log("Transaction Successful:", tx.transactionHash);
        return true;
    } catch(error) {
        console.error("Transaction Failed:", error);
        return false; // transaction will fail
    }
}
// Example Usage:
// Assuming contract, web3, gasLimit and gasPrice have been defined

const inputTokenAmount = 100;
const gasLimit = 100000; // too low for most swaps.
const gasPrice = 20000000000; // Gas price in wei
// If the estimatedGas is higher than gasLimit the swap will be rejected.
executeSwap(web3, contract, inputTokenAmount, gasLimit, gasPrice)
.then(result => {
    if(!result) {
        console.log("Swap did not execute")
    }
});

```

This JavaScript example using web3.js highlights the importance of sufficient gas. The code first estimates the gas required for the swap. If the provided `gasLimit` is lower than the `estimatedGas`, the transaction will fail. This often shows up as “out of gas” error messages, with the initial transaction either failing during execution or being rejected by the network due to insufficient resources.

**Example 3: Incorrect Token Amount Logic**

```solidity
// Solidity Code
pragma solidity ^0.8.0;

contract SimpleSwap {
    uint public tokenA_Balance = 100000;
    uint public tokenB_Balance = 50000;
    uint public constant K_CONSTANT = 5000000000; // tokenA_Balance * tokenB_Balance

    function swap(uint _amountA) public returns (uint) {
        
        //Incorrect implementation here:
        uint new_balanceA = tokenA_Balance + _amountA;
        uint new_balanceB = K_CONSTANT / new_balanceA; //integer division. Might cause inaccuracy. 
       
        require(new_balanceB <= tokenB_Balance, "Insufficient liquidity"); //Incorrect logic. Output is always less or equal

        uint outputAmount = tokenB_Balance - new_balanceB;
        tokenA_Balance = new_balanceA;
        tokenB_Balance = new_balanceB;
        return outputAmount;
    }
}

```

This Solidity example displays incorrect handling of token amounts in the swap function. The core issue is that in Solidity versions before 0.8.0, division results in integer truncation, potentially causing significant inaccuracies in `new_balanceB` calculation. This can result in either insufficient `outputAmount` or a failed transaction due to insufficient liquidity since it uses an incorrect `new_balanceB` value. Additionally, the check `new_balanceB <= tokenB_Balance`  will always be true as `new_balanceB` is a result of division, hence always lower than the initial `tokenB_Balance`, making the require statement faulty. Proper calculation requires a decimal number and correct comparison of the output based on the expected output according to the constant product formula. These types of implementation errors within smart contracts can be challenging to debug but are a significant source of failed swaps.

In summary, addressing the issue of failed sushi swaps requires careful attention to slippage settings, gas configurations, and code integrity. Thorough testing and understanding of the underlying AMM mathematics is paramount to preventing these issues. For more information I would advise referencing resources about Decentralized Finance (DeFi) and Ethereum Virtual Machine (EVM) development, specifically regarding Automated Market Makers, Gas Optimization, and Smart Contract Security. Researching slippage tolerance models and transaction handling will also be beneficial for gaining a solid understanding of swap mechanisms. Lastly, investigating the documentation for your specific swap contract is advised to fully understand the potential issues.
