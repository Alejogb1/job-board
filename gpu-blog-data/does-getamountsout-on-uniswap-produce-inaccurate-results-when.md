---
title: "Does getAmountsOut() on Uniswap produce inaccurate results when running on a Hardhat mainnet fork?"
date: "2025-01-30"
id: "does-getamountsout-on-uniswap-produce-inaccurate-results-when"
---
In my experience working with decentralized exchanges (DEXs) and specifically Uniswap's periphery contracts, inaccuracies in `getAmountsOut()` calculations on a Hardhat mainnet fork stem primarily from discrepancies between the forked state and the live network's state, particularly concerning token reserves and pricing.  This is not inherent to the `getAmountsOut()` function itself, but rather a consequence of the limitations of forking.  A perfectly synchronized fork is exceptionally difficult to achieve and maintain.


**1. Explanation of Inaccuracies:**

The `getAmountsOut()` function, residing within Uniswap's Router contract, leverages the current pool reserves to compute the output amounts for a given input amount. These calculations rely on the constant product formula (x * y = k) for pools using the classic constant product formula, and a different mathematical model for other pool types.  A Hardhat mainnet fork captures a snapshot of the blockchain state at a particular block. However, this snapshot is static.  The live network, conversely, is continuously updating; transactions are occurring, impacting pool reserves and consequently the exchange rate between tokens.

This temporal discrepancy is the root cause of inaccuracies.  The fork's reserve values might deviate from live values, even in short time spans due to high trading volume on the mainnet.  This leads to a mismatch in the `getAmountsOut()` output compared to the result obtained by executing the same function call on the live network.  Other factors contributing to this divergence include:

* **Block Number Differences:**  The fork might not perfectly replicate the precise block number and associated state at which your test is run on the mainnet.
* **Reentrancy:** If the contract interacts with other contracts on the mainnet, reentrancy vulnerabilities could lead to state differences not captured by the fork.
* **External Oracles:** If the pricing relies on external price oracles, the oracle values on the fork may not be updated to match the mainnet's values.
* **Gas Price Differences:** Although unlikely to directly impact calculation results, extreme differences in gas prices between the fork and the mainnet could lead to variations in transaction ordering, again impacting the final pool state.


**2. Code Examples with Commentary:**

The following examples illustrate potential scenarios and highlight how to mitigate these discrepancies.  I've focused on the classic constant product pools for simplicity, but the principles extend to other pool types.

**Example 1:  Basic `getAmountsOut()` call on a forked network:**

```javascript
const { ethers } = require("hardhat");

async function getAmountsOutFork() {
  const [deployer] = await ethers.getSigners();
  const router = await ethers.getContractAt("IUniswapV2Router02", "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"); // Replace with your actual Router address

  const amountIn = ethers.utils.parseEther("1"); // 1 ETH
  const path = ["0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "0xdAC17F958D2ee523a2206206994597C13D831ec7"]; // WETH to USDC (replace with your tokens)

  try {
    const amountsOut = await router.getAmountsOut(amountIn, path);
    console.log("Amounts Out (Fork):", amountsOut);
    return amountsOut;
  } catch (error) {
    console.error("Error getting amounts out:", error);
    return null;
  }
}


getAmountsOutFork()
  .then((amounts) => {
    if(amounts) console.log("USDC received (Fork):", ethers.utils.formatEther(amounts[1]));
  })
  .catch((error) => console.error("Error:", error));
```

This code directly calls `getAmountsOut()` on a forked mainnet. The accuracy depends entirely on the synchronization of the fork.


**Example 2:  Fetching reserves directly and performing manual calculation:**

```javascript
const { ethers } = require("hardhat");

async function getAmountsOutManual() {
  const [deployer] = await ethers.getSigners();
  const pairAddress = await getPairAddress(
      "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", // WETH
      "0xdAC17F958D2ee523a2206206994597C13D831ec7" // USDC
  ); // Requires a function 'getPairAddress' to fetch the pair address from the factory contract
  const pair = await ethers.getContractAt("IUniswapV2Pair", pairAddress);

  const reserves = await pair.getReserves();
  const amountIn = ethers.utils.parseEther("1");
  const amountOut = calculateAmountOut(amountIn, reserves.reserve0, reserves.reserve1); // Custom function to calculate using constant product

  console.log("Amounts Out (Manual):", amountOut);
  return amountOut;
}

//Implementation of calculateAmountOut would be provided here (omitted for brevity)

getAmountsOutManual()
  .then((amount) => console.log("USDC received (Manual):", ethers.utils.formatEther(amount)))
  .catch((error) => console.error("Error:", error));


```

This example fetches the reserves directly from the pair contract and performs the constant product calculation manually.  This reduces reliance on the router's internal state, but still hinges on the accuracy of the forked reserves.


**Example 3:  Using a testnet instead of a mainnet fork:**

```javascript
// ... (Similar setup as before but using a testnet provider) ...
const provider = new ethers.providers.JsonRpcProvider("https://goerli.infura.io/v3/YOUR_INFURA_PROJECT_ID"); // Replace with your testnet provider

// ... rest of the code is similar to Example 1, but using the testnet provider ...
```

This approach employs a testnet (Goerli in this example), providing a more stable and reliable environment for testing than a mainnet fork, though the values will likely not match mainnet.

**3. Resource Recommendations:**

Consult the official Uniswap documentation. Review the source code of the Uniswap Router contract directly. Explore advanced Hardhat documentation focusing on forking techniques and limitations.  Study the Solidity documentation for precise understanding of the data types and functions used in the Uniswap contracts.  Finally, gain a strong understanding of the mathematical formulas used by various Uniswap pool types (constant product, constant sum, etc.).


In conclusion, while `getAmountsOut()` is not inherently flawed, using it on a Hardhat mainnet fork introduces inaccuracies due to inherent discrepancies between the forked state and live network state.  The best practices involve minimizing reliance on the router's internal calculations, fetching reserves directly, using a testnet for integration testing, or accepting some degree of inaccuracy inherent in the process.  Understanding these limitations and implementing appropriate mitigation strategies are vital for reliable smart contract development and testing within the decentralized finance ecosystem.
