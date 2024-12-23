---
title: "What causes the 'No matching declaration found' error in Solidity?"
date: "2024-12-23"
id: "what-causes-the-no-matching-declaration-found-error-in-solidity"
---

Okay, let's tackle this. I’ve definitely seen my share of "No matching declaration found" errors in Solidity, and they’re usually not as cryptic as they initially seem. In essence, this error arises when the Solidity compiler can't find a function, variable, event, or struct that you’re referencing within your contract. It’s a compilation-time error, which is actually a good thing because it means the problem is caught early, before you’ve deployed anything potentially problematic to the blockchain. It’s essentially the compiler telling you, "Hey, I have no clue what you're talking about!" Let’s break down the typical scenarios and how to resolve them.

My first significant encounter with this problem dates back to my early days building a decentralized exchange contract. We were rapidly iterating, and I made a classic mistake: I renamed a function in one file, but forgot to update all references to it in other parts of the project. The compiler immediately threw this error. From then on, I learned to be meticulous with my dependencies and naming conventions.

The core cause is always one of mismatched identifiers or incorrect scoping. Here's what I've observed to be the most common reasons, with a focus on real-world scenarios and solutions:

**1. Typographical Errors:** This one is straightforward, but surprisingly frequent. A simple typo in the name of a function or variable will absolutely trigger this error. I've seen this happen more often than I’d like to admit, especially when working under pressure. For example, if you declare a function `transferTokens()` but later call `transerTokens()`, you'll trigger the error. The fix is simply to carefully review the spelling and capitalization of identifiers and correct them.

**2. Incorrect Scoping and Visibility:** Solidity has scoping rules similar to other languages, and getting them wrong is a classic source of this error. If you declare a function or variable as `private` in one contract, you won't be able to access it from another contract (or even from the same contract in certain contexts). Similarly, if a function is only declared within a `library` and not declared as a `public` function, it will cause errors if called from another context. Consider this scenario:

```solidity
// Contract A
contract ContractA {
  uint256 private _myNumber = 10;

  function getNumber() public returns (uint256) {
      return _myNumber;
  }
}

// Contract B
contract ContractB {
  ContractA myContract;

    constructor(address contractAAddress) {
      myContract = ContractA(contractAAddress);
    }
    function attemptAccess() public returns(uint256) {
        // This line will throw a "No matching declaration found" error because _myNumber is private
        // return myContract._myNumber;

        // this is correct
        return myContract.getNumber();
    }
}
```
In the example above, attempting to directly access `myContract._myNumber` from `ContractB` results in the error because `_myNumber` is declared as private within `ContractA`. The solution is to use the public function `getNumber()` to access the number. You need to be aware of the scope of identifiers and access them only where they're visible.

**3. Incomplete Import Statements:** In larger projects, you’ll likely be splitting your contracts into multiple files. If you attempt to use a contract, library, struct, or enum that’s defined in another file, you must import that file. For example, if `ContractB` uses a type defined in `ContractA` but you don't include `import "./ContractA.sol";` at the top of `ContractB.sol`, you'll get the error. Similarly, if you're using a struct defined in a different file, like a struct `Order` in `Order.sol`, and you attempt to use `Order` in `ContractB`, the correct import statement is essential to ensure the compiler is aware of the declaration.

**4. Interface and Abstract Contract Mismatches:** When interacting with interfaces or abstract contracts, it’s crucial to have all the required functions implemented. Imagine you have an interface defined as follows:

```solidity
// IToken.sol
interface IToken {
  function transfer(address recipient, uint256 amount) external returns (bool);
  function balanceOf(address account) external view returns (uint256);
}
```
And then, you try to make a contract that implements this interface:
```solidity
// MyToken.sol
contract MyToken is IToken{
    mapping(address => uint256) _balances;

    function transfer(address recipient, uint256 amount) external override returns (bool) {
        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;
        return true;
    }
    // missing balanceOf function
}
```
This code snippet, while seeming complete, will cause a "No matching declaration found" error, because `MyToken` must implement *all* the functions specified in the `IToken` interface.  The solution is to implement the missing `balanceOf` function within `MyToken`:
```solidity
    function balanceOf(address account) external view override returns (uint256){
        return _balances[account];
    }
```
Failing to fully implement abstract contracts or interfaces, or having a typo in an interface's function name, will lead to the same error.

**5. Library Usage Errors:** Libraries in Solidity are often used for shared functionality. Libraries must be called using the correct pattern, specifically, functions in libraries called using `LibraryName.functionName()`. A typical cause of this error is using functions from a library incorrectly. See this scenario:
```solidity
//MathLib.sol
library MathLib {
    function add(uint256 a, uint256 b) public pure returns(uint256){
        return a + b;
    }
}
```
And then, a contract attempts to use it:
```solidity
// ContractC.sol
import "./MathLib.sol";

contract ContractC {
    using MathLib for uint256;

    function testAdd(uint256 a, uint256 b) public returns(uint256) {
        // this will fail
        // return add(a,b);
        // this will succeed
        return MathLib.add(a,b);
        //or this too, since we used using MathLib for uint256
        //return a.add(b);
    }
}
```
The solution is to correctly call the library function with `MathLib.add(a, b)` or, if you've used `using MathLib for uint256;`, you can use `a.add(b)`.

**6. Incorrect Contract Instantiation**: This error can arise when you are attempting to instantiate a contract using an incorrect address or when the address is pointing to a contract that doesn't match the declared type. For example, when you declare a variable such as `ContractA myContract;`, and then you try to set its value like `myContract = ContractA(someAddress);`, where `someAddress` points to a different type of contract or is not deployed. This mismatch will lead to the error. Ensure the address is correct and that it corresponds to the contract type you are instantiating.

Debugging this error requires a systematic approach. First, carefully double-check all identifiers (function names, variable names, types, etc.) for typos. Then, examine the scoping and visibility of the declared items. Verify that you have imported all the necessary files. Finally, if you’re dealing with interfaces or abstract contracts, ensure all required methods are implemented. It's often helpful to start at the line number identified in the error, work your way outwards, and analyze the context where the identifier is used.

For a deeper dive into these concepts, I highly recommend reading "Mastering Ethereum" by Andreas M. Antonopoulos and Gavin Wood. It provides a solid foundation on Solidity’s mechanisms. Also, the Solidity documentation itself is an invaluable resource for understanding compiler behavior and language nuances. Pay special attention to the sections on contracts, interfaces, libraries, and visibility. Another excellent paper that touches on these areas is “A Formal Semantics of Solidity”, which provides a strong, rigorous underpinning for the language’s execution model.

In my experience, tackling "No matching declaration found" errors isn't about blindly guessing. It's about applying a methodical, careful, and detailed approach to understanding how identifiers, scope, and imports interact within your Solidity projects.
