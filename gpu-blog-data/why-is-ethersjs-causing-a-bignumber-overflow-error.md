---
title: "Why is Ethers.js causing a BigNumber overflow error when converting a string prop to a BigNumber in the UI?"
date: "2025-01-30"
id: "why-is-ethersjs-causing-a-bignumber-overflow-error"
---
The core reason Ethers.js throws a BigNumber overflow error during string-to-BigNumber conversion in a UI context is due to the intrinsic limitations of JavaScript's native number representation. JavaScript uses double-precision 64-bit floating-point numbers (IEEE 754), which cannot precisely represent all integers, especially those exceeding approximately 2<sup>53</sup>. When a string received from an API or entered via a user interface represents a number larger than what JavaScript can handle, Ethers.js, employing its `BigNumber` class to guarantee accurate arithmetic for blockchain interactions, flags this as an overflow.

I encountered this issue acutely when building a decentralized exchange (DEX) interface where users inputted large token amounts. The API returned token balances as strings to preserve precision, but these strings, when directly converted to Javascript numbers before passing them to Ethers.js, would lose fidelity. Ethers.js, by design, does not implicitly convert native JavaScript numbers to BigNumbers to prevent silent data loss. Instead, it expects string representations to initialize a BigNumber, allowing it to handle arbitrarily large integers. The error arises when the string itself represents a value outside the BigNumber's representable range or contains non-numeric characters, but more commonly, it's due to intermediate processing steps involving Javascript numbers that lose precision before the string is eventually passed to Ethers.

Specifically, before Ethers.js ever sees the string, the Javascript engine may already be attempting to parse and manipulate the string as a native number. Even if the string *appears* to represent an integer to the naked eye, JavaScript's internal representation, which can lead to unintended rounding or loss of precision, causes the Ethers.js `BigNumber` constructor to perceive this as an invalid input, and thus throws the overflow error. This is why passing a string like “123456789012345678901234567890” might succeed, but passing `parseFloat("123456789012345678901234567890")` as a BigNumber parameter will fail because `parseFloat` returns a native Javascript number.

To clarify this further, consider three practical scenarios, each demonstrating different facets of the problem and its solution:

**Code Example 1: The Problematic Path**

```javascript
// Assume amountString is '123456789012345678901234567890'
const amountString = '123456789012345678901234567890';

// Incorrect: Attempting to convert to native number first
const nativeNumber = parseFloat(amountString);

try {
    // This will throw a BigNumber error, as nativeNumber cannot accurately represent the original amountString.
  const bigNumber = ethers.BigNumber.from(nativeNumber);
    console.log("BigNumber Created:", bigNumber.toString());
} catch (error) {
  console.error("Error creating BigNumber:", error); // This error is expected
}
```

*Commentary*: This code fragment demonstrates the most common mistake. The `parseFloat()` function, even though the string appears to represent an integer, returns a JavaScript floating-point number which loses precision because it’s far beyond its accurate representable range. Ethers.js's `BigNumber.from()` method then receives this inaccurate representation which results in the overflow error. This is because it does not know the original string was intended to represent a much larger number.

**Code Example 2: The Correct Path**

```javascript
// Assume amountString is '123456789012345678901234567890'
const amountString = '123456789012345678901234567890';


try {
  // Correct: Passing the string directly to BigNumber.from()
  const bigNumber = ethers.BigNumber.from(amountString);
    console.log("BigNumber Created:", bigNumber.toString()); // This will work correctly

} catch (error) {
  console.error("Error creating BigNumber:", error); // This will not occur here
}
```

*Commentary*: This snippet demonstrates the correct approach. The original string is passed directly to `ethers.BigNumber.from()`. This method is specifically designed to parse string representations of large numbers, thereby preserving precision. Ethers.js handles the parsing internally, avoiding the limitations of Javascript's native number types. The resulting `bigNumber` variable can be used in transactions involving Ethereum.

**Code Example 3: Handling Formatted String Inputs**

```javascript
// Assume formattedAmountString is '1,234,567,890,123,456,789,012,345,678,90'
const formattedAmountString = '1,234,567,890,123,456,789,012,345,678,90';

// Cleaning the string from commas or other non-numeric characters
const cleanString = formattedAmountString.replace(/,/g, '');

try {
  // Now the string can be passed directly to BigNumber.from()
  const bigNumber = ethers.BigNumber.from(cleanString);
    console.log("BigNumber Created:", bigNumber.toString());

} catch (error) {
  console.error("Error creating BigNumber:", error);
}
```

*Commentary*: Often, UI inputs or APIs provide numeric strings that contain formatting characters, such as commas. Before passing these strings to Ethers.js, they must be cleaned. In this example, the commas are removed using the `replace` method. After removing these formatting elements, the remaining string can be safely parsed by Ethers.js, ensuring no overflow error occurs. The key is that you pass the pure string representation of the integer to the `BigNumber.from()` method.

In summary, the BigNumber overflow error commonly stems from JavaScript's imprecision with large numbers. The core solution is to *always* pass the string representation of the number directly to `ethers.BigNumber.from()` and ensure any formatted strings are cleaned before parsing. Avoid any intermediate conversion to JavaScript's native number representation. When diagnosing this error, meticulously examine each step of data processing, particularly the point where data transitions from a string representation into any form of Javascript numeric representation. Ensuring a string representation of the amount gets to Ethers.js as a string is the critical step.

For further study, consult documentation from the Ethers.js project regarding the `BigNumber` class and its usage. General guides on numeric representations in Javascript are also helpful for understanding the root of this problem. Resources covering Javascript's behavior around numeric precision and large numbers provide additional context for avoiding this particular type of error. Specifically, focus on resources that cover the limitations of `Number` objects and best practices when dealing with monetary values or other arbitrarily large integers.
