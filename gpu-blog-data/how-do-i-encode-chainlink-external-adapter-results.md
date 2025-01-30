---
title: "How do I encode Chainlink external adapter results into bytes32?"
date: "2025-01-30"
id: "how-do-i-encode-chainlink-external-adapter-results"
---
The challenge of encoding Chainlink external adapter results into `bytes32` arises from the fundamental mismatch between the diverse data types an external adapter might return (e.g., integers, floating-point numbers, strings, booleans, JSON objects) and the fixed 32-byte size of the `bytes32` data type in Solidity. Effective interaction between external adapters and Chainlink smart contracts mandates precise and consistent data encoding. My experience developing several Chainlink-integrated decentralized applications has shown that understanding and addressing this encoding challenge is crucial for robust oracle integration.

**Understanding the Problem and Its Constraints**

A Chainlink external adapter, essentially an off-chain service, fetches data from external APIs. This data, delivered to a Chainlink node, must ultimately be transmitted to a smart contract executing on the blockchain. Solidity contracts, being statically typed and operating within the EVM’s limitations, accept only specific data types. `bytes32` is frequently used due to its fixed size, simplifying storage and handling of various data, but requires careful transformation of potentially complex adapter responses. Direct casting is rarely suitable because it might truncate or improperly represent the external data.

The primary objective is to deterministically convert the external adapter's response into a `bytes32` representation such that the smart contract can reliably decode and interpret it. The specific encoding method depends entirely on the nature of the data received.

**Encoding Strategies**

I've found that three general encoding patterns cover most scenarios:

1.  **Integer Encoding:** When the external adapter returns an integer, we can directly convert it into a `bytes32`. Since Solidity stores integers with padding, this method is generally efficient for smaller integers that fit comfortably within the `bytes32` capacity. We can accomplish this with left padding to ensure we are operating with the correct number of bytes.

2.  **String Encoding:** Encoding strings requires more care. Simply converting a string to bytes will not guarantee that the resulting byte representation will always fit within 32 bytes. Furthermore, it will not allow us to determine which parts of the `bytes32` represent our string. I have learned that if string length exceeds the 32-byte limit, a common practice is to use a hash function (e.g., keccak256) that can return a 32-byte digest of the input string. This sacrifices the ability to retrieve the original string but is practical when the string itself is not needed within the contract. If the string is guaranteed to be shorter than 32 bytes, we can simply pad it with trailing zeros to reach the required length.

3.  **JSON Encoding and Parsing:** Often external adapters return data in JSON format containing multiple data points. In these cases, the desired value from the JSON response must be extracted and then encoded, often into either an integer, or string, before being converted to `bytes32`. The external adapter needs to do the JSON parsing and handle the extraction. An external adapter is most useful when it only returns the necessary data to be used in the chainlink contract.

**Code Examples**

The following code examples demonstrate how to implement each of these approaches in a Node.js based external adapter.

**Example 1: Integer Encoding**

```javascript
const ethers = require('ethers');

function encodeInteger(integerValue) {
  const hexRepresentation = ethers.BigNumber.from(integerValue).toHexString();
  const paddedHex = ethers.utils.hexZeroPad(hexRepresentation, 32);
  return paddedHex;
}

const integer = 12345;
const encodedInteger = encodeInteger(integer);
console.log("Integer:", integer);
console.log("Encoded Integer:", encodedInteger);
```
*   This code takes an integer value, `integerValue`, and transforms it to a hex string using the `ethers.BigNumber` library, ensuring that it can handle large integers.
*   Then it uses `ethers.utils.hexZeroPad` to pad the hexadecimal representation to a total length of 32 bytes (64 hex characters), which is the standard length for a `bytes32` representation.
*   The resulting string `paddedHex` is the desired `bytes32` encoding of the integer.

**Example 2: String Encoding**

```javascript
const ethers = require('ethers');

function encodeString(stringValue) {
  if (stringValue.length <= 32) {
      const hexString = ethers.utils.hexlify(stringValue);
      return ethers.utils.hexZeroPad(hexString, 32)
  } else {
      const hashedString = ethers.utils.keccak256(ethers.utils.toUtf8Bytes(stringValue));
      return hashedString;
  }
}

const shortString = "hello";
const longString = "This is a string longer than 32 bytes, which requires hashing.";
const encodedShortString = encodeString(shortString);
const encodedLongString = encodeString(longString);
console.log("Short String:", shortString);
console.log("Encoded Short String:", encodedShortString);
console.log("Long String:", longString);
console.log("Encoded Long String (hashed):", encodedLongString);
```
*   The `encodeString` function first checks if the input string's length is less than or equal to 32 bytes. If so, it converts the string to its hexadecimal representation using `ethers.utils.hexlify`, then pads it to 32 bytes using `hexZeroPad`, as in the integer example.
*   If the string is longer than 32 bytes, the function uses `ethers.utils.keccak256` to generate a 32-byte hash from its UTF-8 encoded byte representation.
*   The output demonstrates two strings of different lengths, showcasing both cases: direct encoding with padding and the hash digest approach.

**Example 3: JSON Parsing and Integer Encoding**

```javascript
const ethers = require('ethers');
function encodeJsonInteger(jsonResponse, jsonKey) {
  const parsedJSON = JSON.parse(jsonResponse);
  const extractedValue = parsedJSON[jsonKey];

    if (extractedValue != undefined && typeof extractedValue === 'number'){
        const hexRepresentation = ethers.BigNumber.from(extractedValue).toHexString();
        const paddedHex = ethers.utils.hexZeroPad(hexRepresentation, 32);
        return paddedHex;
    }
    else {
        return null;
    }

}
const jsonResponse = '{"temperature": 25, "humidity": 60}';
const jsonKey = "temperature";
const encodedExtractedInteger = encodeJsonInteger(jsonResponse, jsonKey);
console.log("JSON Response:", jsonResponse);
console.log("Encoded Extracted Integer:", encodedExtractedInteger);
```

*   The `encodeJsonInteger` function takes a JSON string as input, parses the JSON, and extracts a specific numerical value using a given key, `jsonKey`. It handles cases where the key doesn't exist.
*   It then uses the same integer encoding logic to transform the extracted integer value into a 32-byte representation using  `ethers.BigNumber` and `ethers.utils.hexZeroPad`.
*   If the extracted value is not a number it returns null.
*   The output displays the original JSON, and the `bytes32` representing the temperature value after processing.

**Decoding Considerations**

While these examples focus on encoding, it's equally vital to implement the corresponding decoding process on the smart contract side. The chosen encoding method dictates how data is interpreted back to its original format. For integer encodings, the same library can perform the inverse operation in the smart contract using methods such as `uint256(bytes32 value)`. For hashed strings, as mentioned previously, the original string value is not recoverable; the contract can only verify that a given string, when hashed, matches the stored hash. For more complex scenarios, custom logic is often needed within the contract.

**Resource Recommendations**

Several resources can aid in the development of Chainlink external adapters and smart contracts. Documentation for Ethereum libraries, such as Ethers.js and Web3.js are paramount. Furthermore, exploring examples of Chainlink's official documentation and community forums provides excellent guidance on best practices. Understanding the Solidity documentation is also essential for interpreting `bytes32` data. By combining these resources, one can confidently develop robust data pipelines.

In conclusion, encoding external adapter responses into `bytes32` for use in Chainlink smart contracts requires careful consideration of the data types and desired use cases. Understanding the three patterns mentioned above—integer encoding, string encoding (including hashing when needed), and JSON parsing and specific value encoding—along with rigorous testing, helps ensure that your oracle integrations are reliable and secure. The use of libraries such as Ethers.js greatly simplifies the process. Remember that this process is not complete until a reliable decoding strategy is implemented on the smart contract, completing the data transfer pipeline.
