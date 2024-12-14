---
title: "How to Chainlink - How to pass extra parameters to the Fullfil function?"
date: "2024-12-14"
id: "how-to-chainlink---how-to-pass-extra-parameters-to-the-fullfil-function"
---

well, i've seen this one a few times, feels like a classic. passing extra parameters to chainlink's `fulfill` function isn't exactly straightforward, and it's a point where a lot of folks, myself included way back when, have stumbled. it's a good example where the elegance of the chainlink design kinda hits a wall against the practical realities of needing to shoehorn more data into the response. let's break this down and get some actionable info.

first, the issue stems from how chainlink's oracle contracts interact with your consumer contract. your consumer requests data, the oracle fetches it, and the result is passed back to your `fulfill` function. the `fulfill` function's signature is fixed: it typically receives the request id, and the response. there's no direct place to just tack on extra data. this rigidity is intentional. it promotes predictable behavior, but makes us devs think a bit outside the box.

i remember my first encounter with this. i was working on a decentralized sports betting app (ambitious, i know, for my very first decentralized anything). we needed not just the score but also the team names, the game id, and some timestamp information passed back, all with the score. the oracle was scraping this data from a sports api, but the default chainlink setup just gave me the score alone to the `fulfill` function. i was pulling my hair out a little. i thought at that moment “surely i can do this the same way i pass parameters to my other javascript functions”. nope!

so, how do we get past this limitation? basically, we have to encode the extra parameters into the response we're giving the oracle to pass back. the `fulfill` function will then have to unpack that encoded data. there are a few ways to do this, and the best choice often depends on the complexity of the data you're pushing.

one popular approach is using abi encoding. essentially, you pack your various parameters into a single byte string using `abi.encode` in solidity. then, in your consumer contract's `fulfill` function, you unpack that byte string using `abi.decode`. here's a simplified example:

**oracle code** (pseudo-solidity, in reality, it depends on what your oracle implementation is)

```solidity
// pseudo-solidity, your oracle logic is more elaborate than this

function constructResponse(int256 _score, string memory _teamA, string memory _teamB, uint256 _gameId) public pure returns (bytes memory){
    return abi.encode(_score, _teamA, _teamB, _gameId);
}

```
so this is like a pseudo-oracle contract, where the oracle would use this function in his processing logic to prepare the response.

now, let's see how this data is handled in your consumer contract when the `fulfill` function is executed:

```solidity
// consumer contract
function fulfill(bytes32 requestId, bytes memory response) public {
    //unpacking the data
    (int256 score, string memory teamA, string memory teamB, uint256 gameId) = abi.decode(response, (int256, string, string, uint256));
     //use the data
    emit ScoreUpdate(requestId, score, teamA, teamB, gameId);
}

event ScoreUpdate(bytes32 requestId, int256 score, string teamA, string teamB, uint256 gameId);
```
this example takes an integer, two strings, and an unsigned integer, encodes them into bytes and unpacks them in the consumer contract to be used.

the key here is the `abi.encode` and `abi.decode`. they make sure the data is packed and unpacked correctly. you need to know the types of your data when you unpack it with `abi.decode`.

another option, if you're dealing with less complex data, is to encode the data into a simple string, perhaps using some delimiter like a comma or a pipe. then, in your `fulfill` function, you simply split that string into parts. here's an example:

```solidity
// oracle side (pseudo-solidity)
function constructStringResponse(int256 _score, string memory _teamA, string memory _teamB) public pure returns(string memory) {
    return string(abi.encodePacked(
        string.concat(
            Strings.toString(_score),
            ",",
            _teamA,
            ",",
            _teamB
        )));
}

```

and then the consumer contract part:

```solidity
//consumer contract
function fulfill(bytes32 requestId, bytes memory response) public {
    string memory stringResponse = string(response);
    string[] memory parts = split(stringResponse, ',');

    int256 score = Strings.parseInt(parts[0]);
    string memory teamA = parts[1];
    string memory teamB = parts[2];
    emit ScoreUpdate(requestId, score, teamA, teamB);
}

event ScoreUpdate(bytes32 requestId, int256 score, string teamA, string teamB);

//simple string split function
function split(string memory str, string memory separator) public pure returns (string[] memory) {
    //this is a minimal splitter not full implementation
    uint256 separatorLength = bytes(separator).length;
    uint256 strLength = bytes(str).length;
    uint256 partCount = 1;

    for (uint256 i = 0; i < strLength - (separatorLength - 1); i++) {
        if (string(slice(str, i, separatorLength)) == separator) {
             partCount++;
            i += separatorLength - 1;
        }
    }
    string[] memory parts = new string[](partCount);
    uint256 partStartIndex = 0;
    uint256 partIndex = 0;

    for (uint256 i = 0; i < strLength; i++) {
         if (i <= strLength - separatorLength && string(slice(str, i, separatorLength)) == separator) {
            parts[partIndex] = string(slice(str, partStartIndex, i - partStartIndex));
             partIndex++;
            partStartIndex = i + separatorLength;
             i+= separatorLength -1;
        }
    }
     parts[partIndex] = string(slice(str, partStartIndex, strLength - partStartIndex));
    return parts;
}

function slice(string memory str, uint256 startIndex, uint256 length) internal pure returns (bytes memory) {
    bytes memory strBytes = bytes(str);
    bytes memory result = new bytes(length);
    for (uint256 i = 0; i < length; i++) {
        result[i] = strBytes[startIndex + i];
    }
    return result;
}

```

in this example we are encoding the same values but now into a string delimited by comma and using a minimal split implementation to split the string when we receive the response. the simple `split` code provided here is not full proof and lacks error checking. this is only a minimal example, do not copy this split into your main production code.

which method is better? abi encoding is more robust and type safe, which reduces the risk of errors, and is probably a much better idea. the simple string encoding can be easier to implement for quick tests but is way less robust and less safe, so probably not suitable for more than playing around.

a final note of caution, be very careful about the size of the encoded data, especially with strings. the gas costs of dealing with very large byte arrays can become very expensive. it is good practice to keep your data payload as small as possible. also, in more complex scenarios, you can use more advanced encoding methods, such as data structures with mappings or even more complex custom encoding implementations.

also, it's important to note, and this catches some people off guard, that the data passed back through the oracle isn't exactly the same data that was on the external api. the chainlink oracle, when it picks data from an external api, it stores the data in a format that suits the chain, and that data is passed to the fulfill. so, if you are expecting the raw json data back, you are out of luck.

for a better understanding of chainlink, the documentation is a good place to start, although sometimes it's not enough, so i would suggest doing some deeper study in the architecture of smart contracts and how they interact with oracles. i would recommend the classic "mastering ethereum" by andreas antonopoulos and gavin wood, it gives a good theoretical view of the inner workings of smart contracts, along with solidity's documentation. chainlink's own documentation is great too, but sometimes it does lack some very specific edge cases, the stackoverflow community, where you are right now, has lots of great solutions and examples, so use that to your advantage.

and here is a little joke: why did the oracle break up with the smart contract? because they couldn't agree on the data types.

hopefully, this covers most of the ground on how to pass parameters to the chainlink fulfill function. remember to try simple examples and build up complexity incrementally. debugging smart contracts is no walk in the park, so it is better to be cautious and take it slowly.
