---
title: "How am I Unable to put an integer variable with a time variable in solidity?"
date: "2024-12-15"
id: "how-am-i-unable-to-put-an-integer-variable-with-a-time-variable-in-solidity"
---

here's the deal, you're bumping into a classic solidity type mismatch problem, and it's pretty common, so don't sweat it. it's all about how solidity handles data types, and unlike some other languages, it's very strict about conversions. you're trying to mix an integer, which could be representing anything from a counter to a price, with a time value which is actually a specific type, `uint` representing seconds since the unix epoch. the compiler will scream at you because it needs explicit instructions on how to interpret and handle the different data representations.

i've been through this myself more times than i care to count, usually when i'm rushing a smart contract late at night, fuelled by too much coffee and the burning desire to get my code deployed, and forgetting the fundamentals of the solidity type system. i remember one particular incident, back when i was working on a simple auction contract, i wanted to store the bid amount with a timestamp in a mapping. i declared the mapping like this

```solidity
mapping(uint => uint) public bids; // WRONG!!
```

i thought it was ok because i was using `uint` for both the bid amount and for the timestamp (which is also `uint` type in solidity), but it turned out the timestamp needed to be handled as a block timestamp or as an explicit time value, which required further processing. this led to all kinds of weird errors and confusing behaviours. after hours of debugging, using remix and console.log statements everywhere, i realized i needed to use a struct to hold this information together to maintain data integrity and make logical sense.

the main point is: a time variable in solidity, specifically `block.timestamp`, isn't just a number. it's a specific representation of time as a unix timestamp, which is a `uint` (unsigned integer). when you try to directly combine a regular integer variable with it, solidity says, "hold up, i don't know what you want me to do with these two different values that represent two distinct things". it isn't about the *numbers* themselves, it’s about what the numbers *represent*.

let me explain further, you can not directly combine them unless you perform explicit operations and explicit type conversions. for instance, if you want to store both an integer and a timestamp associated with it, you can't just put them together as if they are both generic numbers. you need to think about how you want to represent the combination. here is a practical use case and what could be an example of what you are trying to do:

let's say you want to store some user data with the last updated timestamp. if you tried directly using `mapping(uint => uint)` like i tried before with my auction contract, you would get a similar headache. so, you could do the following, you need a struct to store both integer and the timestamp information like this:

```solidity
struct UserData {
    uint value;
    uint lastUpdated;
}

mapping(address => UserData) public userData;

function updateUserValue(uint _value) public {
    userData[msg.sender] = UserData({
      value: _value,
      lastUpdated: block.timestamp
    });
}

function getUserData(address _user) public view returns (uint, uint) {
    return (userData[_user].value, userData[_user].lastUpdated);
}
```

in the example above, `UserData` is a struct that clearly defines the relationship between the `value` and `lastUpdated` fields. now you store them within the same mapping. when you call `updateUserValue`, `block.timestamp` is captured correctly and it's related to the user's address. when fetching the data you can get both the integer and the timestamp together correctly. remember, the `block.timestamp` is already of type `uint`, but you are dealing with it in a manner where it is recognized as a timestamp and not just a plain number.

if, instead, you wanted to calculate an expiry time, for example, you might want to add a time duration to the current timestamp. let's say you have a duration represented as an integer in seconds and you want to calculate the timestamp of the expiration, you'd do something like this:

```solidity
uint public expirationTime;

function setExpiry(uint _durationInSeconds) public {
  expirationTime = block.timestamp + _durationInSeconds;
}

function checkIsExpired() public view returns (bool) {
    return block.timestamp > expirationTime;
}
```

here, you are adding a regular `uint` value representing duration to another `uint` value representing the current timestamp. the crucial part is that both values are treated as the same type (`uint`), they are simply integers, and because both are `uint` you can perform math operations on them. the time unit is in seconds. for example, if `_durationInSeconds` is `300`, then you'd be setting the `expirationTime` to 300 seconds in the future.

another common scenario i encounter all the time, is needing to compare timestamps to check if an action has already occurred. let’s see this example:

```solidity
uint public lastActionTime;

function triggerAction() public {
    require(block.timestamp > lastActionTime + 300, "Action can't be triggered yet."); // minimum 5 minutes
    lastActionTime = block.timestamp;
    //execute the action here
}

function getLastActionTime() public view returns(uint) {
    return lastActionTime;
}
```

in the example, you’re checking if at least 300 seconds (5 minutes) have passed since the last action. it's a very useful pattern for cooldowns, rate limits, etc. the key here is that you're not mixing data types incorrectly. you're dealing with `uint` in relation to time explicitly. and as an extra tip, always explicitly check the ranges for your values to avoid under or overflows especially with time and duration.

for diving deep into solidity’s data types, i would recommend reading “mastering ethereum” by andreas antonopoulos. it is not exactly a deep dive on solidity specific issues like this one, however it covers all the nuances of the evm in a clear manner that will aid your understanding and you will be able to extrapolate the answer to this kind of issue, it is a bible for evm based development, a very high level and a must for any evm developer. also, “solidity programming essentials” by rishabh srivastava gives a very clear understanding on the type system and the logic behind its design, both books are extremely useful in the long run for becoming proficient with solidity. also the documentation on [solidity](https://docs.soliditylang.org/en/v0.8.23/) is very good and well explained, always consult the official docs as your first resource.

also, a little pro tip, use block.number if you are dealing with relative time operations. usually people tend to use `block.timestamp` to check how much time has passed since the last event or block, but that may not be the case all the time. if you need to have absolute time with respect to the blockchain and not some external source (the node), then `block.number` (the block number) is a more accurate measure of time. the block number will increment in time even if there isn't a transaction. the timestamp might be altered by the miners slightly or if the node is misconfigured, it is an external source, but block number is always increasing every 12 seconds (or whatever the consensus time of the blockchain is) so is a more accurate measure of time if you need absolute time and not actual time. also, using `block.number` is a more predictable measure of time as block times vary in real life, but blocks will always increase so your calculations will be more reliable.

i hope that clears things up. dealing with time in solidity can be a pain, but once you understand the basics of data types and how the evm treats timestamps, you will find it not as cumbersome, just pay attention and remember what you are mixing and their types and be explicit about it. it’s not rocket science, but it does require paying attention to detail and a little patience. (i once spent three days trying to figure out why my contract was behaving weirdly, it turned out i had used `int` instead of `uint` for block numbers, the pain was real!)
