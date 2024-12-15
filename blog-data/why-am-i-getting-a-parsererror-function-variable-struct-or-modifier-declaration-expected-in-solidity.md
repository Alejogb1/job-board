---
title: "Why am I getting a ParserError: Function, variable, struct or modifier declaration expected in Solidity?"
date: "2024-12-15"
id: "why-am-i-getting-a-parsererror-function-variable-struct-or-modifier-declaration-expected-in-solidity"
---

alright, let's get into this parsererror. "function, variable, struct or modifier declaration expected" in solidity. i've seen this error a bunch of times, and it usually points to a structural issue in your solidity code, not necessarily a logical flaw. think of it as solidity telling you, "hey, i was expecting a building block, but i got something else entirely".

basically, solidity has very specific rules about where things should go and what should be inside what. it's like a really picky construction worker – everything has its place, and if something's out of order, they'll throw their hands up and say it's not going to work.

i remember this one time back when i was working on a decentralized lottery system, i kept getting this error and it was driving me nuts. i was trying to define a mapping inside a function, something i wouldn't even consider doing now, but, back then, my knowledge of solidity was still fairly new. spent a good few hours with a friend tracing every line, only to find i had misplaced the declaration. classic mistake.

so, what's likely happening is that you're trying to declare something (a function, variable, struct or modifier) in a place where it's not allowed. let's break down where these things *should* be located:

**1. contracts, libraries, and interfaces: the main containers**

solidity code lives inside one of these three structures. everything needs a home. it's like a house, you have walls, rooms etc, right?. functions, variables, structures, and modifiers all live inside these containers. if you try to define any of these outside these primary scopes you will get the error in question.

**2. function declarations: inside contracts, libraries, or interfaces**

functions are where the bulk of your executable code lives. they have a specific structure which includes the `function` keyword, the function's name, input parameters, modifiers (like `public`, `private`, `view`, `pure`), return types, and of course the function body (code block).

here's how a basic function looks:

```solidity
function myFunction(uint256 _input) public pure returns (uint256) {
    return _input * 2;
}
```

if you miss the `function` keyword, or declare it outside of the scope of a contract or library it will trigger this parser error.

**3. variable declarations: inside contracts, libraries, or structs**

state variables in solidity represent the persistent data that is stored in blockchain state. these need to be declared at the contract level not inside functions, also, they should have a type (`uint`, `address`, `string` etc) and access modifier.

here's a simple state variable:

```solidity
uint256 public myNumber;
```

local variables, which are only visible within the function where they’re defined should be declared inside a function scope. local variables do not require access modifiers like public or private.

```solidity
function addTwoNumbers(uint256 _a, uint256 _b) public pure returns (uint256) {
    uint256 sum = _a + _b;
    return sum;
}
```
if you try to define state variables within a function scope, or declare a variable with no type you will encounter this error, as variable declaration cannot exist outside the contract or library scope.

**4. struct declarations: inside contracts or libraries**

structs are custom data structures that you can define. they're super useful for grouping related data together. they need to be declared at contract level, not inside functions, and the members of the struct can be of various types.

here's what a struct definition looks like:
```solidity
struct MyStruct {
    uint256 id;
    address owner;
    string name;
}
```
if you attempt to define this struct inside a function, or outside a contract you will see this parser error.

**5. modifier declarations: inside contracts or libraries**

modifiers are like pre- or post-conditions for functions. they let you add checks or restrictions before/after your function executes. they are defined with the `modifier` keyword and they need to be declared at the same level of scope as state variables, they cannot live inside a function.

```solidity
modifier onlyOwner() {
    require(msg.sender == owner, "not owner");
    _;
}
```

if you happen to declare a modifier inside a function instead of at contract level you will trigger this error.

**common culprits and tips**

here’s a more refined list of things that usually trigger this error based on my experiences:

*   **missing keywords:** forgetting the `function`, `struct`, `modifier`, or variable type is a usual suspect. just one missing keyword makes the compiler confused, it doesn't really know if it's a typo or you want to do some non-existent operation. it's very fussy about this.
*   **incorrect placement:** declaring a variable inside a function where it should be a state variable, or defining state variables within a function.
*   **incomplete definitions:** not providing a return type for a function, or the body of the function, or variable type can cause this error.
*   **syntax errors:** typos in keywords or misplaced parenthesis, brackets or semicolons. you'd be surprised how much this happens.
*   **incorrect or missing visibility modifiers:** missing or invalid modifiers like `public`, `private`, `internal` for variables and functions. a common mistake.
*   **out-of-order statements:** for example, attempting to call a function before it's defined, or trying to access a variable before its declaration. solidity is very particular about the order things are declared. it's like telling a story out of order.

**debugging advice**

1.  **read the error carefully:** solidity error messages can actually tell you a lot, pay attention to the line number. look at the line number and surrounding lines. see what does not fit the expected structure.
2.  **check for missing keywords:** make sure every declaration has its `function`, `struct`, `modifier`, and data types.
3.  **trace declaration scope:** identify where declarations should live and ensure they are not misplaced.
4.  **break code down:** if your code is long, try simplifying it into smaller pieces that can be tested individually. the smaller the isolated code the faster you will find where the error is located.

**resources**

i'd suggest looking into these resources to understand the structure of solidity in-depth:

*   **the official solidity documentation:** this is always the go-to source for the most accurate and up-to-date information. everything from syntax to compiler details. it's like the bible for solidity developers.
*   **"mastering ethereum" by andreas antonopoulos and gavin wood:** it's a good in-depth look at the ethereum virtual machine and the smart contract model. not purely about solidity, but it helps you understand underlying workings.
*   **"solidity programming essentials" by ryan c. o'leary:** this book offers a good structured introduction to solidity programming, it walks you through many important concepts and has code examples.
*   **the chainlink documentation:** for when you dive into the specifics of using external data in solidity code. also, it provides many use cases of solidity code in a real world context.

a bit of advice based on my experience, debugging solidity code can be a real pain sometimes. you start seeing these errors and wondering if you have done something so terribly wrong in your life. it's like trying to solve a puzzle in the dark while also having to deal with the picky construction worker mentioned before. however, with practice, and patience, and a little bit of understanding it gets easier. remember when i had to spend several hours debugging a contract, only to realize i had declared something in the wrong place? i'm telling you, it happens to all of us. it's part of the journey. you will get the hang of it. you just need to keep coding and checking your code.

anyhow, i hope that helps and happy coding!
