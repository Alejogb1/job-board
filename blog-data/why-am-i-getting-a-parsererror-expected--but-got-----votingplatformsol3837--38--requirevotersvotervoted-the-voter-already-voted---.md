---
title: "Why am I getting a ParserError: Expected ';' but got ',' --> votingplatform.sol:38:37: | 38 | require(voters[voter].voted), "The voter already voted"); | ^ ?"
date: "2024-12-14"
id: "why-am-i-getting-a-parsererror-expected--but-got-----votingplatformsol3837--38--requirevotersvotervoted-the-voter-already-voted---"
---

hey there,

so, you're hitting a `parsererror: expected ';' but got ','` on line 38 of your `votingplatform.sol` contract, specifically within the `require` statement. i've seen this type of error more times than i care to count, and usually, it boils down to a syntax issue with how solidity expects expressions to be structured. let’s break this down step by step.

first off, that error message is pretty spot on. solidity, much like a strict compiler, is very particular about syntax. it's telling you that where it was expecting a semicolon `;`, it found a comma `,`. it's common to run into this type of issue when building smart contracts. i had one project back in '19 where the code looked almost identical to the one you have. the problem was actually due to the fact that I was using an external library with a different signature on a function, after much time trying to understand what was the problem. let's jump into your case.

let's look closely at your line again:

```solidity
require(voters[voter].voted), "The voter already voted");
```

the `require` statement in solidity has this basic form:

```solidity
require(condition, "error message");
```

it's very simple; it takes one boolean condition and an error message as a string. the comma separates these two arguments. you've written it as if the boolean condition was a list of conditions. in solidity, a single condition must evaluate to a boolean true or false. if it does not, solidity's compiler will throw an error.

in your case, the error arises because you’ve placed a comma immediately after the `voters[voter].voted` expression as if it is part of the same expression. solidity thinks you are trying to add additional conditions within the expression, when, in reality, the comma should separate the condition from the error message string.

here's what's likely happening in your case:

you're attempting to check if the `voter` has already voted using the boolean variable stored in `voters[voter].voted`. i bet your intent is to throw the error, `"the voter already voted"`, if the condition `voters[voter].voted` is `true`. however, since the condition must be evaluated to be `false` for the execution of the rest of the code after the `require` statement, your code is the incorrect approach.

the fix is to negate the condition. this will mean that if the `voters[voter].voted` is `true`, `require` will throw an error.

let's look at how to fix this. a quick change to the line will do the trick:

```solidity
require(!voters[voter].voted, "The voter already voted");
```

here, the `!` operator inverts the boolean value of `voters[voter].voted`. so, if `voters[voter].voted` is `true`, `!voters[voter].voted` is `false`, and the `require` statement will throw the specified error.

now, let's assume you want to include additional conditions in your require statement, like ensuring the voter exists before checking their vote status. you can do that using logical operators such as `&&` (and) or `||` (or) within the same condition. for example:

```solidity
require(voters[voter].exists && !voters[voter].voted, "voter is not allowed to vote");
```

in this example we are assuming the struct has an extra field `exists` to check if the voter exist in the first place, before checking if he has voted or not. this uses `&&` which will ensure the second condition to be evaluated only if the first condition is evaluated to `true` (short-circuit evaluation). it is like saying: *first check if the voter exists, if he does then check if he hasn't voted yet*

if you are dealing with complex conditions, make sure to use parentheses to make the order of operations crystal clear and prevent common logic errors.

```solidity
require((voters[voter].exists && !voters[voter].voted) || block.timestamp > votingEndTime , "voter is not allowed to vote");
```

here, the condition will evaluate if the voter exists and has not voted, or if the current block timestamp is greater than the voting end time. the parenthesis groups the conditions to make them clear.

for reference, there are several good resources on how to build smart contracts and handle common errors like this. i recommend looking into *mastering ethereum* by andreas antonopoulos and gavin wood; it's a bible for anyone working with solidity. also, *solidity programming essentials* by rishabh gupta is a good one to get into the nuts and bolts of the language. also, it is very important to keep updated with the latest version of solidity since language features may vary with each version.

in the end the error you got is pretty common. once you understand why it happened, you'll start spotting those syntax errors from a mile away. when i was starting out i made similar mistakes, my code looked like a patchwork of commas and semicolons until i got used to how the language worked. i remember once i was looking into one such an error for a whole day, and turns out there was a hidden character in my code... it was a zero-width space! computers, right?

let me know if that clears things up. good luck!
