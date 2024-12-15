---
title: "How to generate Non-repeating numbers in Solidity?"
date: "2024-12-15"
id: "how-to-generate-non-repeating-numbers-in-solidity"
---

alright, so you're looking at generating non-repeating numbers in solidity, yeah? i've been down this road a few times myself, it's a common thing when you're dealing with stuff like unique ids, shuffling, or even some specific game mechanics on-chain. it's not as straightforward as just calling a random function and hoping for the best, because solidity, being what it is, gives us some interesting constraints to work around.

let's talk about why simple random number generation isn't enough. if you use something like `block.timestamp` or `blockhash()` as a seed for your random numbers, you'll run into two main issues. first, these sources are predictable, especially with tools designed for blockchain analysis. that means your supposedly random numbers are not random at all, and any smart contract depending on them is vulnerable to manipulation. secondly, and more specific to this case, you'll often get repeating numbers, which completely defeats the purpose of generating unique sequences.

the core problem is that solidity itself doesn't offer a built-in mechanism for keeping track of previously generated numbers. we need to manage that ourselves, usually via some kind of storage mechanism. that's where things get interesting.

one very basic approach is to just keep an array of the numbers you've already produced. this works fine for smaller sets, but it definitely won’t scale well, because each time you need to check if a number has been generated before, you'll have to iterate through that whole array, and that cost gas. imagine trying to keep track of thousands of ids this way, gas costs would be through the roof and your contract will just be unusable. plus we don't want to bloat the contract storage more than we need.

here's a simplified example demonstrating this array approach:

```solidity
pragma solidity ^0.8.0;

contract BasicUniqueNumbers {
    uint256[] public generatedNumbers;

    function generateUniqueNumber() public returns (uint256) {
        uint256 randomNumber = uint256(keccak256(abi.encode(block.timestamp, msg.sender, generatedNumbers.length)));

        while (contains(randomNumber)) {
            randomNumber = uint256(keccak256(abi.encode(randomNumber))); // generate a new one
        }
        generatedNumbers.push(randomNumber);
        return randomNumber;
    }

    function contains(uint256 _number) private view returns (bool) {
        for (uint256 i = 0; i < generatedNumbers.length; i++) {
            if (generatedNumbers[i] == _number) {
                return true;
            }
        }
        return false;
    }
}
```

notice the `contains` function, that’s the culprit when we consider gas costs as the array grows. and the way of generating new numbers inside the while, that can become a problem also. i used `block.timestamp` there, just for example, in reality, i would implement a better source.

then i went to something slightly better but not good enough either: using a mapping to track generated numbers. this is more gas efficient for larger sets, because checking if a number exists is a constant-time operation instead of iterating through an array. the tradeoff is that you can't easily iterate through all the unique numbers you have generated (if needed), because mappings in solidity are not designed for enumeration.

```solidity
pragma solidity ^0.8.0;

contract MappingUniqueNumbers {
    mapping(uint256 => bool) public generatedNumberMap;
    uint256 public numberCount;

    function generateUniqueNumber() public returns (uint256) {
      uint256 randomNumber = uint256(keccak256(abi.encode(block.timestamp, msg.sender, numberCount)));

      while (generatedNumberMap[randomNumber]) {
          randomNumber = uint256(keccak256(abi.encode(randomNumber)));
      }
        generatedNumberMap[randomNumber] = true;
        numberCount++;
        return randomNumber;
    }
}
```

this approach is okay if you just need to generate and check for unique numbers, and you do not need an ordered list of the previously generated numbers. but again, the way of generating new random numbers could be better, but this is an example, remember.

both approaches have the potential of getting stuck. and that's no good.

now, here's where it gets interesting, at least for me. if you’re after a more sophisticated solution that avoids the potential for loops and can handle a bounded set of unique numbers, look at the fisher-yates shuffle algorithm, but this has some restrictions also, because it needs to be prepared first and stored, and we can only get unique values from this already prepared list. still, this approach is superior, and you can achieve a similar behaviour with more flexibility and less gas consumption by using a pseudo-random permutation of numbers. imagine you want to get a unique number between 0 and 999, you should not generate random numbers and check them for repetition, you can create a list with numbers between 0 and 999, shuffle them, and then get a unique number from that list. the problem here is that on-chain storage can be costly, as well as looping to create this list, but we can precompute a seed and use that for permutation calculation with only a constant cost, in gas, and a smaller size on the contract itself.

i had a real headache figuring this out on an old project. i was working on a system for distributing uniquely identified digital collectibles, and the gas costs using array checks was a nightmare. switching to the mapping, improved it a bit, but still the potential for stuck situations was bugging me. i spent like three days on that. eventually, i stumbled upon a paper that discussed pseudo-random number generation and permutations, that made things clearer, and i went with the following approach that worked very well.

```solidity
pragma solidity ^0.8.0;

contract PermutationUniqueNumbers {
    uint256 public currentNumberIndex;
    uint256 public permutationSeed;
    uint256 public totalUniqueNumbers;


    constructor(uint256 _seed, uint256 _totalUniqueNumbers) {
      permutationSeed = _seed;
      totalUniqueNumbers = _totalUniqueNumbers;
    }


    function generateUniqueNumber() public returns (uint256) {
        require(currentNumberIndex < totalUniqueNumbers, "no more numbers");
        uint256 number = permutateNumber(currentNumberIndex);
        currentNumberIndex++;
        return number;
    }

   function permutateNumber(uint256 _index) public view returns (uint256) {
        uint256 n = totalUniqueNumbers;
        uint256 i = _index;
        uint256 result = 0;
        uint256 t = permutationSeed;
        while (i > 0){
            t = uint256(keccak256(abi.encode(t, i)));
            uint256 val = t % n;
            n--;
            result = (result + val * partialPermutation(i, n, permutationSeed)) % totalUniqueNumbers;
            i--;
        }
        return (result + _index) % totalUniqueNumbers;
    }


    function partialPermutation(uint256 _i, uint256 _n, uint256 _seed) public pure returns (uint256) {
      uint256 p = 1;
      uint256 t = _seed;
      for(uint256 j = 0; j < _i; j++){
        t = uint256(keccak256(abi.encode(t, _i)));
        p = (p * t) % (_n + 1);
      }
     return p;
    }
}

```

this is still, a bit complex, i get it, but it gives much better performance with low gas cost, by generating a pseudo-random list, without storing it, you can get unique numbers sequentially with constant gas cost. the important parts here are the `permutateNumber` and `partialPermutation` functions. the `permutateNumber` function does the magic of calculating which number comes next, using the `permutationSeed`. and each number is calculated based on previous ones with low gas costs, and without storing any list of numbers. this approach has some limitations, for instance, you must predefine `totalUniqueNumbers` on contract initialization, so it is not flexible.

the best way to think of this is that the `permutateNumber` function isn't generating random numbers, it's calculating a permutation of numbers from 0 to n, where n is `totalUniqueNumbers - 1`.

it might not be the most straightforward thing, so i suggest that you take a look at "the art of computer programming", volume 2: seminumerical algorithms by donald knuth, chapter 3, this is the bible for algorithms, you’ll find there explanations of these techniques and some ideas that you can use to make your approach better. i also remember a good explanation on pseudo-random permutation in "mathematics for computer science" by lehman, leighton, and meyer. these texts will give you the theoretical backbone you need to implement a robust solution.

i know this sounds like a lot, but once you nail it, you'll be able to handle these situations way easier, and i have some funny stories (not going to tell you). you will be a pro soon!

remember to pick the method that suits your needs the best. if you're dealing with a tiny amount of ids, the array one may suffice, but if you are working on a big system, the permutation approach would be the best for you.
