---
title: "Why am I getting a parsererror when writing inheritance solidity code on remix?"
date: "2024-12-14"
id: "why-am-i-getting-a-parsererror-when-writing-inheritance-solidity-code-on-remix"
---

well, parsererror in remix when dealing with inheritance? that's a classic, i've seen it pop up more times than i care to count. it usually boils down to a couple of common culprits, and i'll walk you through them based on my own experiences, it is never a single issue you need to check things one by one.

first off, lets talk about the basic setup of inheritance in solidity. remember, it's all about creating a hierarchical relationship between contracts. you have a base contract (the parent), and then you have child contracts that inherit properties from this base. the syntax is pretty simple on the surface using the `is` keyword. you define your child contract like this:

```solidity
contract childcontract is basecontract {
    // ... your child contract specific code here
}
```

the most common reason for a parsererror, in my experience, is a misnamed contract. seriously, it sounds simple, but it's an easy mistake. i once spent a good half-hour staring at my screen, convinced it was some deep compiler problem, only to find out i had a typo in the contract name in my `is` clause. double-check that both names, `childcontract` and `basecontract` in the example above, are spelled exactly the same as your contract definitions. solidity is case-sensitive, so a subtle difference will break it and will give you that frustrating error message.

another frequent issue is the order of contract definitions. solidity needs to know what a contract `basecontract` is *before* you try to inherit from it. i've tripped over this many times. you can have this:

```solidity
contract basecontract {
  // base contract code here
}

contract childcontract is basecontract {
   // child contract code here
}
```
the base contract declared before the child contract, this will compile ok. however, if you try to flip it like this, it will fail with a parsererror:

```solidity
contract childcontract is basecontract {
  // child contract code here
}

contract basecontract {
    // base contract code here
}
```

remix, like a good compiler, goes through your code from top to bottom. if it encounters the `childcontract` and hasn't seen the definition for `basecontract` yet, it gets confused and throws the parsererror. it’s all about declaring things in the correct order.

then we get to visibility. now, this is where things get a little more nuanced. when a child contract inherits from a base contract, it only has access to the `public` and `internal` members of the base contract. private members are, well, private. if your child contract tries to access a private member of the base contract, it won't work and could lead to a parsererror, or other compiler errors depending on the type of access. but that's more of a type or semantic error and not a parser one. but it is related to inheritance, and it is important to consider it.

```solidity
contract basecontract {
    uint256 internal myinternalvariable;
    uint256 public mypublicvariable;
    uint256 private myprivatevariable;

    constructor(){
        myinternalvariable = 10;
        mypublicvariable = 15;
        myprivatevariable = 20;
    }
}

contract childcontract is basecontract {
    function accessParent() public view returns (uint256, uint256) {
        // these work
       return (myinternalvariable, mypublicvariable);

       // this will produce an error because myprivatevariable is private
       //return (myinternalvariable, mypublicvariable, myprivatevariable);
    }
}
```

the example code above will compile and execute fine and the function `accessParent` will return the values from `myinternalvariable` and `mypublicvariable` fields, however, if you uncomment the line with `myprivatevariable` you would get a compilation error, but probably not a `parsererror` the compiler will tell you something like `member 'myprivatevariable' is not visible`.  it is not the case we are describing but the example highlights important points to keep in mind. it is not the error you described but it may help to pinpoint the real issue you may have.

now, let's talk about constructor arguments. this is an area where inheritance can get a bit tricky. if your base contract has a constructor with arguments, your child contract *must* either implement its own constructor that calls the parent's constructor with the necessary arguments *or* the base constructor should have no parameters. this is one of the first errors i had when i was starting to use solidity. i had this contract that was calling the base contract and i was getting errors everywhere. i did not even know about the way constructors are called. the most simple solution would be like in the example below.
```solidity
contract basecontract {
    uint256 public data;

    constructor(uint256 _initialData) {
        data = _initialData;
    }
}


contract childcontract is basecontract {
    constructor(uint256 _childInitialData, uint256 _baseInitialData) base(_baseInitialData) {
        // ...
    }
}
```

in this example, the child contract's constructor `childcontract(uint256 _childInitialData, uint256 _baseInitialData)` calls the base contract's constructor `base(_baseInitialData)` by the `base` function call in the definition, it sends the necessary argument `_baseInitialData`.

another common mistake that produces a parsererror is, while rare, is circular inheritance. that is, a contract cannot inherit from itself, nor can it create a loop where contracts inherit from each other in a circular fashion. like A inherit from B and B inherit from A this leads to infinite loops on the compiler, usually producing a `parsererror`. for example the code below will fail.
```solidity
contract contracta is contractb {

}

contract contractb is contracta {

}
```
now that is basic and easy to spot, but imagine a longer dependency chain. that is something you need to keep in mind. it is rarely the cause but is another reason for those pesky errors.

and sometimes the issue is not on your code, remix itself has its own limitations, and every now and then it might be remix itself having some issues. before tearing your hair out, try refreshing your browser or even using a different browser. i know, it sounds silly, but it's happened to me, and it is a pain to get tricked by something like that.

if all the points above are ok, then you may have a more profound issue, there are many situations that produce parsererrors, a syntax error, a simple typo you haven't found, or even a deep compiler bug. in this cases it is better to try another tool, like hardhat or forge, they have more robust and more helpful error messages. or you may ask a friend to review your code and ask for help, sometimes having a second pair of eyes is a blessing.

as for further resources, there are a lot of materials you can use to better understand these kinds of problems. i recommend the solidity documentation, it has a lot of useful examples, as well as the section on inheritance. the "mastering ethereum" book by andreas antonopoulos and gavin wood also has a great discussion on solidity inheritance and the underlying theory. for more advanced use cases, the papers from the ethereum foundation are a good place to start, they have some good research papers on solidity type systems, also, there are quite a few good papers on formal verification of smart contracts, those go very deep into the theory behind them.

so there you have it, my guide to debugging those parsererrors when dealing with inheritance in solidity on remix. remember to check those contract names, order of declarations, visibility, constructor arguments, and also for circular dependencies. and if all fails, try turning it off and on again (i had to add some programmer humor there). happy coding.
