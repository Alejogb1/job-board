---
title: "How to test a Solidity function returning a struct using Chai in Hardhat?"
date: "2024-12-23"
id: "how-to-test-a-solidity-function-returning-a-struct-using-chai-in-hardhat"
---

Alright, let's tackle this one. It's not uncommon to find yourself needing to thoroughly test Solidity functions that return structs, especially when dealing with complex data structures in your smart contracts. I've certainly been in that position countless times, often having to unravel some rather intricate test scenarios. The core of the challenge, as I see it, isn't simply *calling* the function; it's about effectively asserting that the returned struct contains the values you expect after your logic has executed.

Using Chai within Hardhat's testing environment provides a powerful and relatively straightforward mechanism to achieve this. The typical stumbling block for many developers tends to be how to access the individual fields within the returned struct for assertion purposes, since the returned value is essentially a complex object. This is where a clear understanding of how Solidity and Javascript (the language used within Hardhat's test files) interact is critical.

Let's dive into the practicalities. First and foremost, you must ensure your Hardhat test environment is set up correctly, and that your smart contract is compiled. I'll assume these foundational steps have been completed, and we'll focus on the testing specifics.

The key idea, and where I've seen teams go wrong, is understanding that Hardhat and ethers.js, which is the library commonly used in conjunction with Hardhat, will often return structs as Javascript objects with numerical indices by default. While this *can* be used, it's far less readable and more prone to errors than accessing struct fields by name. Therefore, our first step is to ensure we’re handling the returned struct gracefully by destructuring the data based on field names. This approach provides clarity, reduces the risk of misinterpreting the returned values, and ensures your test code remains maintainable.

Let’s illustrate this with an example. Imagine a simple Solidity contract with a function that returns a struct representing a user’s data:

```solidity
pragma solidity ^0.8.0;

contract UserData {

    struct User {
        uint256 id;
        string name;
        uint256 balance;
    }

    mapping(uint256 => User) public users;

    function createUser(uint256 _id, string memory _name, uint256 _balance) public {
        users[_id] = User(_id, _name, _balance);
    }

    function getUser(uint256 _id) public view returns (User memory) {
        return users[_id];
    }
}

```

Here is the corresponding test setup using Hardhat and Chai:

```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("UserData", function () {
    it("Should return the correct user data", async function () {
      const UserData = await ethers.getContractFactory("UserData");
      const userDataContract = await UserData.deploy();
      await userDataContract.deployed();

      const userId = 1;
      const userName = "Alice";
      const userBalance = 100;

      await userDataContract.createUser(userId, userName, userBalance);

      const returnedUser = await userDataContract.getUser(userId);

      expect(returnedUser.id).to.equal(userId);
      expect(returnedUser.name).to.equal(userName);
      expect(returnedUser.balance).to.equal(userBalance);

    });
});

```

Notice how I can directly access `.id`, `.name`, and `.balance` within the returned `returnedUser` variable? That’s because ethers.js, in this case, when interacting with the smart contract function will return the struct in such a way that we can access its properties directly by name using dot notation. This is the most common and easiest way to access data returned from smart contracts.

But there are scenarios, especially in more complex or nested structures, where you might have a more indirect approach. Here is a slightly more complex situation:

```solidity
pragma solidity ^0.8.0;

contract ComplexData {
    struct AddressDetails {
       string street;
       string city;
    }

    struct Person {
       uint256 id;
       string name;
       AddressDetails address;
    }

    mapping(uint256 => Person) public people;

    function createPerson(uint256 _id, string memory _name, string memory _street, string memory _city) public {
      people[_id] = Person(_id, _name, AddressDetails(_street, _city));
    }

    function getPerson(uint256 _id) public view returns(Person memory) {
      return people[_id];
    }

}

```

And the following is an example of how to test the above complex struct, which includes a nested struct:

```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ComplexData", function () {
  it("Should return the correct complex person data", async function () {
    const ComplexData = await ethers.getContractFactory("ComplexData");
    const complexDataContract = await ComplexData.deploy();
    await complexDataContract.deployed();

    const personId = 1;
    const personName = "Bob";
    const street = "Main Street";
    const city = "Anytown";

    await complexDataContract.createPerson(personId, personName, street, city);
    const returnedPerson = await complexDataContract.getPerson(personId);

    expect(returnedPerson.id).to.equal(personId);
    expect(returnedPerson.name).to.equal(personName);
    expect(returnedPerson.address.street).to.equal(street);
    expect(returnedPerson.address.city).to.equal(city);


  });
});

```

Again, you will see that properties from the nested struct are also returned as Javascript properties that we can directly assert against using dot notation. This makes tests highly readable and understandable.

If, for some reason, the returned struct is behaving unexpectedly, examining the raw return values using `console.log(returnedUser)` before running your tests can provide a clue. This way you can quickly identify how the returned data is being represented, and can access it accordingly. While this is a more primitive form of debugging, it is sometimes necessary, especially when there is doubt about how values are being returned from the contract.

There are also more advanced techniques we can use. For instance, if a struct contains a large amount of data and I want to perform a specific comparison of two objects, I would often reach for the `deep.equal` assertion provided by Chai. This allows for a much clearer assertion statement, which tests that each property in the struct matches a target struct.

For example, here's how you might use `deep.equal` if you want to test a more extensive comparison on our previous UserData contract:

```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("UserData - deep.equal example", function () {
  it("Should return the correct user data using deep.equal", async function () {
      const UserData = await ethers.getContractFactory("UserData");
      const userDataContract = await UserData.deploy();
      await userDataContract.deployed();

      const userId = 1;
      const userName = "Charlie";
      const userBalance = 200;

      await userDataContract.createUser(userId, userName, userBalance);

      const expectedUser = { id: userId, name: userName, balance: userBalance };
      const returnedUser = await userDataContract.getUser(userId);
      expect(returnedUser).to.deep.equal(expectedUser);
    });
});

```

This shows that we can construct an expected object, and use the `deep.equal` assertion to verify that all the fields in the returned struct from the smart contract match this expected object. This can be a much more concise approach when testing structs with many fields, or when testing multiple properties simultaneously.

In summary, testing functions that return structs in Solidity using Chai within Hardhat relies on understanding how Hardhat and ethers.js represent these structs in JavaScript. By accessing fields directly with dot notation and occasionally using methods such as deep.equal, you can create robust and reliable tests that ensure the proper functionality of your smart contracts. It is also worthwhile to consult the ethers.js documentation directly for further information on how data returned from smart contracts is handled. I recommend reading the documentation for ethers.js and the Chai assertion library to dive deeper into all possible testing scenarios, especially when you encounter more complicated scenarios than what we have seen here today.
