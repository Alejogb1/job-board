---
title: "How can I obtain all addresses owned by a contract in Solidity?"
date: "2024-12-23"
id: "how-can-i-obtain-all-addresses-owned-by-a-contract-in-solidity"
---

Right,  Obtaining all addresses owned by a contract in Solidity—that's a problem I’ve certainly navigated a few times in my projects, and it's not as straightforward as you might initially assume. The inherent nature of the ethereum virtual machine (evm) and solidity's design doesn't provide a direct way to query "all addresses a contract controls.” Instead, we need to consider the specific patterns by which a contract manages addresses and how to programmatically track them.

The primary challenge stems from the fact that smart contracts operate within a deterministic, state-based environment. Solidity contracts don't intrinsically track what external accounts (or contracts) they 'own' or control in the traditional sense. The concept of ownership here is not explicit; it's more about the contract's interactions and state changes that *imply* ownership or control over specific addresses.

In my experience, the methods to extract such addresses usually revolve around logging events emitted by the contract and meticulously tracking state changes using storage variables. Let’s consider a few common scenarios.

First off, if your contract uses mappings to manage balances or associations with addresses, this is the most common pattern I've encountered. For example, a token contract likely holds a mapping that associates each address with its corresponding balance:

```solidity
pragma solidity ^0.8.0;

contract ExampleToken {
    mapping(address => uint256) public balances;
    address public owner;

    constructor() {
       owner = msg.sender;
       balances[owner] = 1000; // Owner starts with tokens
    }

    function transfer(address _to, uint256 _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
        emit Transfer(msg.sender, _to, _amount);
    }

    event Transfer(address indexed from, address indexed to, uint256 value);
}
```

In this case, to extract the addresses holding tokens, you would not iterate through the entire mapping in the contract (that's not viable due to gas costs). Instead, you'd have to parse all emitted `Transfer` events. With web3 libraries, you'd filter for the specific `Transfer` events associated with this contract, extract the `from` and `to` addresses, and keep a unique set of address that have ever been part of the `balances` mapping. It requires a bit of post-processing but is the most practical method. While the `owner` is an address the contract operates with, it is known during creation and we are looking for all addresses that interact with the contract. Note that this is not an exhaustive list, because not all addresses that transfer to or from the token contract must store a balance.

Now, let’s move onto a more complex scenario where a contract might dynamically create other contracts, effectively ‘owning’ them in an operational sense.

```solidity
pragma solidity ^0.8.0;

contract Factory {
    address[] public deployedContracts;

    event ContractDeployed(address indexed newContract);

    function deployNewContract() public {
        SimpleContract newContract = new SimpleContract();
        deployedContracts.push(address(newContract));
        emit ContractDeployed(address(newContract));
    }

    function getDeployedContracts() public view returns(address[] memory) {
        return deployedContracts;
    }

}

contract SimpleContract {
  address public owner;

  constructor() {
    owner = msg.sender;
  }
}
```

Here, the `Factory` contract deploys instances of `SimpleContract`. To get all deployed contract addresses, you would either read the `deployedContracts` array through a view call, or parse for `ContractDeployed` events. Using view call is the most straightforward method in this case. If you want the owners of all contracts deployed by the factory, you will have to perform a read operation on each of the deployed `SimpleContract`'s `owner` variable.

Finally, let’s examine a more nuanced case where ownership might be implied through a contract’s internal logic, such as when a contract acts as a multisignature wallet.

```solidity
pragma solidity ^0.8.0;

contract MultiSigWallet {
    address[] public owners;
    uint public requiredConfirmations;
    mapping(bytes32 => bool) public confirmations;
    mapping(bytes32 => Transaction) public transactions;
    uint public transactionCount;


    struct Transaction {
        address destination;
        uint value;
        bytes data;
        bool executed;
    }

    constructor(address[] memory _owners, uint _requiredConfirmations) {
        require(_owners.length > 0, "Owners required");
        require(_requiredConfirmations > 0 && _requiredConfirmations <= _owners.length, "Invalid required confirmations");
        owners = _owners;
        requiredConfirmations = _requiredConfirmations;
    }

    function submitTransaction(address _destination, uint _value, bytes memory _data) public  {
        bytes32 txHash = keccak256(abi.encode(_destination, _value, _data, transactionCount));
        transactions[txHash] = Transaction(_destination, _value, _data, false);
        transactionCount++;
    }

    function confirmTransaction(bytes32 txHash) public {
        require(transactions[txHash].executed == false, "Transaction already executed");
        require(isOwner(msg.sender), "Only owners can confirm transactions");
        confirmations[txHash] = true;
        uint confirmationCount = 0;
        for (uint i = 0; i < owners.length; i++){
            if(confirmations[keccak256(abi.encode(transactions[txHash].destination, transactions[txHash].value, transactions[txHash].data, transactionCount - 1))]) {
               confirmationCount++;
            }
        }
        if (confirmationCount >= requiredConfirmations) {
          transactions[txHash].executed = true;
           (bool success, ) = transactions[txHash].destination.call{value: transactions[txHash].value}(transactions[txHash].data);
            require(success, "Transaction call failed");
        }
    }

    function isOwner(address _address) internal view returns (bool) {
        for(uint i = 0; i < owners.length; i++){
            if(owners[i] == _address) {
              return true;
            }
        }
        return false;
    }
}

```

In this multisig wallet, the `owners` array directly holds the addresses that have control over the wallet. You would simply retrieve the `owners` array via a view call, as it is public. Note that we are not talking about transaction history in this context, but addresses of the owners themselves. These examples highlight a critical point: extracting "owned" addresses is a case-by-case analysis of a given contract's behavior.

There isn’t a universal solution due to the variety of patterns a smart contract can utilize. The common threads are analyzing contract events and state variables, and then employing an iterative filtering and processing technique to derive the address list. I'd recommend studying the *Solidity documentation* thoroughly. The section on mappings, events, and contract interactions are particularly relevant here. Also, the book *Mastering Ethereum* by Andreas Antonopoulos, Gavin Wood, is invaluable for understanding the underlying mechanics of smart contract design and behaviour on the evm. Finally, look at academic papers on blockchain data analysis, particularly those that explore how to extract information from on-chain events, as these often have methodologies you can leverage and refine for contract-specific situations. Remember, careful analysis of the specific contract's logic is key to accurately determining what addresses can be considered "controlled" by that contract. You'll likely find the most pertinent solutions by adapting the concepts I've discussed to your specific implementation, by focusing on events, contract storage and state variables.
