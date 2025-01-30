---
title: "Why is web3.py throwing a TypeError: 'type' object is not subscriptable?"
date: "2025-01-30"
id: "why-is-web3py-throwing-a-typeerror-type-object"
---
The `TypeError: 'type' object is not subscriptable` when using `web3.py` typically arises from attempting to use square bracket indexing (e.g., `MyType[int]`) on a class, where it expects an instance of that class or an object capable of resolving such indexing. This error often surfaces within the context of type hinting or attempting to define dynamic type structures within the `web3.py` library, particularly concerning contract interactions and data encoding. I've frequently encountered this while developing decentralized applications, specifically when dealing with complex smart contract function signatures.

The Python interpreter raises this error because types (like `int`, `str`, `list`) are themselves objects representing a kind of data. They are not containers that hold other types; thus, you cannot subscript them the same way you would a list or a dictionary. `web3.py` relies on the `eth-abi` library for encoding and decoding data based on the expected types specified in smart contract ABIs (Application Binary Interfaces). When incorrect type annotations or usage patterns are present, this underlying mechanism fails. The issue commonly occurs when developers misunderstand how to properly define the types of arguments or return values, especially with composite types like tuples, arrays, or mappings.

To clarify, type hinting in Python (using annotations like `arg: int`) is primarily for static analysis tools (like linters) and human readability. At runtime, Python does not enforce these type hints in the same manner as strongly typed languages, except within specific contexts where it is implicitly expected. `web3.py`, however, utilizes these annotations when performing complex data serialization and deserialization.

Let’s look at a typical scenario that would cause this error:

**Example 1: Incorrect Type Annotation for Contract Function Call**

```python
from web3 import Web3
from web3.contract import Contract

# Assuming 'contract_abi' and 'contract_address' are correctly defined and valid.
# This is simplified for demonstration.
contract_abi = [...] # Placeholder, assumed to be a valid ABI definition
contract_address = "0x..." # Placeholder, assumed to be a valid address

w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Incorrect Type Hinting. Should be a specific type, not the generic type
def my_function(arg: type) -> None:
  print(arg)

try:
    tx_hash = contract.functions.myContractFunction(my_function).transact()
except Exception as e:
    print(f"Error: {e}")
```

In this code, `my_function` is intended to be used as a callback, taking the result of a smart contract function call as its argument. However, the type annotation `arg: type` is incorrect and is not a concrete type. When `web3.py` or `eth-abi` attempt to determine how to encode or pass data to this function, it encounters the `type` object instead of a concrete data type like an integer, string, or address, thus raising the `TypeError`. The correct annotation should align with the smart contract’s return type or a type that can be converted into it. For example, if the smart contract function returns an unsigned integer, the callback should expect `arg: int`. The error here is not explicitly about how we are using my_function but rather how web3.py parses type information which is needed for ABI encoding. The problem isn’t with the function itself but rather how web3.py tries to parse type hints.

A corrected version of this example would look like this, assuming the contract function returns an integer:

```python
from web3 import Web3
from web3.contract import Contract

contract_abi = [...] # Placeholder, assumed to be a valid ABI definition
contract_address = "0x..." # Placeholder, assumed to be a valid address

w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Correct Type Hinting assuming the contract returns an int
def my_function(arg: int) -> None:
  print(arg)


tx_hash = contract.functions.myContractFunction(my_function).transact()
```

**Example 2: Incorrectly Defining a Tuple**

```python
from web3 import Web3
from web3.contract import Contract
from typing import Tuple

contract_abi = [...] # Placeholder, assumed to be a valid ABI definition
contract_address = "0x..." # Placeholder, assumed to be a valid address


w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def process_tuple(my_tuple: Tuple):
    # This fails because Tuple requires a specification of the types.

    print(my_tuple[0])

try:
    result = contract.functions.myFunctionReturningATuple().call()
    process_tuple(result)
except Exception as e:
    print(f"Error: {e}")

```

Here, the `Tuple` type hint is used without specifying the types within the tuple. When `web3.py` receives the output from a function returning a tuple, it needs concrete type information to know how to decode the data. The error arises when the type hint given to the function `process_tuple` is ambiguous. Specifically, `typing.Tuple` needs to be defined with the type information `Tuple[int, str]` rather than `Tuple`.

The corrected version with the correct type information would be:

```python
from web3 import Web3
from web3.contract import Contract
from typing import Tuple

contract_abi = [...] # Placeholder, assumed to be a valid ABI definition
contract_address = "0x..." # Placeholder, assumed to be a valid address


w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def process_tuple(my_tuple: Tuple[int, str]):
    # This works because the type of each entry is specified.
    print(my_tuple[0])
    print(my_tuple[1])


result = contract.functions.myFunctionReturningATuple().call()
process_tuple(result)

```

**Example 3: Using generic Lists in type annotations**

```python
from web3 import Web3
from web3.contract import Contract
from typing import List

contract_abi = [...] # Placeholder, assumed to be a valid ABI definition
contract_address = "0x..." # Placeholder, assumed to be a valid address


w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def process_list(my_list: List):
    # Fails because the type of list elements is unknown to web3.
    print(my_list[0])

try:
    result = contract.functions.myFunctionReturningAList().call()
    process_list(result)
except Exception as e:
    print(f"Error: {e}")
```

Similar to the tuple example, this shows the same problem occurring with Lists. The type `List` requires a specification of the type held by the elements, such as `List[int]` or `List[str]`. Web3.py requires this level of specification to properly handle encoding and decoding data from smart contracts.

A corrected version of this example is:

```python
from web3 import Web3
from web3.contract import Contract
from typing import List

contract_abi = [...] # Placeholder, assumed to be a valid ABI definition
contract_address = "0x..." # Placeholder, assumed to be a valid address


w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def process_list(my_list: List[int]):
    print(my_list[0])

result = contract.functions.myFunctionReturningAList().call()
process_list(result)

```

In summary, the `TypeError: 'type' object is not subscriptable` within `web3.py` is a direct consequence of the library being unable to deduce correct types based on incomplete or incorrect type annotations. When writing callback functions, handling return values from smart contracts, or when dealing with composite data types like tuples and lists, it is essential to define the type hints with enough specificity. If type hinting is omitted, `web3.py` often falls back to defaults which don't trigger the specific error detailed above; rather, the error occurs when using generic types that would work at runtime but do not allow the library to process the encoded data. Specifically, this error arises when it encounters `type` objects where the `web3.py` library expects more concrete data type definitions.

To further improve understanding and avoid this error, I recommend exploring the official Python documentation on type hinting, specifically the `typing` module. Furthermore, studying the `eth-abi` documentation can help clarify the specific requirements of data encoding and decoding in Ethereum transactions. Examining examples within the `web3.py` documentation and its test suite related to contract interactions, events, and data types provides practical examples. Finally, meticulous review of ABI definitions and their corresponding Python representations is key.
