---
title: "Can Web3py call smart contract functions concurrently using threads?"
date: "2025-01-30"
id: "can-web3py-call-smart-contract-functions-concurrently-using"
---
Web3.py, by design, relies on synchronous calls for interacting with Ethereum smart contracts. Dispatching multiple contract function calls concurrently using threads requires careful consideration of the underlying architecture and potential pitfalls related to thread-safety and transaction nonce management. I've encountered and resolved these issues in various decentralized application deployments, and will outline the process and its complexities.

The crucial constraint stems from Web3.py's reliance on a single, underlying JSON-RPC provider connection. Each interaction, whether reading contract state or initiating a transaction, communicates via this shared connection. While Python threads can execute concurrently, they are susceptible to issues when sharing mutable resources, which in this context is essentially the Web3.py provider and its related transaction state. If multiple threads attempt to send transactions simultaneously through the same provider without proper locking mechanisms or nonce management, race conditions and transaction failures are almost certain. Consequently, attempting a direct threaded approach without careful design will likely result in transaction collisions or nonce errors.

The correct solution doesn’t involve direct use of threading to speed up contract calls within a single Web3 instance, rather it requires a careful orchestration of multiple Web3 instances, each with its own provider, or the use of asynchronous programming that leverages non-blocking I/O. Using threading directly, without a Web3 provider per thread will not actually be concurrent. A more effective strategy involves managing a pool of Web3 instances, each operating independently, or by using asynchronous programming paradigms. Threading for concurrent execution in Web3.py is effective for transaction broadcast, nonce management, and other non-I/O related tasks. I will demonstrate both approaches: using separate Web3 instances in a thread pool and using asynchronous I/O.

**Approach 1: Threaded Execution with Separate Web3 Instances**

The core principle here is to avoid contention for a single Web3 provider. Each thread receives a fresh Web3 instance configured with its own provider. Transactions initiated in different threads then proceed independently, side-stepping nonce collisions. This approach works well if the number of concurrent operations is relatively modest. If large-scale concurrency is desired, managing and cleaning these resources becomes an added complexity.

```python
import threading
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from concurrent.futures import ThreadPoolExecutor

# Helper function to get a new Web3 instance
def create_web3_instance(rpc_url, private_key):
  w3 = Web3(Web3.HTTPProvider(rpc_url))
  w3.middleware_onion.inject(geth_poa_middleware, layer=0)
  account = Account.from_key(private_key)
  w3.eth.default_account = account.address
  return w3

# Function to execute a contract interaction
def interact_with_contract(w3, contract_address, function_name, *args):
    contract = w3.eth.contract(address=contract_address, abi=CONTRACT_ABI)
    try:
        tx_hash = contract.functions[function_name](*args).transact()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        return {"error": str(e)}


# Example usage
def run_threaded_contract_calls(rpc_url, private_keys, contract_address, function_name, args_list):
    results = []
    with ThreadPoolExecutor(max_workers=len(private_keys)) as executor:
        futures = [
            executor.submit(
                interact_with_contract,
                create_web3_instance(rpc_url, private_key),
                contract_address,
                function_name,
                *args
            )
            for private_key, args in zip(private_keys, args_list)
        ]
        for future in futures:
            results.append(future.result())
    return results

# Example Contract Address
CONTRACT_ADDRESS = '0xd8da6bf26964af9d7eed9e03e53415d37aa96045'

# Example contract ABI
CONTRACT_ABI = [...]  # The actual contract abi should be here

# Configuration
RPC_URL = "http://localhost:8545"
PRIVATE_KEYS = [
    "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    "0x112233445566778899aabbccddeeff112233445566778899aabbccddeeff"
]
FUNCTION_NAME = "myFunction"
ARGS_LIST = [
    (100,),
    (200,),
    (300,)
]

if __name__ == "__main__":
  results = run_threaded_contract_calls(RPC_URL, PRIVATE_KEYS, CONTRACT_ADDRESS, FUNCTION_NAME, ARGS_LIST)
  print(results)
```

**Code Commentary:**

1.  **`create_web3_instance`**: This function initializes a new `Web3` object with its own HTTP provider and sets the default account using a unique private key. This prevents thread contention by ensuring each thread uses a distinct Web3 instance.
2.  **`interact_with_contract`**: This function encapsulates the logic for a single contract call, taking a dedicated `Web3` object, contract address, function name, and arguments. It executes the transaction and waits for the receipt.
3. **`run_threaded_contract_calls`:** This function demonstrates the use of the ThreadPoolExecutor, submitting tasks to execute the interact\_with\_contract function, using a different private key for each.
4.  **Example Usage**:  Demonstrates a simple execution of the functions including contract address, ABI, rpc, private keys, function name and arguments.

**Approach 2: Asynchronous Programming with `asyncio`**

Python's `asyncio` module offers a different, and often more efficient, method for concurrent tasks. This approach avoids many of the complexities and overhead associated with threads, especially for I/O-bound operations like network interactions. However, it requires a different coding paradigm, using `async` and `await` keywords. For Web3.py this means using a web3 provider with an underlying asynchronous HTTP client, like `aiohttp`. This removes thread contention by only allowing a single thread to make Web3 calls, while leveraging non-blocking I/O.

```python
import asyncio
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
import aiohttp

# Helper function to get a new Web3 instance with async provider
def create_async_web3_instance(rpc_url, private_key):
    async def get_session():
        return aiohttp.ClientSession()

    w3 = Web3(Web3.AsyncHTTPProvider(rpc_url, session=asyncio.run(get_session())))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    account = Account.from_key(private_key)
    w3.eth.default_account = account.address
    return w3

# Function to execute a contract interaction asynchronously
async def async_interact_with_contract(w3, contract_address, function_name, *args):
    contract = w3.eth.contract(address=contract_address, abi=CONTRACT_ABI)
    try:
        tx_hash = await contract.functions[function_name](*args).transact()
        receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        return {"error": str(e)}


# Example usage
async def run_async_contract_calls(rpc_url, private_keys, contract_address, function_name, args_list):
    tasks = [
        async_interact_with_contract(
            create_async_web3_instance(rpc_url, private_key),
            contract_address,
            function_name,
            *args
        )
        for private_key, args in zip(private_keys, args_list)
    ]
    results = await asyncio.gather(*tasks)
    return results

# Example Contract Address
CONTRACT_ADDRESS = '0xd8da6bf26964af9d7eed9e03e53415d37aa96045'

# Example contract ABI
CONTRACT_ABI = [...]  # The actual contract abi should be here

# Configuration
RPC_URL = "http://localhost:8545"
PRIVATE_KEYS = [
    "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    "0x112233445566778899aabbccddeeff112233445566778899aabbccddeeff"
]
FUNCTION_NAME = "myFunction"
ARGS_LIST = [
    (100,),
    (200,),
    (300,)
]


if __name__ == "__main__":
  results = asyncio.run(run_async_contract_calls(RPC_URL, PRIVATE_KEYS, CONTRACT_ADDRESS, FUNCTION_NAME, ARGS_LIST))
  print(results)
```

**Code Commentary:**

1.  **`create_async_web3_instance`**: Creates a `Web3` object using an asynchronous HTTP provider, and each has its own private key. This utilizes aiohttp for non-blocking I/O.
2.  **`async_interact_with_contract`**: An asynchronous version of the contract interaction function, utilizing `await` for contract call and waiting for receipt, making sure to return the result.
3.  **`run_async_contract_calls`**: Gathers all of the asynchronous tasks and waits for them to complete.

**Resource Recommendations**

*   **Web3.py Documentation**: The official documentation provides detailed explanations of provider configuration, middleware, and contract interaction.
*   **Python Concurrency Documentation**:  The official documentation on the `threading` and `asyncio` modules are essential for understanding how concurrency functions.
*  **Ethereum JSON-RPC Specification**:  Understanding the underlying JSON-RPC interface is helpful for troubleshooting issues related to connection management.

In conclusion, while direct threading with a single Web3.py instance is not advisable for concurrent contract function calls due to nonce collisions, you can use threading with a pool of web3 instances, or employ asynchronous programming with aiohttp-based providers to achieve concurrent transaction processing without these limitations. Careful consideration of nonce management and thread-safety is crucial for successful implementation. My experiences with high-volume transaction processing in decentralized applications has led me to prefer the asynchronous model, which tends to have better overall performance and fewer headaches related to thread-safety. However, using thread pools can be a faster entry point into concurrent transaction calls in simpler scenarios.
