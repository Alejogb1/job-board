---
title: "How can I retrieve Cardano UTxOs via an API?"
date: "2025-01-30"
id: "how-can-i-retrieve-cardano-utxos-via-an"
---
The core challenge in retrieving Cardano UTxOs through an API stems from the inherent structure of the blockchain, where transaction outputs are not directly associated with addresses but exist as Unspent Transaction Outputs (UTxOs), which must be consumed as inputs for subsequent transactions. This is fundamentally different from an account-based ledger, requiring a different approach to data retrieval.

As a developer who has spent considerable time building decentralized applications on Cardano, I can say that directly querying for all UTxOs associated with a single address using a typical RESTful API call isn't the norm. Instead, we have to rely on a combination of address-based queries and subsequent parsing of the response data to piece together the relevant UTxOs. Cardano's blockchain is designed for efficiency and verification, not simple address-centric lookups. The process primarily involves querying a node (via a public or private API) for the UTxOs associated with one or more addresses. This returns a set of output details, but not necessarily in a form directly consumable. It often involves further processing.

The most common pathway to access UTxO information is via a Cardano node's API. This can be a locally-run node or a third-party provider’s API. The data usually is exposed through a JSON-formatted response. The structure returned by the API contains a list of UTxO objects, where each object includes details such as: transaction ID (tx_hash), output index (tx_index), address, datum hash (if applicable), output value (coins), and associated assets. The actual API endpoint varies based on the API provider, but the basic principle remains the same: request UTxOs at a given address.

Let’s consider three different scenarios that illustrate practical use.

**Example 1: Retrieving UTxOs for a Single Address**

I’ll begin with the simplest scenario: querying for all UTxOs associated with a single address. This assumes we’re using a hypothetical API endpoint, `/api/v1/addresses/utxos/{address}`, which can be substituted with the provider-specific URL.

```python
import requests
import json

def get_utxos_for_address(address, api_url):
    url = f"{api_url}/api/v1/addresses/utxos/{address}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        utxos = response.json()
        return utxos
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return None

def process_utxos(utxos):
    if not utxos:
        print("No UTxOs found or error occurred.")
        return
    for utxo in utxos:
        tx_hash = utxo['tx_hash']
        tx_index = utxo['tx_index']
        value = utxo['value'] # represents lovelace
        print(f"Transaction ID: {tx_hash}, Index: {tx_index}, Value: {value}")
        if 'assets' in utxo:
           for asset in utxo['assets']:
                policy_id = asset['policy_id']
                asset_name = asset['asset_name']
                quantity = asset['quantity']
                print(f"  Asset - Policy: {policy_id}, Name: {asset_name}, Quantity: {quantity}")


if __name__ == "__main__":
    address = "addr1qyf6w9z49m5x3l7t729j2s987q94w93f8550h2j3g8g7j4t6c6j5z6w" # Replace with a valid Cardano address
    api_url = "https://example-cardano-api.com" # Replace with your API endpoint
    utxos = get_utxos_for_address(address, api_url)
    process_utxos(utxos)
```

**Commentary:**

This Python script first defines a function, `get_utxos_for_address`, that sends a GET request to the specified API endpoint with the address as a path parameter.  Error handling is included with `response.raise_for_status()`.  If the call is successful, it parses the JSON response into a Python dictionary. The `process_utxos` function iterates through the list of UTxO objects. It extracts key details like the transaction hash, output index, lovelace value, and any associated native assets. Finally, a sample usage shows how to retrieve and print UTxO details. Notice that specific error handling such as rate limiting or authentication are omitted to focus on the core operation of UTxO retrieval.

**Example 2: Retrieving UTxOs for Multiple Addresses**

Often, we need to retrieve UTxOs from multiple addresses simultaneously. This can be achieved by sending multiple individual requests or utilizing an endpoint accepting an array of addresses. The example below uses the latter, assuming the endpoint `/api/v1/addresses/utxos` accepts addresses via a POST request:

```python
import requests
import json

def get_utxos_for_multiple_addresses(addresses, api_url):
    url = f"{api_url}/api/v1/addresses/utxos"
    try:
        response = requests.post(url, json={'addresses': addresses})
        response.raise_for_status()
        utxos_per_address = response.json()
        return utxos_per_address
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return None

def process_all_utxos(utxos_per_address):
    if not utxos_per_address:
        print("No UTxOs found or error occurred.")
        return
    for address, utxos in utxos_per_address.items():
        print(f"UTxOs for address: {address}")
        for utxo in utxos:
            tx_hash = utxo['tx_hash']
            tx_index = utxo['tx_index']
            value = utxo['value']
            print(f"  Transaction ID: {tx_hash}, Index: {tx_index}, Value: {value}")
            if 'assets' in utxo:
                for asset in utxo['assets']:
                    policy_id = asset['policy_id']
                    asset_name = asset['asset_name']
                    quantity = asset['quantity']
                    print(f"    Asset - Policy: {policy_id}, Name: {asset_name}, Quantity: {quantity}")

if __name__ == "__main__":
    addresses = [
        "addr1qyf6w9z49m5x3l7t729j2s987q94w93f8550h2j3g8g7j4t6c6j5z6w",
        "addr1q9g8h7j6k5l4z3t2s1p0o9n8m7l6k5j4h3g2f1e0d9c8b7a6z5y4x"  # Replace with valid Cardano addresses
    ]
    api_url = "https://example-cardano-api.com" # Replace with your API endpoint
    utxos = get_utxos_for_multiple_addresses(addresses, api_url)
    process_all_utxos(utxos)
```

**Commentary:**

The code now sends a POST request to the `/api/v1/addresses/utxos` endpoint, passing a JSON payload containing a list of addresses. The response is structured as a dictionary with addresses as keys and lists of their UTxOs as values. The  `process_all_utxos` function iterates through this structure, extracting and printing the relevant information in a similar format as the first example. This approach is generally more efficient than performing multiple separate queries, especially if there are a significant number of addresses to query.

**Example 3: Filtering UTxOs by Asset**

It may be crucial to retrieve UTxOs only containing specific assets, such as native tokens or NFTs. This usually involves iterating through the returned UTxO list and performing filtering logic on the client side, as there isn’t a standard filter on the API query itself for specific assets.

```python
import requests
import json

def get_utxos_for_address(address, api_url): #reusing from example 1
    url = f"{api_url}/api/v1/addresses/utxos/{address}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        utxos = response.json()
        return utxos
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return None

def filter_utxos_by_asset(utxos, policy_id, asset_name):
    filtered_utxos = []
    if not utxos:
        return filtered_utxos
    for utxo in utxos:
        if 'assets' in utxo:
            for asset in utxo['assets']:
                if asset['policy_id'] == policy_id and asset['asset_name'] == asset_name:
                    filtered_utxos.append(utxo)
                    break
    return filtered_utxos

def process_utxos(utxos):
    if not utxos:
        print("No UTxOs found or error occurred.")
        return
    for utxo in utxos:
        tx_hash = utxo['tx_hash']
        tx_index = utxo['tx_index']
        value = utxo['value'] # represents lovelace
        print(f"Transaction ID: {tx_hash}, Index: {tx_index}, Value: {value}")
        if 'assets' in utxo:
           for asset in utxo['assets']:
                policy_id = asset['policy_id']
                asset_name = asset['asset_name']
                quantity = asset['quantity']
                print(f"  Asset - Policy: {policy_id}, Name: {asset_name}, Quantity: {quantity}")

if __name__ == "__main__":
    address = "addr1qyf6w9z49m5x3l7t729j2s987q94w93f8550h2j3g8g7j4t6c6j5z6w" # Replace with a valid Cardano address
    api_url = "https://example-cardano-api.com" # Replace with your API endpoint
    policy_id_filter = "1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd" # Replace with desired policy_id
    asset_name_filter = "MyToken" # Replace with desired asset_name
    utxos = get_utxos_for_address(address, api_url)
    filtered_utxos = filter_utxos_by_asset(utxos, policy_id_filter, asset_name_filter)
    process_utxos(filtered_utxos)
```

**Commentary:**

Here, the `filter_utxos_by_asset` function takes the list of UTxOs, a desired policy ID, and asset name as input. It then iterates over each UTxO and its assets.  It checks if any of the assets match the given policy ID and asset name and, if a match is found, adds that UTxO to the filtered list. Only UTxOs that match the specified asset criteria are subsequently processed and outputted. This example highlights the need for client-side logic when API endpoints don't offer specific filtering capabilities.

**Resource Recommendations:**

To deepen your understanding of Cardano UTxO retrieval, I would recommend exploring the documentation associated with your chosen node API provider, whether it's Blockfrost, Koios, or another service. Also, examining the source code of existing Cardano libraries or SDKs (particularly written in Javascript, Python, or Haskell) can provide insight into established patterns for interacting with Cardano node APIs, and how to process the returned UTxO data efficiently. Finally, the Cardano documentation itself offers fundamental knowledge and deep-dives that help in grasping the core principles of the UTxO model.
