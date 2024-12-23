---
title: "How can I retrieve Solana NFT on-chain data using Python?"
date: "2024-12-23"
id: "how-can-i-retrieve-solana-nft-on-chain-data-using-python"
---

Okay, let's get into this. I remember a particularly tricky project a while back where we needed to deeply analyze on-chain metadata for a large collection of Solana NFTs. The sheer volume of data, coupled with Solana's specific architecture, presented some interesting challenges. We weren’t just after basic information like the token name; we needed details locked inside the metadata URI, and sometimes even had to parse program account data related to specific NFT standards. So, retrieving Solana NFT on-chain data using Python, it's not a single step but more of a structured process. Here’s how I approached it, and how you can too.

At the foundation, you'll be interacting with the Solana blockchain primarily through its json rpc interface. Python has excellent libraries for that, so we’ll start there. We'll primarily need the `requests` library for making http requests and `json` for handling the data returned. While you *could* use a raw rpc client directly, the `requests` module makes things far more approachable for most use cases.

The first step always involves getting the token account information. We need to fetch the data associated with a specific mint address, which is the unique identifier for an NFT collection. This is done by using the `getTokenAccountsByOwner` method. The query requires the wallet address holding the token and a filter specifying the mint address. This data gives us vital information, especially the account address where the NFT's metadata is stored. Here's a basic snippet to accomplish this:

```python
import requests
import json

def get_nft_account_data(rpc_url, wallet_address, mint_address):
    headers = {'Content-Type': 'application/json'}
    data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            wallet_address,
            {
                "mint": mint_address
            },
            {
                 "encoding": "jsonParsed",
            }
        ]
    }
    response = requests.post(rpc_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
         print(f"Error fetching token accounts: {response.status_code} {response.text}")
         return None
```

In this function, we construct the json rpc request, specifying the `getTokenAccountsByOwner` method and the relevant parameters: the owner's address, the mint address, and our requirement for 'jsonParsed' encoding. This is the preferred format because it presents data in a more structured manner that's readily usable in python. Remember that `rpc_url` would need to point to a valid Solana RPC node endpoint. There are many public providers, but for serious projects you'll likely want your own reliable node or a paid provider. The result of this call, if successful, is a json object containing the token account data. This often includes the token account address which is critical for retrieving metadata.

Now that we have the token account address, we can retrieve the actual metadata. Many Solana NFTs store metadata off-chain using a URI. We get this URI by fetching the account data using `getAccountInfo` rpc method, and then look for specific structures within the data field associated with the token account. The following snippet showcases this process:

```python
def get_nft_metadata_uri(rpc_url, token_account_address):
    headers = {'Content-Type': 'application/json'}
    data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [
               token_account_address,
               {
                    "encoding": "base64"
               }
          ]
     }
    response = requests.post(rpc_url, headers=headers, json=data)
    if response.status_code == 200:
        account_data = response.json()
        if 'result' in account_data and 'value' in account_data['result'] and 'data' in account_data['result']['value']:
             encoded_data = account_data['result']['value']['data'][0]
             decoded_data = base64.b64decode(encoded_data).decode('utf-8', errors='ignore')

            # Metadata URI is often found after certain constants - you might need to adjust this
            # depending on the specific standards and programs involved. This assumes Metaplex standards.
             try:
                start_index = decoded_data.find("https://")
                if start_index != -1:
                      uri_part = decoded_data[start_index:]
                      end_index = uri_part.find('\x00')
                      if end_index != -1:
                         return uri_part[:end_index]
                      else:
                           return uri_part
                else:
                       return None
             except ValueError:
                   return None


        else:
            print ("Account data format unexpected")
            return None

    else:
        print (f"Error fetching account info: {response.status_code} {response.text}")
        return None
```

This function retrieves the raw base64-encoded data, decodes it, then searches for the metadata uri. This assumes, as is often the case, that the uri begins with `https://`. The specific offsets and patterns for locating the uri depend on the program used to create the NFT; specifically whether it is a Metaplex certified program, or something more bespoke. Decoding base64 data correctly, and extracting that uri from the structured data requires some careful attention to detail and sometimes iterative debugging and careful inspection of the on-chain data using an explorer. After retrieving the uri, you'd fetch the metadata itself. This usually involves another http get request to the uri.

```python
import requests
import json
import base64

def get_nft_metadata(metadata_uri):
    try:
         response = requests.get(metadata_uri)
         if response.status_code == 200:
             return response.json()
         else:
             print (f"Error fetching metadata: {response.status_code} {response.text}")
             return None
    except requests.exceptions.RequestException as e:
         print (f"Error during metadata fetch: {e}")
         return None

if __name__ == "__main__":
    rpc_endpoint = "YOUR_RPC_URL"
    wallet = "YOUR_WALLET_ADDRESS"
    mint = "YOUR_NFT_MINT_ADDRESS"

    token_account_data = get_nft_account_data(rpc_endpoint, wallet, mint)
    if token_account_data and 'result' in token_account_data and 'value' in token_account_data['result']:
        if len(token_account_data['result']['value']) > 0:
            token_account_address = token_account_data['result']['value'][0]['pubkey']
            metadata_uri = get_nft_metadata_uri(rpc_endpoint,token_account_address)
            if metadata_uri:
                metadata = get_nft_metadata(metadata_uri)
                if metadata:
                  print ("NFT Metadata:",json.dumps(metadata,indent=4))
                else:
                    print ("Failed to retrieve NFT metadata")
            else:
                print ("Failed to retrieve metadata uri")

        else:
              print ("No token accounts found for this mint")
    else:
        print ("Failed to retrieve token account data")

```

This main section ties everything together. It shows the use of all functions created previously. Please replace `YOUR_RPC_URL`, `YOUR_WALLET_ADDRESS`, and `YOUR_NFT_MINT_ADDRESS` with your actual values. Remember, NFT on-chain data isn't always straightforward, and you will often see different patterns depending on how the contract is deployed.

For deeper understanding, I’d recommend digging into the Solana Program Library documentation, particularly the token program and Metaplex's metadata standard if you are working with those. A good starting point is the official Solana documentation site. Beyond that, "Programming Solana" by Matt Riley provides a fantastic technical overview that can help you reason about the underlying structure of account data more effectively. And finally, don’t underestimate the value of exploring the source code for established Solana SDKs, specifically for python, as these will provide a great foundation for working with the blockchain programmatically. This process might appear complex at first, but with time and practice, you'll develop a solid understanding of how to navigate on-chain data retrieval efficiently.
