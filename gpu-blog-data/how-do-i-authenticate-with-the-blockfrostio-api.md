---
title: "How do I authenticate with the Blockfrost.io API?"
date: "2025-01-30"
id: "how-do-i-authenticate-with-the-blockfrostio-api"
---
Authentication with the Blockfrost.io API relies exclusively on API keys, specific to each project and network. Unlike some APIs that use OAuth or other complex methods, Blockfrost employs a straightforward header-based authentication, making it relatively simple to implement, though proper handling of the API key is crucial. Over the past three years, developing various Cardano-based applications, I have consistently interacted with their API, and have fine-tuned my approach to securing and using these keys.

**Understanding the API Key Mechanism**

The core concept is that each request to the Blockfrost.io API must contain an `Authorization` header with the format: `Authorization: Bearer YOUR_API_KEY`. The `YOUR_API_KEY` placeholder is where you input the API key generated from your Blockfrost account. These API keys are specific to a single network and a single project. Therefore, attempting to use a mainnet key on a testnet endpoint will invariably result in an authentication error. The API also supports multiple API keys per project, allowing for granular control and enabling specific tasks with separate credentials if required. Managing these keys carefully is essential to avoid unauthorized usage and maintain the security of your application. Exposing an API key in client-side code can lead to its compromise, so they should be handled server-side or within a secure environment.

**Code Examples and Explanations**

The following examples demonstrate how to authenticate with the Blockfrost API in three different programming languages, using common HTTP client libraries.

**Example 1: Python using `requests`**

```python
import requests
import json

API_KEY = "YOUR_API_KEY_GOES_HERE"  # Replace with your actual API key
PROJECT_ID = "your-project-id" # Replace with your actual project id

def get_block_info(block_hash):
  headers = {
      "project_id": PROJECT_ID,
      "Authorization": f"Bearer {API_KEY}"
  }
  url = f"https://cardano-mainnet.blockfrost.io/api/v0/blocks/{block_hash}"
  try:
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    return response.json()
  except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
    return None
  except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
    return None

if __name__ == "__main__":
    block_hash_to_fetch = "04a21a67b1299980879a5a462a7010ebff575a609793491c2a0b1c4f2d1c0992" # Example hash
    block_data = get_block_info(block_hash_to_fetch)
    if block_data:
        print(json.dumps(block_data, indent=2))

```

*   **Explanation:**  This Python snippet uses the `requests` library to perform a GET request to the `/blocks/{hash}` endpoint. The `API_KEY` variable needs to be replaced with your actual Blockfrost API key. The function `get_block_info` constructs the necessary headers, including `Authorization`, and sends the request. The `response.raise_for_status()` method is a crucial step for handling HTTP errors returned by the API. The `try-except` blocks ensure that both network related errors and errors decoding the JSON are handled gracefully.  The `json.dumps(block_data, indent=2)` formats the output for better readability.  A `project_id` header has been included as is now required by Blockfrost API.

**Example 2: JavaScript using `fetch` (Node.js compatible)**

```javascript
const fetch = require('node-fetch');

const API_KEY = "YOUR_API_KEY_GOES_HERE";  // Replace with your actual API key
const PROJECT_ID = "your-project-id" // Replace with your actual project id

async function getTransactionInfo(txHash) {
  const headers = {
    'project_id': PROJECT_ID,
      'Authorization': `Bearer ${API_KEY}`,
    };

    const url = `https://cardano-mainnet.blockfrost.io/api/v0/txs/${txHash}`;
  try {
    const response = await fetch(url, { headers });
    if (!response.ok) {
       throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Fetch Error: ${error}`);
    return null;
  }
}

async function main() {
    const txHashToFetch = "b751d97b63b3865776a4d61e333a56ffad9232649a98df98812b9349726b130a"; // Example hash
  const transactionData = await getTransactionInfo(txHashToFetch);
  if (transactionData) {
        console.log(JSON.stringify(transactionData, null, 2));
  }
}

main();

```

*   **Explanation:** This JavaScript example uses the `node-fetch` library to make HTTP requests, emulating the behavior of the `fetch` API that you would find in a browser environment. This makes it usable within Node.js.  Similar to the Python example, the `Authorization` header is constructed with the `Bearer` token, along with a project id header. The `response.ok` condition checks if the HTTP response status code is within the 200-299 range. The `async/await` keywords are used for asynchronous programming, and error handling is implemented with a `try...catch` block. The `JSON.stringify` method is used to format the output.  The `main` function allows easier integration with a Node.js runtime.

**Example 3: C# using `HttpClient`**

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text.Json;

public class BlockfrostApi
{
   private const string API_KEY = "YOUR_API_KEY_GOES_HERE";  // Replace with your actual API key
   private const string PROJECT_ID = "your-project-id" // Replace with your actual project id
  private static readonly HttpClient client = new HttpClient();

    public static async Task<JsonDocument?> GetAccountInfo(string address)
    {
        client.DefaultRequestHeaders.Clear();
         client.DefaultRequestHeaders.Add("project_id", PROJECT_ID);
        client.DefaultRequestHeaders.Add("Authorization", $"Bearer {API_KEY}");
        string url = $"https://cardano-mainnet.blockfrost.io/api/v0/accounts/{address}";

        try
        {
          HttpResponseMessage response = await client.GetAsync(url);
          response.EnsureSuccessStatusCode();  // Throw exception on error responses

          string responseBody = await response.Content.ReadAsStringAsync();

          return JsonDocument.Parse(responseBody);


        }
        catch (HttpRequestException e)
        {
            Console.WriteLine($"Request Error: {e.Message}");
           return null;
        }
        catch (JsonException e){
           Console.WriteLine($"JSON Parse Error: {e.Message}");
           return null;
        }

    }
   public static async Task Main(string[] args)
    {
        string addressToFetch = "addr1v9d74m3w952h547s50u6h9r9z8v4q783g0z4d465609f37n72"; // Example address

         var accountData = await GetAccountInfo(addressToFetch);
        if (accountData!= null) {
          Console.WriteLine(JsonSerializer.Serialize(accountData, new JsonSerializerOptions { WriteIndented = true }));

        }

    }

}
```

*   **Explanation:**  This C# example leverages the `HttpClient` class for making HTTP requests. The `API_KEY` variable is a constant, and must be updated to your actual key.  The `DefaultRequestHeaders` is used to add the necessary headers, including `Authorization`.  The `EnsureSuccessStatusCode` method does similar work to python's `response.raise_for_status` to detect 4xx and 5xx error codes. `JsonDocument` is the new high performance method of handling json in .NET Core. `JsonSerializer` is used to serialize the document to a readable string. Error handling within the `try...catch` blocks includes catching both network related exceptions and json parsing exceptions. The `Main` function makes the function readily runnable from the command line.

**Resource Recommendations**

For a deeper understanding of HTTP request fundamentals, explore materials on the following topics:

*   **HTTP Headers:** Familiarize yourself with the purpose of request headers, especially the `Authorization` header and the structure of a `Bearer` token. There are plenty of tutorials online and also within the W3C recommendations.
*   **HTTP Status Codes:** Learn about different HTTP status codes and what they signify. This will help you to debug errors arising from server-side and client-side issues. The standards are published by IETF.
*   **API Key Management:** Investigate secure ways to store and handle API keys. The OWASP guidelines for API security provide helpful guidance on best practices for authentication.
*   **Networking Libraries:** Study the documentation for the networking libraries used in each language of your choice. This will enhance your ability to leverage each library to its full potential. You should also read through the Blockfrost documentation to understand what endpoints are available.
