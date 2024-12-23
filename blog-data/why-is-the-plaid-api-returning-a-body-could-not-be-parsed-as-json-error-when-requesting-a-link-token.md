---
title: "Why is the Plaid API returning a 'body could not be parsed as JSON' error when requesting a link token?"
date: "2024-12-23"
id: "why-is-the-plaid-api-returning-a-body-could-not-be-parsed-as-json-error-when-requesting-a-link-token"
---

Alright, let's tackle this. I’ve definitely seen this “body could not be parsed as json” error with the Plaid API, more times than I’d care to recall, frankly. It usually pops up when you’re attempting to create a link token, and it can be incredibly frustrating, especially if you're new to the API or you're under a time constraint. This error, in its essence, points to a mismatch between the data you're sending to Plaid and what Plaid’s server expects. Let’s break down the typical culprits, based on my experiences debugging this over the years, and then I’ll offer some concrete examples.

The core issue is that the Plaid API, like most modern APIs, expects a specific data format – specifically, a properly formatted JSON payload – in the body of your http request. This error message is essentially saying, “I got something that's not valid JSON when I tried to interpret your request." Now, there are a handful of common reasons for this to occur.

First and foremost, verify your `content-type` header in your request. It needs to be explicitly set to `application/json`. If this header is absent or set to something else (like `text/plain` or `application/x-www-form-urlencoded`), the server may not interpret your request body as JSON, leading directly to this error. I remember one particular instance where a junior developer overlooked this header entirely; took us a while to catch it. It seems trivial, but in the heat of development, small details are easy to miss.

Another potential reason is that the structure of your json payload is incorrect, or it contains missing required fields or invalid values for those fields. Plaid's API is particular about what parameters it expects for a link token request. Even something as basic as an incorrect data type for one of the values can cause issues. For example, if you're passing a boolean value as a string, or an integer as a string, the JSON parser at Plaid will likely stumble. It's absolutely crucial to double-check the Plaid API documentation for the specific endpoint. They usually have a very clear schema outlining exactly what fields are needed and the expected types for those fields. The `plaid.client.link_token.create` method, for example, in one of the client libraries, requires a specific data structure. A common mistake I've seen is using older documentation or copy-pasting snippets of code, only to find that some values or fields have become outdated.

Furthermore, sometimes the encoding of the request body itself can cause the parse to fail. For example, if you’re somehow using an encoding other than utf-8, the server may struggle to interpret the request correctly. It’s a less common scenario, but I have seen situations with systems using custom or unusual encoding defaults. Ensure your system is transmitting data with utf-8.

Finally, sometimes, the issue isn't your code; it's something temporary on the Plaid side, although that’s far less common. While infrequent, there can be temporary issues with Plaid’s servers that might lead to sporadic JSON parsing failures. However, before suspecting their infrastructure, always rule out the issues within your own application. Network instability can also introduce errors; however, these issues typically produce different error codes rather than the "could not be parsed as json" message.

To illustrate all this more clearly, let's get into some practical examples with code snippets, across various environments:

**Example 1: Using Python with the Plaid Client Library**

Here’s a basic example of how you’d construct a link token request correctly, using the official python library. Note the `content-type` header is handled by the library for you:

```python
import plaid

client = plaid.Client(
    client_id='your_plaid_client_id',
    secret='your_plaid_secret',
    environment='sandbox' # or 'development' or 'production'
)

try:
  response = client.link_token.create({
      'user': {
          'client_user_id': 'some-unique-id' # Unique user identifier
      },
      'client_name': 'Your App Name',
      'products': ['auth', 'transactions'],
      'country_codes': ['US'],
      'language': 'en',
  })

  link_token = response['link_token']
  print(f"Link Token: {link_token}")

except plaid.ApiException as e:
  print(f"Error creating link token: {e}")
```

If you are getting the parsing error using this type of code, it might mean that the `client_id` and/or `secret` are incorrect. The error returned by `plaid.ApiException` will *not* return a parsing error in this case, but might be a cause of an improperly configured instance. I have seen a case where the `client_user_id` was an empty string, and that caused a similar response. Ensure you have unique user ids, and that they are not null, empty, or otherwise invalid.

**Example 2: Directly Sending an HTTP Request with `curl`**

For those who want to interact with the API directly without a library, here is an example using `curl`. Pay extra attention to the headers and payload format here:

```bash
curl -X POST \
  https://sandbox.plaid.com/link/token/create \
  -H 'Content-Type: application/json' \
  -H 'PLAID-CLIENT-ID: your_plaid_client_id' \
  -H 'PLAID-SECRET: your_plaid_secret' \
  -H 'Plaid-Version: 2020-09-14' \
  -d '{
    "user": {
        "client_user_id": "some-unique-id"
    },
    "client_name": "Your App Name",
    "products": ["auth", "transactions"],
    "country_codes": ["US"],
    "language": "en"
  }'
```

In the `curl` example, note that the `-H` flags are specifying the necessary headers, including `content-type: application/json`. If you omit the `content-type` header, or specify it incorrectly you will get the error message. Make sure the `plaid-version` header is set to a valid date, and that the `PLAID-CLIENT-ID` and `PLAID-SECRET` headers match your credentials, which must be set in your Plaid environment. I once saw a situation where a developer was using copy-pasted headers from an outdated example, which resulted in a similar error.

**Example 3: JavaScript/Node.js using `axios`:**

Here’s how you’d construct the request in a node environment using axios:

```javascript
const axios = require('axios');

const client_id = 'your_plaid_client_id';
const secret = 'your_plaid_secret';

axios.post('https://sandbox.plaid.com/link/token/create', {
    user: {
        client_user_id: 'some-unique-id'
    },
    client_name: 'Your App Name',
    products: ['auth', 'transactions'],
    country_codes: ['US'],
    language: 'en',
  }, {
    headers: {
        'Content-Type': 'application/json',
        'PLAID-CLIENT-ID': client_id,
        'PLAID-SECRET': secret,
        'Plaid-Version': '2020-09-14'
    }
})
.then(response => {
  const linkToken = response.data.link_token;
  console.log("Link Token:", linkToken);
})
.catch(error => {
  console.error("Error creating link token:", error);
});
```
In this example, the `axios.post()` call includes the data as the first argument, and the second argument contains the headers configuration. Here, if the content-type header is not set to ‘application/json’, it will produce the parse error. You’ll also see, for example, that if any of the request parameters such as `user`, `client_name`, `products`, or `country_codes` are undefined or do not have the correct type (like a string, or an array of strings), this may also cause the issue.

For further study and to gain more depth of understanding, I'd recommend the Plaid API documentation itself, found directly on their website, as they frequently update it. Also, the book "RESTful Web Services" by Leonard Richardson and Sam Ruby offers a great theoretical framework for understanding HTTP and JSON, and how to interact with APIs correctly. For a deeper dive into JSON parsing and web technologies, "High Performance Browser Networking" by Ilya Grigorik is an excellent resource.

By paying careful attention to your `content-type` headers, validating your json payload structure and values against the documentation, ensuring you are using utf-8 encoding, and being vigilant about client id and secret errors, you will likely eliminate this annoying error. I’ve found that consistently following these practices drastically reduces the occurrence of this problem. These practices are also important for most other api calls, regardless of which third party service you are using. These issues are not unique to Plaid’s API, so these are good general debugging techniques to understand.
