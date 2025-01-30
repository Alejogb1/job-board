---
title: "How can I asynchronously send and receive JSON messages in Node.js?"
date: "2025-01-30"
id: "how-can-i-asynchronously-send-and-receive-json"
---
Asynchronous communication is fundamental for efficient network applications. In Node.js, handling JSON messages asynchronously typically involves employing non-blocking I/O with libraries focused on network operations, particularly when dealing with streams. The core concept is to avoid blocking the main thread while waiting for network responses, enabling the application to process other tasks concurrently. I've found this approach crucial in maintaining application responsiveness, particularly under heavy load.

Let’s break this process down into its constituent parts: sending and receiving. Sending JSON asynchronously often involves converting a JavaScript object to a JSON string, and then transmitting that string over a socket or HTTP connection. Receiving involves the reverse process: accumulating data fragments as they arrive over the network, reconstructing a complete message, parsing it from a JSON string back into a JavaScript object, and then acting upon that data. Both these operations need to be handled without blocking the Node.js event loop.

The `net` and `http` modules are the usual suspects for network operations. However, for streamlined JSON processing, I’ve often used custom abstraction layers that build upon these core modules. The focus remains on handling network events asynchronously through callbacks or promises/async-await.

Here’s an example of a simple TCP client using the `net` module, demonstrating asynchronous JSON sending and receiving:

```javascript
const net = require('net');

const client = net.createConnection({ port: 3000 }, () => {
  console.log('Connected to server.');

  // Prepare a JSON payload
  const jsonData = { messageType: 'request', data: { id: 123, name: 'Test Item' } };
  const jsonString = JSON.stringify(jsonData);

  // Send the JSON string with a newline as a delimiter
  client.write(jsonString + '\n');
});

let receivedData = '';

client.on('data', (chunk) => {
  receivedData += chunk.toString();

  // Attempt to extract and process full JSON messages
  let lastNewlineIndex = receivedData.indexOf('\n');
  while (lastNewlineIndex !== -1) {
    try {
        const message = JSON.parse(receivedData.slice(0, lastNewlineIndex));
        console.log('Received JSON:', message);
        // Handle the received JSON message here
    }
    catch(err){
        console.error("Failed to parse a message", err)
    }
    
    receivedData = receivedData.slice(lastNewlineIndex + 1);
    lastNewlineIndex = receivedData.indexOf('\n');
  }
});


client.on('end', () => {
  console.log('Disconnected from server.');
});


client.on('error', (err) => {
  console.error('Client error:', err);
});
```

In this code, I create a client that connects to a server on port 3000. A JSON object is converted into a string and sent, followed by a newline character. The server also needs to be implemented to recognize the newline character as a delimiter. The client then listens for data events, concatenating the received data, and parses JSON messages based on newline delimiters. If there’s no new line, the data is held in memory until a new line character is received. Error handling is included to catch JSON parsing or network errors, which are essential in asynchronous programming.

This approach of using a delimiter is one way to ensure complete messages are parsed.  Streaming JSON over sockets can sometimes result in partial reads, hence needing to reconstruct the entire message before parsing.

For a more robust solution with HTTP, consider the following example using the `http` module to POST JSON data:

```javascript
const http = require('http');

const jsonData = { messageType: 'update', data: { status: 'online', timestamp: Date.now() } };
const jsonString = JSON.stringify(jsonData);

const options = {
  hostname: 'localhost',
  port: 8080,
  path: '/api/update',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(jsonString),
  },
};

const req = http.request(options, (res) => {
  console.log(`Status code: ${res.statusCode}`);

  let responseData = '';
  res.on('data', (chunk) => {
    responseData += chunk;
  });

  res.on('end', () => {
    try{
      const responseJson = JSON.parse(responseData);
      console.log("Response JSON:", responseJson);
        //Handle response json here
    }
    catch(err){
        console.error("Failed to parse response json", err)
    }

  });
});

req.on('error', (err) => {
  console.error('Request error:', err);
});

req.write(jsonString);
req.end();
```
This example sends a POST request to an HTTP server. We set the 'Content-Type' to 'application/json' and calculate the 'Content-Length' header beforehand. This is important for HTTP requests to ensure that the server knows the message boundaries. The response is collected in a similar manner as the TCP client, with concatenation of chunks and error handling. The returned data, ideally JSON, is then parsed.

For a more complex scenario, where we have several messages being sent and received, managing the flow with asynchronous programming constructs like `async/await` coupled with a promise-based wrapper around the `http` or `net` modules can make the code significantly easier to reason about:

```javascript
const https = require('https');

function postJSON(url, jsonData) {
    return new Promise((resolve, reject) => {
        const jsonString = JSON.stringify(jsonData);
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(jsonString)
            }
        };

        const req = https.request(url, options, (res) => {
            let responseData = '';
            res.on('data', (chunk) => responseData += chunk);
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                  try {
                       const parsed = JSON.parse(responseData);
                       resolve(parsed);
                    } catch (err) {
                      reject(err);
                     }
                } else {
                    reject(new Error(`Request failed with status code: ${res.statusCode}`));
                }
            });
            res.on('error', (err) => reject(err));
        });

        req.on('error', (err) => reject(err));
        req.write(jsonString);
        req.end();
    });
}

async function main() {
    try {
        const response1 = await postJSON('https://example.com/api/data', { id: 1, status: 'pending' });
        console.log("First Response:", response1);
        const response2 = await postJSON('https://example.com/api/log', { logMessage: 'Operation started', timestamp: Date.now() });
        console.log("Second Response:", response2);

    } catch (err) {
        console.error("Error in main:", err);
    }
}

main();
```

In this case, the `postJSON` function encapsulates the asynchronous HTTP request in a promise, which returns the response body parsed as JSON upon a successful request. This approach significantly cleans up the asynchronous logic. The `main` function uses `async/await` to sequentially send two requests, making it easier to read and maintain the code. Error handling with `try/catch` at this top level is also improved compared to scattered callbacks. The use of `https` instead of `http` here assumes secure communication is required.

For additional learning, Node.js documentation provides in-depth insights into the `net`, `http` and `https` modules. Textbooks on network programming and asynchronous patterns also offer comprehensive theoretical underpinnings to these topics. Practical exercises, such as building a simple chat server using TCP sockets or a REST client that communicates with a third-party API, provide invaluable hands-on experience to solidify the concepts. It's crucial to understand the fundamentals of networking and asynchronous programming alongside the specifics of how these Node.js modules operate for efficient and robust asynchronous JSON communication.
