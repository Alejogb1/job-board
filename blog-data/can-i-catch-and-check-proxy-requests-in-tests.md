---
title: "Can I catch and check proxy requests in tests?"
date: "2024-12-23"
id: "can-i-catch-and-check-proxy-requests-in-tests"
---

,  It's a situation I've encountered more than once, especially when dealing with complex system integrations where parts of your application rely on external services accessed via proxies. The short answer is a resounding *yes*, you can absolutely catch and check proxy requests in tests, and frankly, you *should*. It's critical for verifying the correct behavior of your software and ensuring its resilience. Let me explain how I've approached this in the past, and break down a few strategies with code examples.

From experience, the challenge isn’t just *catching* the request, it’s also verifying that the request is formatted correctly, has the proper headers, and is being sent to the correct endpoint behind the proxy. If those details are not tested effectively, you're building a fragile system.

One of the most common scenarios I've run into involves testing a system where microservices communicate through an API gateway or a service mesh acting as the proxy. In those situations, isolating the service I was testing from the actual downstream service was vital. For that, mock servers became my primary tool, particularly ones capable of simulating proxy behavior.

Here's how you'd generally approach catching and checking these proxy requests. You'll need to effectively intercept outgoing requests before they hit the actual proxy. One common method utilizes a mock http client or server, as the basis, and here's one scenario using python as example:

```python
import unittest
import requests
from unittest import mock
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
import json

class MockProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {"message": "Request captured and verified", "received_data": json.loads(post_data.decode('utf-8'))}
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def do_GET(self):
      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
      response_data = {"message": "Request captured and verified for GET"}
      self.wfile.write(json.dumps(response_data).encode('utf-8'))
        


def start_mock_server(port):
    server_address = ('localhost', port)
    httpd = HTTPServer(server_address, MockProxyHandler)
    thread = Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd

def stop_mock_server(httpd):
    httpd.shutdown()


class TestProxyRequests(unittest.TestCase):

    def setUp(self):
        self.mock_proxy_port = 9999  # using a custom port here
        self.mock_proxy_server = start_mock_server(self.mock_proxy_port)
        self.proxy_url = f'http://localhost:{self.mock_proxy_port}'

    def tearDown(self):
      stop_mock_server(self.mock_proxy_server)


    def test_post_request_via_proxy(self):

      data_to_send = {"key1": "value1", "key2": "value2"}

      response = requests.post(
        'http://fake-external-service/resource',
        json=data_to_send,
        proxies={'http': self.proxy_url, 'https': self.proxy_url}
      )

      self.assertEqual(response.status_code, 200)
      response_json = response.json()
      self.assertEqual(response_json['message'], "Request captured and verified")
      self.assertEqual(response_json['received_data'], data_to_send)

    def test_get_request_via_proxy(self):

        response = requests.get(
            'http://fake-external-service/resource',
            proxies={'http': self.proxy_url, 'https': self.proxy_url}
        )

        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json['message'], "Request captured and verified for GET")



if __name__ == '__main__':
    unittest.main()
```

In this python example, we're not just catching the request; we’re dissecting it. We start a mock server that handles both `GET` and `POST` requests and inspect their data. The tests then send requests using the `requests` library and specify our mock server as the proxy to use. This enables us to assert the correct data was forwarded through the proxy.

Another scenario I frequently encountered was using a mock client, specifically with a library like `requests-mock` or its equivalent in your respective language. This method allows you to intercept and control how requests are made using your application’s existing HTTP client. Here's a javascript example using node with Jest and `axios` along with the `axios-mock-adapter`:

```javascript
const axios = require('axios');
const MockAdapter = require('axios-mock-adapter');


describe('Proxy request testing with mock adapter', () => {
  let mock;

  beforeEach(() => {
     mock = new MockAdapter(axios);
  });

  afterEach(() => {
      mock.restore();
  });


  it('should intercept and verify POST request through proxy', async () => {

    const dataToSend = { key1: "value1", key2: "value2" };

    mock.onPost('http://proxy-host:8080/').reply(200, { message: "Request captured and verified", received_data: dataToSend });

    const response = await axios.post('http://fake-external-service/resource', dataToSend, {
      proxy: {
        host: 'proxy-host',
        port: 8080,
      },
    });

    expect(response.status).toEqual(200);
    expect(response.data.message).toEqual("Request captured and verified");
    expect(response.data.received_data).toEqual(dataToSend);
  });


    it('should intercept and verify GET request through proxy', async () => {
        mock.onGet('http://proxy-host:8080/').reply(200, { message: "Request captured and verified for GET" });

        const response = await axios.get('http://fake-external-service/resource',  {
            proxy: {
                host: 'proxy-host',
                port: 8080,
            },
        });

        expect(response.status).toEqual(200);
        expect(response.data.message).toEqual("Request captured and verified for GET");
    });
});

```

Here, instead of a mock server, `axios-mock-adapter` is used to directly mock outgoing requests made by `axios`. This avoids the overhead of running an actual mock server and is often more convenient for unit tests. Notice how we intercept the request destined for the proxy’s URL and provide a controlled response, ensuring the correct data was sent.

Finally, for environments where direct manipulation of the request pipeline isn’t easily achieved, using techniques for testing at integration level, especially if your system uses an API Gateway, it was useful to deploy a ‘test’ gateway instance. The key here is setting up the gateway such that its logs are easily retrievable and auditable during test execution.

Here's a simplified example using a shell script for demonstration purposes but in production environments you will need to use tools from your specific API Gateway vendor or logging/metrics infrastructure:

```bash
#!/bin/bash

# Assume API gateway logs to stdout or a file. This is a simplified version
# In real-world scenario, it will be more elaborate parsing of specific logging format

# Simulate application making a request through a gateway

API_GATEWAY_LOG_FILE="gateway.log"

#Function to simulate calling your application
call_application(){

  # Simulate a GET request with curl, assuming proxy configuration is handled internally
  curl -s "http://your-application/api/resource" > /dev/null 2>&1

}


#Start with a clean slate:
> $API_GATEWAY_LOG_FILE

# Call our application
call_application

# Check if a request to the correct proxy destination was logged
grep  "proxy-host:8080" $API_GATEWAY_LOG_FILE

if [ $? -eq 0 ]; then
    echo "Proxy request to proxy-host:8080 logged successfully."
else
    echo "Error: Proxy request not found in the logs or gateway not configured."
    exit 1
fi

#Clean up log file
> $API_GATEWAY_LOG_FILE

echo "Test passed!"

exit 0
```

This script is a basic example and assumes your API gateway logs requests and you can grep log files. In reality, you would likely need a more sophisticated log aggregator and parsing pipeline. The idea, however, is consistent: Verify, at the system level, that a request through the proxy to the intended destination occurred. The specifics of how this is done vary depending on your setup.

To delve further into these techniques, I recommend looking into "Testing in Microservices" by Sam Newman. It provides an excellent overview of testing various microservice patterns, including those interacting with proxies. For more specific detail on intercepting network requests, I would recommend checking out resources specific to your language’s mocking libraries like `unittest.mock` in python, `axios-mock-adapter` in JavaScript and their respective counterparts in other languages. Additionally, “Software Testing Techniques”, by Boris Beizer is a great resource on the fundamentals of software testing which includes ideas for testing communication between different parts of a software.

These strategies have been instrumental in my experience, leading to robust and dependable systems. The key is to select the method that fits your environment best and to continually verify not just the behavior of your application, but also the behavior of all the components it interacts with, which certainly includes proxies. Always aim to test what you truly intend to execute, and always be critical in the interpretation of test results.
