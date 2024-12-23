---
title: "Can I catch and check a proxy request in my tests?"
date: "2024-12-16"
id: "can-i-catch-and-check-a-proxy-request-in-my-tests"
---

, let's dive into this. Handling proxy requests in tests, it’s a scenario I’ve bumped into more than a few times, particularly when working with complex microservices and integrated systems. It's definitely doable, and it's often crucial for building robust testing suites. The short answer is, yes, you can absolutely catch and inspect proxy requests during your tests, but the how really hinges on the architecture of your application and the testing approach you’re employing.

Let's start by acknowledging that a proxy, at its core, acts as an intermediary. Your application isn't directly talking to the final service; it's chatting with the proxy first, which then forwards the request. This gives us a vital point of interception for our tests. When I first dealt with this, I was developing a system using a reverse proxy (nginx in that case) to handle routing and authentication before requests reached our backend services. The challenge was verifying that the proxy was correctly forwarding requests and modifying headers as we intended, without testing the external systems themselves. We needed to ensure the proxy logic was sound.

The first fundamental approach is to mock the proxy itself. This lets you isolate your code under test and simulate proxy behavior, without having to run a live instance of the proxy during testing. We would use this technique most commonly for unit testing individual components that rely on interacting with a proxy. In these cases, we typically use a mocking library to intercept the outbound calls that the application makes.

Here's a basic example in Python, utilizing the `unittest` framework and the `requests-mock` library (which I found invaluable for this):

```python
import unittest
import requests
import requests_mock

class MyService:
    def __init__(self, base_url):
        self.base_url = base_url

    def fetch_data(self, endpoint):
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url)
        return response.json()


class TestMyServiceWithMockProxy(unittest.TestCase):
    def test_fetch_data_with_proxy(self):
        service = MyService("http://proxy-address:8080") # Assume this is the proxy

        with requests_mock.mock() as mock_request:
           mock_request.get('http://proxy-address:8080/data', json={'key': 'value'})

           data = service.fetch_data("data")

           self.assertEqual(data, {'key': 'value'})
           request_history = mock_request.request_history
           self.assertEqual(len(request_history), 1)
           self.assertEqual(request_history[0].url, 'http://proxy-address:8080/data')
```

In this snippet, `requests_mock` intercepts any `requests.get` calls made by `MyService`. We're not hitting an actual proxy; instead, we're simulating its response and verifying that our service is interacting as expected with this mocked proxy. This approach works well for testing that your application is making the right calls, but it doesn't verify that the actual proxy is functioning.

For integration tests, we need to go a step further. The second major approach focuses on observing the actual proxy requests during integration or end-to-end tests. This involves a setup where you can either observe the requests going through the real proxy or use a tool that acts as a proxy observer. One excellent tool for that is mitmproxy, which I've used extensively in my own testing. It is a powerful, intercepting proxy that lets you watch and modify network traffic.

Here is how you can observe the proxy through `mitmproxy` :

1.  **Setup mitmproxy:** Start mitmproxy with the command `mitmproxy`.

2.  **Configure your tests to use mitmproxy:** In your tests, point the requests towards your mitmproxy address. The requests should now be going to `http://127.0.0.1:8080` by default.

3.  **Use mitmproxy client to verify request data:** You can use `mitmproxy`'s client to analyze the captured requests. Specifically, look for your targeted proxy url and confirm the request details are valid.

4. **Python Code Integration:** You can embed code to launch `mitmproxy` in your testing framework to automate this process:

```python
import subprocess
import time
import requests
import unittest

class TestIntegrationWithMitMProxy(unittest.TestCase):
    def setUp(self):
        # Start mitmproxy as a subprocess
        self.mitmproxy_process = subprocess.Popen(['mitmproxy', '--listen-port', '8081'])
        time.sleep(1) # give mitmproxy time to start

    def tearDown(self):
        # Terminate mitmproxy process
        self.mitmproxy_process.terminate()
        self.mitmproxy_process.wait()

    def test_proxy_request_captured(self):
        # Configure your application to use mitmproxy (127.0.0.1:8081 in this case)
        proxy_url = 'http://127.0.0.1:8081'

        # Application under test (replace with your actual code)
        app_url = 'http://example.com/data'
        response = requests.get(app_url, proxies={'http': proxy_url, 'https':proxy_url})

        # At this point you can manually verify the requests captured in mitmproxy client
        # Or you can use mitmdump to capture and parse the requests automatically
        # Further testing could involve launching mitmdump in your test to analyze the requests automatically


        self.assertTrue(response.status_code != 0)  # Dummy assertion, more thorough analysis can be done.
```

This approach provides an end-to-end test, making sure that both your application and the proxy behave as expected. While `mitmproxy` isn’t purely a testing tool, it’s invaluable for understanding the actual network requests, especially when your system involves more than just basic API interactions.

Finally, there’s a technique I’ve applied when the proxy itself is a component within your application or testing infrastructure, often the case for custom, in-house proxy logic. This often involves creating a simplified version of your proxy for test cases. This can be a component written in the same language as your tests that logs or verifies all requests received, but without actually forwarding them. This allows you to test your core code and also verify the proxy layer in a simplified, controlled environment. For example:

```python
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import requests
import json

class MockProxyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = json.dumps({'message': 'Mock proxy response'})
        self.wfile.write(response_data.encode())

    def log_message(self, format, *args):
        self.request_log = format % args

class MyService:
    def __init__(self, base_url):
        self.base_url = base_url

    def fetch_data(self, endpoint):
       url = f"{self.base_url}/{endpoint}"
       response = requests.get(url)
       return response.json()

class TestMyServiceWithCustomProxy(unittest.TestCase):
    def setUp(self):
        # Start a Mock proxy in a separate thread
        self.server_address = ('localhost', 8082)
        self.httpd = HTTPServer(self.server_address, MockProxyServer)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.server_thread.join()

    def test_fetch_data_with_custom_proxy(self):
        service = MyService("http://localhost:8082")
        data = service.fetch_data("data")
        self.assertEqual(data, {'message': 'Mock proxy response'})
        self.assertIn('/data',self.httpd.RequestHandlerClass.request_log)
```

Here, the `MockProxyServer` does not act as a full fledged proxy, but it allows your test to inspect and verify the incoming requests. It’s lightweight and gives you direct access to the requests received without using external tools.

To summarize, catching and checking proxy requests in your tests is definitely achievable, and the optimal approach depends largely on the level of integration you’re aiming for. Unit tests can benefit from mocked proxy responses, integration tests often need tools like `mitmproxy` to observe real network traffic, and component level tests might work well with a simple mock server. For further insight, I'd recommend looking at "Testing Python" by Brian Okken for best practices in the python testing environment and the documentation for `mitmproxy`. Understanding these techniques will allow you to write comprehensive tests that make sure your applications interact with proxies correctly.
