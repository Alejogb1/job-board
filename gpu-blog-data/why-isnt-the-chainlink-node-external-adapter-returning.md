---
title: "Why isn't the Chainlink node external adapter returning data?"
date: "2025-01-30"
id: "why-isnt-the-chainlink-node-external-adapter-returning"
---
Okay, here’s my take on troubleshooting a Chainlink external adapter not returning data, based on common issues I've encountered in practice.

The core problem when a Chainlink external adapter fails to return data typically stems from a disconnect between how the Chainlink node expects to receive data and how the adapter is configured to provide it. These failures manifest not as outright crashes, but rather as the node receiving no response or a malformed one, leading to job failures. It's essential to remember that the Chainlink node does not directly "understand" or execute the logic within your external adapter; it relies on well-defined API contracts for data exchange.

The data flow is as follows: a Chainlink job initiates a request, the node forwards this request to the external adapter’s specified endpoint, the adapter processes the request, performs any necessary logic (e.g., fetching from an external API), and then returns a properly formatted JSON response. If any step in this process goes wrong, data won’t flow back to the Chainlink node. We can focus on three primary areas: the adapter’s logic, the node’s configuration, and the network connectivity between them.

**1. Adapter Logic and Response Formatting**

One of the most common causes of external adapter failure is incorrect response formatting. The Chainlink node expects a JSON object with specific keys, primarily `data`, to contain the actual value it should consume. Any deviation from this format, including missing keys or incorrect data types, can result in the node ignoring the response. My experience shows this is especially prevalent when developers assume the response structure is flexible, rather than adhering to a strict specification.

Here's a typical example demonstrating the problem:

```python
# Incorrect Adapter Response (Python)
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/', methods=['POST'])
def adapter_endpoint():
    req = request.get_json()
    external_api_url = "https://api.example.com/data" # Hypothetical endpoint
    try:
        response = requests.get(external_api_url).json()
        return jsonify(response) # Improper response formatting
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

In this example, the adapter retrieves data from an external API and directly returns the API's JSON response, without encapsulating it inside a `data` field. If the external API returns something like `{"price": 123.45}`, the Chainlink node wouldn't recognize it as a valid response.

To remedy this, the response must be formatted correctly:

```python
# Correct Adapter Response (Python)
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/', methods=['POST'])
def adapter_endpoint():
    req = request.get_json()
    external_api_url = "https://api.example.com/data" # Hypothetical endpoint
    try:
        response = requests.get(external_api_url).json()
        return jsonify({"data": response}) # Correct response formatting
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

Here, I've wrapped the external API's response inside a dictionary with the key `data`. This ensures the Chainlink node can properly extract the relevant value. This seemingly minor change has resolved numerous integration issues across various projects. The key takeaway here is always to verify the format of your JSON response against the Chainlink node's requirements, focusing on the presence of the `data` key.

**2. Node Configuration and Job Specifications**

Beyond the adapter itself, misconfigurations within the Chainlink node or its job specifications frequently contribute to data retrieval issues. Specifically, issues arise in the `bridges` configuration within the Chainlink node configuration file and the job's `initiators` and `tasks`. The bridge specification must correctly define the URL endpoint of the external adapter and any necessary authentication parameters, if required. Furthermore, the job's task chain must specify the correct path in the JSON response to the desired data point.

Assume a job configuration defined as follows:

```json
{
  "initiators": [
    {
      "type": "runlog",
      "params": { }
    }
  ],
  "tasks": [
      {
          "type": "bridge",
           "params": {
               "name": "my_adapter"
           }
      },
        {
            "type": "jsonparse",
            "params": {
                "path": ["price"]
          }
       },
     {
        "type": "multiply",
         "params": {
            "times": 100
         }
       },
    {
      "type": "ethint256",
     }
   ]
}
```

This job specification assumes that the external adapter will return a response where the desired value can be accessed via `["price"]`. If, however, the adapter responds with `{"data": {"price": 123.45}}`, then the task `jsonparse` will fail since it is targeting the wrong key. The correct path for this response would be `["data", "price"]`. To resolve this we would need to alter the task definition accordingly:

```json
{
  "initiators": [
    {
      "type": "runlog",
      "params": { }
    }
  ],
  "tasks": [
      {
          "type": "bridge",
           "params": {
               "name": "my_adapter"
           }
      },
        {
            "type": "jsonparse",
            "params": {
                "path": ["data", "price"]
          }
       },
     {
        "type": "multiply",
         "params": {
            "times": 100
         }
       },
    {
      "type": "ethint256",
     }
   ]
}
```

By updating the path in the `jsonparse` task, the Chainlink node will now extract the value from the correct location within the response. Another typical problem is not having the bridge definition correctly associated with a job specification or that bridge failing due to not being properly configured (e.g. incorrect adapter URL). This illustrates the importance of meticulously tracing the data path through the Chainlink node configuration and job specifications to verify that each step correctly identifies and extracts the required information.

**3. Network Connectivity and Security**

Network related issues can create invisible but significant obstacles to data retrieval from external adapters. I have seen instances where adapters deployed on internal networks lack visibility to the Chainlink node, especially when the node is running on a separate network or in a container. Firewall rules, security configurations, and incorrect adapter URL specifications in the bridge configuration are all potential culprits.

Here’s a minimal example to illustrate a typical bridge configuration within a Chainlink node config. The adapter URL is set incorrectly:

```json
"bridges": [
    {
      "name": "my_adapter",
      "url": "http://localhost:8081",  // Incorrect URL example
      "requestTimeout": "10s"
    }
  ]
```

In this instance, the specified adapter URL is `http://localhost:8081`, whereas the external adapter is actually listening on `http://localhost:8080`. Consequently, the Chainlink node will not be able to reach the adapter, leading to a timeout and data retrieval failure. This problem is resolved simply by correcting the URL:

```json
"bridges": [
    {
      "name": "my_adapter",
      "url": "http://localhost:8080",  // Corrected URL
      "requestTimeout": "10s"
    }
  ]
```

This highlights that carefully verifying network accessibility between your Chainlink node and external adapter is crucial. This process will involve testing connectivity via standard utilities such as `curl` and `ping` from the Chainlink node's environment to the adapter’s designated endpoint. For containerized setups, confirming that Docker networking is correctly configured is also paramount.

In summary, troubleshooting a Chainlink external adapter's failure to return data requires a systematic approach. It is imperative to begin with a careful examination of the adapter's response formatting, ensuring strict compliance with Chainlink’s expectations. Subsequently, attention must be given to the Chainlink node configuration and job specification, meticulously mapping the data path through the tasks and any dependencies. Network accessibility between the node and adapter is an important consideration, requiring testing and careful configuration. I would highly recommend reviewing Chainlink’s documentation regarding bridge specifications and task parameters. Additionally, consulting with community forums and documentation concerning networking for containers can prove beneficial.
