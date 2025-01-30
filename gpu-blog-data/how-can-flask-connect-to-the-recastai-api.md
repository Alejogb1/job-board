---
title: "How can Flask connect to the recastai API?"
date: "2025-01-30"
id: "how-can-flask-connect-to-the-recastai-api"
---
Connecting a Flask application to the Recast.AI (now known as SAP Conversational AI) API involves several key steps focused on establishing secure communication and correctly formatting requests and responses. I've implemented this integration in a few projects, and the core process remains consistent: you manage authentication, construct API calls, and then handle the returned data within your Flask routes.

**Explanation**

The SAP Conversational AI API, accessed via HTTP, uses a token-based authentication system. Therefore, before sending any requests, you need to secure an application token, either through your account or a bot-specific token if you intend to communicate with a specific bot instance. This token acts as an access credential, demonstrating that your Flask application is authorized to interact with the service. After obtaining the token, you’ll incorporate it into each request’s header. Typically, this is achieved with the `Authorization` header using a 'Token' scheme.

Interaction with the API typically centers on two core functions: sending text queries and handling the parsed intent and entity responses. The process begins when the user sends a text-based query to the Flask application; this query is subsequently formatted into a JSON payload and submitted to the SAP Conversational AI API's `/request` endpoint. The API's NLP engine then processes this query and returns a JSON structure containing identified intents, entities, confidence scores, and other relevant context. Your Flask application must then parse this JSON response to extract valuable data for responding to the user or performing further actions in your application's backend.

To achieve this process smoothly, you'll use Flask's ability to accept requests (usually through a `/webhook` route) and then employ the `requests` Python library (or a similar library) to make HTTP calls to the SAP Conversational AI API. It is essential to ensure that you catch errors gracefully during this process, providing informative feedback to both your user and your logs should a network or API issue arise. The integration further relies on a clear understanding of the SAP Conversational AI API documentation. Understanding the specific fields and types within the JSON response is crucial for correct extraction of intent data, and also helps your application perform actions based on user input. Furthermore, it's important to recognize that, since the SAP Conversational AI's API may change over time, a maintenance plan is needed.

**Code Examples**

Below are three code examples that demonstrate different aspects of the integration: authentication, sending a text request, and handling responses.

**Example 1: Authentication Setup and Initial Request**

This snippet focuses on establishing the authentication credentials, then making a simple test request to the API.

```python
import requests
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Replace with your SAP Conversational AI token
CONVERSATIONAL_AI_TOKEN = "YOUR_SAP_CONVERSATIONAL_AI_TOKEN"
API_ENDPOINT = "https://api.recast.ai/v2/request"

def query_conversational_ai(text):
    headers = {
        "Authorization": f"Token {CONVERSATIONAL_AI_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {"text": text}
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=data)
        response.raise_for_status() # Raises an exception for 4XX or 5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None

@app.route("/test_api", methods=["POST"])
def test_api_route():
    data = request.get_json()
    if 'message' not in data:
        return jsonify({'error': 'Message parameter is missing.'}), 400
    user_message = data.get('message')
    api_response = query_conversational_ai(user_message)

    if api_response:
      return jsonify(api_response)
    else:
      return jsonify({"error":"API request failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

*   **Commentary:** This snippet defines the core function `query_conversational_ai()`, which constructs the API request headers and the JSON body with the text query. Error handling is incorporated with `try-except` block to catch network issues with the `requests` library.  The `response.raise_for_status()`  line is very important as it checks the HTTP code and raises an exception if it is an error code. The `test_api_route()` shows how to integrate the function to your Flask application as a POST request handler for a route. Note the importance of setting the content type header for JSON requests.

**Example 2: Extracting Intents and Entities**

Here’s an example of how to process the JSON response to extract key information.

```python
def process_api_response(api_response):
    if not api_response:
        return None #Handle empty response gracefully

    try:
      intent = api_response['results']['intents'][0]['slug'] if api_response['results']['intents'] else None
      entities = api_response['results']['entities']
      return intent, entities
    except KeyError as e:
      print(f"Key error during response parsing:{e}")
      return None, None


@app.route("/process_message", methods=["POST"])
def process_message_route():
  data = request.get_json()
  if 'message' not in data:
        return jsonify({'error': 'Message parameter is missing.'}), 400
  user_message = data.get('message')
  api_response = query_conversational_ai(user_message)

  if api_response:
      intent, entities = process_api_response(api_response)
      if intent:
        return jsonify({"intent":intent,"entities":entities})
      else:
         return jsonify({"error": "Could not parse intent information from response"}), 500
  else:
      return jsonify({"error":"API request failed"}), 500
```

*   **Commentary:** This snippet features the `process_api_response()` function to parse the JSON response. It specifically targets the intent and entity data. Note the check if the intents list is present to avoid errors if intents are not found in the response. Error handling for missing keys is added to prevent exceptions. This demonstrates retrieving the most confident intent and all available entities. This is integrated into a different flask route `/process_message`.

**Example 3: Error Handling and Logging**

This example integrates logging and enhanced error handling.

```python
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def query_conversational_ai(text):
    headers = {
        "Authorization": f"Token {CONVERSATIONAL_AI_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {"text": text}
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during API request: {e}") #Log errors
        return None

def process_api_response(api_response):
   if not api_response:
        logging.error("Empty API response received")
        return None, None
   try:
        intent = api_response['results']['intents'][0]['slug'] if api_response['results']['intents'] else None
        entities = api_response['results']['entities']
        return intent, entities
   except KeyError as e:
        logging.error(f"Key error during response parsing: {e}")
        return None, None

@app.route("/process_with_log", methods=["POST"])
def process_with_log_route():
   data = request.get_json()
   if 'message' not in data:
       logging.warning("Message parameter missing")
       return jsonify({'error': 'Message parameter is missing.'}), 400
   user_message = data.get('message')
   api_response = query_conversational_ai(user_message)

   if api_response:
      intent, entities = process_api_response(api_response)
      if intent:
        return jsonify({"intent":intent,"entities":entities})
      else:
        logging.warning("Intent not found or could not be parsed.")
        return jsonify({"error": "Could not parse intent information from response"}), 500
   else:
      return jsonify({"error":"API request failed"}), 500
```

*   **Commentary:** The `logging` library is initialized. In both the `query_conversational_ai` and `process_api_response` functions, error conditions, like failed API calls and key errors, are logged using `logging.error()`. Furthermore, warning messages are logged when unexpected scenarios happen such as missing message data from the request or an intent not being found, providing more details than just returning an HTTP code. This improves traceability and debugging capabilities. This code has been incorporated into another endpoint, `/process_with_log`.

**Resource Recommendations**

To further your understanding, I would suggest reviewing the official SAP Conversational AI documentation, which includes extensive explanations of the API's functionalities and data structures, and the `requests` library's documentation, which covers HTTP handling effectively. Additionally, you should explore tutorials on Flask application development to grasp the underlying concepts of routes, requests and responses, to structure your application. Understanding HTTP protocols and status codes is also crucial for proper error handling during integration of APIs like the one used. Finally, studying design patterns when using external APIs should also be considered for robust application development.
