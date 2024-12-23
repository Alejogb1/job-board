---
title: "Why can't I create HubSpot contacts using the hubspot-api-client?"
date: "2024-12-23"
id: "why-cant-i-create-hubspot-contacts-using-the-hubspot-api-client"
---

Let's tackle this. So, you're running into a frustrating roadblock trying to create HubSpot contacts via the `hubspot-api-client`. I've been there—more times than I'd care to remember, actually. It's rarely a single, clear-cut issue. More often than not, it's a confluence of several things conspiring against you. Let's break down the most common culprits, drawing from experiences battling similar situations in past projects.

The first place to scrutinize is your authentication method. The `hubspot-api-client` requires either an api key or, more robustly, an access token obtained through oAuth. If you're using an api key, be certain it is associated with a portal that has the proper permissions to create contacts. While it seems fundamental, I’ve witnessed more than one colleague spend hours tracing errors stemming from a simple, invalid api key. On the other hand, oAuth offers better security and flexibility, but its complexity also means it introduces more possible points of failure. Have you correctly configured the oAuth flow and is your access token actually active, or has it expired? These tokens generally have a lifetime, so refreshing them is essential, especially in long-running applications.

Next, let’s examine the data you're sending. The HubSpot api expects a specific format, generally json, and this includes a defined structure for contact properties. Mismatches here will result in errors. Are you using the proper property names (remember HubSpot's case sensitivity)? Are you sending required fields? HubSpot’s api documentation outlines which properties are mandatory, and missing these will invariably prevent contact creation. Also, check your data types. Passing a string where an integer is expected, or attempting to send a date that isn't formatted as an iso string, can all cause problems. I recall spending a particularly long evening debugging an issue where a seemingly identical date format was subtly causing errors—it turned out the api was strict about milliseconds, which my code wasn't always including.

Then, there’s rate limiting. HubSpot enforces strict rate limits to maintain the stability of their platform. If you’re making too many requests in a short period, the api will respond with a 429 error, indicating you’ve been temporarily throttled. This is particularly relevant in batch operations or if your code contains loops that are making api requests without proper delays. The error messaging from the HubSpot api usually provides some insight, so read them carefully. It is a good practice to incorporate a retry mechanism with exponential backoff to deal with rate limits.

Lastly, it’s important to make sure you are properly handling any errors that are returned. The `hubspot-api-client` should raise exceptions when something goes wrong, but catching and correctly interpreting them is vital. You might also find it helpful to examine the raw http requests and responses if the exceptions aren't providing enough information, this can expose subtle differences in what you intended to send and what the api actually received.

To illustrate these points more concretely, let’s consider a few code examples using Python and the official `hubspot-api-client`.

**Example 1: Basic contact creation (assuming authentication is correctly set up)**

```python
from hubspot import HubSpot
from hubspot.crm.contacts import SimplePublicObjectInput

# Initialize the client (replace with your access token or api key)
client = HubSpot(access_token="your_access_token")

contact_properties = {
    "email": "test@example.com",
    "firstname": "Test",
    "lastname": "User"
}

simple_input = SimplePublicObjectInput(properties=contact_properties)

try:
    api_response = client.crm.contacts.basic_api.create(simple_public_object_input=simple_input)
    print(f"Contact created with ID: {api_response.id}")
except Exception as e:
    print(f"Error creating contact: {e}")

```

This example showcases the fundamental process of contact creation. Note that all required contact properties (email, firstname, and lastname) are included, each with appropriate string data. If you ran this and encountered an error, double-check that your `access_token` is valid and you have properly authenticated with HubSpot.

**Example 2: Handling rate limits using exponential backoff**

```python
import time
from hubspot import HubSpot
from hubspot.crm.contacts import SimplePublicObjectInput
from requests.exceptions import HTTPError

client = HubSpot(access_token="your_access_token")

contact_data = [
   {"email": f"test{i}@example.com", "firstname": f"Test{i}", "lastname": "User"} for i in range(5)
]


def create_contact(contact_properties):
  simple_input = SimplePublicObjectInput(properties=contact_properties)
  attempts = 0
  delay = 1  # Initial delay in seconds

  while attempts < 5:
      try:
          api_response = client.crm.contacts.basic_api.create(simple_public_object_input=simple_input)
          print(f"Contact created with ID: {api_response.id}")
          return api_response
      except HTTPError as e:
          if e.response.status_code == 429:
              attempts += 1
              print(f"Rate limit encountered, retrying in {delay} seconds...")
              time.sleep(delay)
              delay *= 2 # exponential increase of delay
          else:
             print(f"Error creating contact: {e}")
             return None
  print("Max retries exceeded, unable to create contact.")
  return None

for contact in contact_data:
    create_contact(contact)

```

This example demonstrates a basic retry mechanism for dealing with rate limits. If the api returns a 429 error (too many requests), it waits for an increasing amount of time before retrying the request. This ensures a more stable application when handling multiple requests. This prevents the code from crashing and gives you a chance to create all of your contacts.

**Example 3: Specifying non-standard property types, including date and multiple-choice**

```python
from datetime import datetime, timezone
from hubspot import HubSpot
from hubspot.crm.contacts import SimplePublicObjectInput

client = HubSpot(access_token="your_access_token")


contact_properties = {
    "email": "test_special@example.com",
    "firstname": "Test",
    "lastname": "Special",
    "date_of_birth": datetime(1990, 5, 15, tzinfo=timezone.utc).isoformat(),
    "lifecycle_stage": "customer" # Assuming "customer" is a valid choice for lifecycle_stage in your portal
}

simple_input = SimplePublicObjectInput(properties=contact_properties)

try:
    api_response = client.crm.contacts.basic_api.create(simple_public_object_input=simple_input)
    print(f"Contact created with ID: {api_response.id}")
except Exception as e:
    print(f"Error creating contact: {e}")
```

Here, we see a more complex example including a date of birth and a lifecycle stage. This requires careful attention to the data types: the date must be formatted as an iso string and the choice should exist as an existing option in HubSpot.

For further exploration, I'd recommend diving into the official HubSpot api documentation, which is very detailed, including specifics about data types and allowed values. A deep understanding of this is crucial. Additionally, the book "Building Microservices" by Sam Newman can provide useful guidance on designing resilient systems that handle api interactions effectively, especially in the context of retry mechanisms and rate limiting. For a solid foundation on http, I’d suggest "HTTP: The Definitive Guide" by David Gourley and Brian Totty, it provides comprehensive knowledge of request and response structure, which will be immensely helpful when troubleshooting api errors.

In conclusion, remember to methodically review your authentication, validate the data you're sending against the api documentation, implement robust error handling and be prepared to manage rate limits. This systematic approach will help you isolate and resolve the problem, allowing you to create HubSpot contacts with the `hubspot-api-client` effectively.
