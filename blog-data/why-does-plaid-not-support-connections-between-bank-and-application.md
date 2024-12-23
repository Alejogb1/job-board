---
title: "Why does Plaid not support connections between 'Bank' and 'Application'?"
date: "2024-12-23"
id: "why-does-plaid-not-support-connections-between-bank-and-application"
---

Let's tackle this. The absence of a direct connection between a specific bank and a financial application using Plaid is usually less about a deliberate snub and more about a complex interplay of technical, legal, and business considerations. I've encountered this scenario numerous times over the years, often needing to troubleshoot similar situations in fintech projects. Here's a breakdown of why these situations happen, drawing from my past experience and some of the common patterns I've seen.

Primarily, it comes down to a few core reasons, which aren't always transparent to the end-user. It's important to understand that Plaid, as a financial data aggregator, doesn't inherently support every single financial institution out there. Instead, they need to actively build and maintain integrations. This involves establishing protocols for secure data transfer and constant updates to accommodate changes on the bank's side, which can be a very resource-intensive process. Let's explore the major factors in detail:

First, **the institution's technical implementation** is paramount. Banks utilize a variety of internal systems for user authentication and data representation. If a bank's API doesn't conform to common standards, or if it lacks a well-documented API entirely, Plaid needs to dedicate significant engineering resources to create a bespoke integration. Sometimes, banks use older systems that are difficult to interface with reliably. In these cases, maintaining stability and security becomes incredibly challenging, and Plaid may decide against building an integration altogether or postpone it until the bank implements more modern technologies. Banks often prioritize their core banking functions over exposing public-facing APIs.

Second, **data aggregation is regulated territory**. Not every bank is amenable to having its customer data funneled through a third-party platform like Plaid. Legal hurdles, data privacy concerns, and the perceived risk of security breaches often lead some institutions to restrict access to their data. This can manifest as a bank actively blocking Plaid's requests, either intentionally or by imposing strict rate limits or security measures that make reliable data collection impossible. Compliance requirements, such as GDPR, CCPA and other region specific regulations, also affect this deeply, requiring financial applications to ensure that they process user data responsibly. Often banks haven’t taken the steps to be considered compliant or aren’t keen on opening themselves to such risks.

Third, **it's also about economics and priorities**. Plaid, like any for-profit company, must weigh the cost of building and maintaining an integration against the potential benefits. If a specific bank has a very small user base, or if the bank is unresponsive to attempts at establishing an integration partnership, Plaid may deprioritize it in favour of more impactful integrations. They have to focus on which banks cover the most substantial portion of the market and which collaborations are practically feasible. Sometimes it is also a matter of bandwidth—the engineering teams have a finite capacity to build and support integrations, leading to a prioritization of which integrations are completed.

Fourth, **security considerations always play a crucial role**. Aggregating financial data comes with enormous responsibility. If Plaid detects vulnerabilities within the bank's API or if the authentication methods are deemed unsafe or unstable, they won't risk their users’ financial security. If a bank uses older technology without robust protection against common attacks such as man-in-the-middle (MITM) attacks, this can be a showstopper. In my past experience, I recall dealing with instances where even if the bank had a functional API, we simply couldn’t rely on it for security reasons which in turn lead to abandoning the integration attempt with that particular institution.

To understand this better, let’s look at some working code snippets, demonstrating how the complexities I've described would translate into code and how such issues might be handled at the application level. These are simplified examples, but they capture the essence:

**Snippet 1: Checking for Institutional Support**

This pseudo-code example demonstrates how we might check whether a given institution is supported by a fictional data aggregation service (similar to Plaid) before attempting a connection. This is important to not waste the user's time and resources trying a connection that will ultimately fail.

```python
class DataAggregator:
    def __init__(self, supported_institutions):
        self.supported_institutions = supported_institutions

    def is_supported(self, institution_id):
        return institution_id in self.supported_institutions

    def connect(self, institution_id, user_credentials):
        if not self.is_supported(institution_id):
            raise InstitutionNotSupportedError(f"Institution {institution_id} is not currently supported.")
        # ... actual connection logic would go here ...
        return {"success": True, "data": "account_details"} # Placeholder response

# Example usage
supported_banks = ["bank_of_example", "credit_union_alpha"]
aggregator = DataAggregator(supported_banks)

try:
    bank_id = "bank_of_example" # Example of a supported bank
    result = aggregator.connect(bank_id, {"username": "user123", "password": "securepassword"})
    print(f"Connection successful. Details: {result['data']}")

    unsupported_bank_id = "bank_of_unsupported" # Example of a non-supported bank
    result = aggregator.connect(unsupported_bank_id, {"username": "user123", "password": "securepassword"})
except InstitutionNotSupportedError as e:
    print(f"Error: {e}")
```

**Snippet 2: Handling API Errors**

This demonstrates how an integration layer can handle common errors arising from a bank's API—these need to be handled gracefully and communicate this failure back to the application.

```python
import requests

class BankAPIClient:
    def __init__(self, api_base_url):
        self.api_base_url = api_base_url

    def get_account_data(self, user_id, user_credentials):
        try:
            response = requests.post(f"{self.api_base_url}/accounts", json={
                "user_id": user_id,
                "username": user_credentials["username"],
                "password": user_credentials["password"]
            })
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid credentials.")
            elif e.response.status_code == 403:
                raise PermissionDeniedError("Permission denied by the bank API.")
            elif e.response.status_code == 500:
                 raise BankServerError("Server error by the bank API")
            else:
                raise APIError(f"An unexpected error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Network error: {e}")

class AuthenticationError(Exception):
    pass
class PermissionDeniedError(Exception):
    pass
class APIError(Exception):
    pass
class BankServerError(Exception):
  pass

# Example usage
client = BankAPIClient("https://api.bankofexample.com")

try:
    user_credentials = {"username": "testuser", "password": "incorrectpassword"}
    account_data = client.get_account_data("user123", user_credentials)
    print(f"Account data: {account_data}")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except PermissionDeniedError as e:
  print(f"Permission denied by the bank API: {e}")
except BankServerError as e:
  print(f"Bank server error: {e}")
except APIError as e:
    print(f"API error: {e}")
```

**Snippet 3: Mapping and Normalizing Data**

This shows how data received from different banks needs to be normalized before being consumed by an application. Different institutions use different conventions for describing financial data, and this must be translated into a common format.

```python
class DataNormalizer:
    def __init__(self, bank_data_schemas):
        self.bank_data_schemas = bank_data_schemas

    def normalize_data(self, bank_id, raw_data):
        if bank_id not in self.bank_data_schemas:
           raise UnsupportedBankSchema(f"No schema available for {bank_id}")

        schema = self.bank_data_schemas[bank_id]
        normalized_data = {}

        for normalized_key, bank_key in schema.items():
          if bank_key in raw_data:
             normalized_data[normalized_key] = raw_data[bank_key]
          else:
            normalized_data[normalized_key] = None # Or a default value

        return normalized_data

class UnsupportedBankSchema(Exception):
    pass

# Example
schemas = {
    "bank_of_example": {"account_balance": "currentBalance", "account_number": "accountNo"},
    "credit_union_alpha": {"account_balance": "balance", "account_number": "acctNum"}
}

normalizer = DataNormalizer(schemas)
bank_id = "bank_of_example"
raw_bank_data = {"currentBalance": 1234.56, "accountNo": "1234567890"}

try:
  normalized_data = normalizer.normalize_data(bank_id, raw_bank_data)
  print(f"Normalized data: {normalized_data}")

  unsupported_bank_id = "unsupported_bank"
  normalized_data = normalizer.normalize_data(unsupported_bank_id, raw_bank_data)
except UnsupportedBankSchema as e:
  print(f"Error: {e}")

```

These code examples demonstrate the complexities involved. Real-world implementations are considerably more intricate, involving extensive error handling, security measures, and data transformations, all of which contribute to why some banks aren't readily available via Plaid.

For further study, I recommend delving into the official Plaid API documentation. "Building Financial Applications with Plaid" by Adam L. Wood is a great introductory book, and for more technical detail, "Designing Data-Intensive Applications" by Martin Kleppmann provides an excellent background on designing resilient systems. Additionally, staying updated on financial data privacy regulations is essential. Reading papers on OAuth 2.0, and understanding how banking APIs are typically architected also provides valuable insight into how systems like Plaid function.

In conclusion, the absence of a Plaid integration for a specific bank is not typically due to simple oversight but rather a result of a complex set of interrelated factors involving technical challenges, legal and regulatory considerations, strategic business decisions, and security concerns. Understanding these reasons helps provide a more realistic expectation of what can and cannot be achieved with financial data aggregation.
