---
title: "How can Airflow access NiFi APIs using a Keycloak OpenID token?"
date: "2024-12-23"
id: "how-can-airflow-access-nifi-apis-using-a-keycloak-openid-token"
---

Let's tackle this one, shall we? I've actually had to architect a very similar pipeline involving airflow and nifi, both secured by keycloak, back when I was working on a data ingestion project for a multinational retail group. The challenge, as you’re hinting, is not merely making the api call; it’s doing so securely with the right authorization flow. It requires a delicate dance between python, airflow's extensibility, and the specific authentication protocols of nifi and keycloak. Let's break down how this can be achieved, and, in particular, how you should handle the authorization token.

The central issue here is authentication: ensuring airflow, acting as a client, can successfully interact with nifi apis, which are protected by keycloak. We're not talking about simple username/password authentication. We're dealing with the more robust oidc flow using keycloak to provide access tokens. The process, at a high level, consists of these steps:

1.  **Obtain a Keycloak Access Token:** This is the most critical step. Airflow needs to request and receive a valid keycloak access token. This usually involves client credentials grant or authorization code grant flows, but for service-to-service interactions like this, we mostly utilize client credentials grant. The token provides proof that airflow is authorized to access resources in nifi.
2.  **Include the Token in Nifi API Requests:** Once you have the token, it needs to be included in each request to the nifi api as an authorization header. Nifi will then validate the token through keycloak (or by its own validation mechanisms), granting access if valid.
3.  **Token Management:** Tokens expire. Your airflow implementation needs to periodically refresh the token to ensure continuous access to the nifi api. This part also includes the handling of failures to obtain the token and retries.

Now, let’s dive into some practical code examples, which i’ve adapted from how we initially set this up in the data ingestion project mentioned earlier. I've abstracted away the company specific aspects to provide clear and usable snippets:

**Example 1: Obtaining the access token using client credentials flow**

This example uses the `requests` library to directly interact with the keycloak token endpoint. It also makes use of `json` for formatting the response. You'll need the client id, client secret, and the keycloak token endpoint configured in your airflow environment variables or a secure configuration management system. Consider setting these as airflow variables to prevent hardcoding them in the code.

```python
import requests
import json
import os
from airflow.exceptions import AirflowException

def get_keycloak_token():
    token_endpoint = os.getenv("KEYCLOAK_TOKEN_ENDPOINT")
    client_id = os.getenv("KEYCLOAK_CLIENT_ID")
    client_secret = os.getenv("KEYCLOAK_CLIENT_SECRET")

    if not all([token_endpoint, client_id, client_secret]):
       raise AirflowException("Missing Keycloak credentials or token endpoint.")


    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    try:
        response = requests.post(token_endpoint, data=payload)
        response.raise_for_status() # raise exception for bad status codes (4xx, 5xx)
        json_response = json.loads(response.text)
        return json_response.get("access_token")
    except requests.exceptions.RequestException as e:
        raise AirflowException(f"Failed to obtain keycloak token: {e}")

#Example of usage:
# access_token = get_keycloak_token()
# if access_token:
#    print(f"Token acquired: {access_token[:20]}...")
# else:
#   print("Failed to acquire token")
```

This function encapsulates the logic to acquire a token, making your airflow tasks cleaner. Error handling here is crucial; an inability to obtain a token should be logged and can cause the task to fail preventing downstream issues. In our implementation, we configured airflow to retry failed token acquisition attempts, and raised an error after a certain number of failures, with appropriate alerts to operations teams.

**Example 2: Making an authenticated Nifi API request**

This snippet shows how to use the access token to make a request to the nifi api. It utilizes the `requests` library and demonstrates the use of the `Authorization` header. Make sure your Nifi url and the specific endpoint are also properly configured and retrievable by the airflow tasks.

```python
import requests
import os
from airflow.exceptions import AirflowException

def trigger_nifi_flow(nifi_url, nifi_endpoint, access_token, payload):

    if not all([nifi_url, nifi_endpoint, access_token]):
        raise AirflowException("Missing nifi parameters or access token.")

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    try:
         response = requests.post(f"{nifi_url}{nifi_endpoint}", headers=headers, json=payload)
         response.raise_for_status() # raise exception for bad status codes (4xx, 5xx)
         return response.json()
    except requests.exceptions.RequestException as e:
        raise AirflowException(f"Failed to trigger nifi flow: {e}")
    except ValueError as e:
         raise AirflowException(f"Nifi response was not valid json: {e}")

#Example of usage
#nifi_url = os.getenv("NIFI_URL")
#nifi_endpoint = "/process-groups/root/processors"
#payload = {
#  "component": {
#     "state": "RUNNING"
#    }
#}
#try:
#     nifi_response = trigger_nifi_flow(nifi_url, nifi_endpoint, get_keycloak_token(), payload)
#     print(f"Nifi response: {nifi_response}")
#except AirflowException as e:
#    print(f"Error: {e}")
```

This function is designed to encapsulate the logic for interacting with nifi, making the airflow task simpler to maintain.

**Example 3: Integrating with airflow DAG**

This final snippet demonstrates how to integrate the previous functions into an airflow dag using the `PythonOperator`. This setup ensures that the token is fetched and passed to the task responsible for triggering the nifi flow.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

def trigger_nifi_task():
    access_token = get_keycloak_token()
    nifi_url = os.getenv("NIFI_URL")
    nifi_endpoint = "/process-groups/root/processors"
    payload = {
       "component": {
          "state": "RUNNING"
      }
    }

    if access_token:
       trigger_nifi_flow(nifi_url, nifi_endpoint, access_token, payload)

with DAG(
    dag_id='nifi_keycloak_integration',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=['nifi', 'keycloak'],
) as dag:
    trigger_nifi_flow_task = PythonOperator(
        task_id='trigger_nifi_flow',
        python_callable=trigger_nifi_task,
        retries=2, # Allow retries for temporary failures
    )
```

This basic dag shows how to use a python operator to chain the two core functions in order to authenticate with keycloak and subsequently trigger a flow in nifi. The retries attribute on the python operator helps us in dealing with transient failures in obtaining a token or interacting with nifi's API. Note that for production-ready deployments, you should consider using the `airflow.utils.trigger_rule.TriggerRule` to handle situations where the prior step may fail, and consider integrating a retry mechanism inside the functions in case of networking errors.

In terms of additional reading, i'd highly recommend the following:

*   **O'Reilly's “OAuth 2 in Action” by Justin Richer and Antonio Sanso:** This will provide a thorough understanding of OAuth 2.0, a critical piece of the puzzle for this architecture.
*   **The official Keycloak documentation:** This is invaluable for understanding the specifics of keycloak's configuration and endpoints. Pay attention to the section related to token endpoints and client configuration.
*   **The Apache Nifi documentation:** Understanding the Nifi API is, of course, vital. The documentation describes the various rest endpoints that can be interacted with.
*  **"Python for Data Analysis" by Wes McKinney:** This is more general, but it will provide foundational knowledge for writing effective python scripts for data processing, a core competency for any airflow developer.

Remember, security is not an add-on; it's a critical component to bake into your system. Always be cognizant of where you are storing your sensitive credentials, use secure connections (https), and ensure that access tokens are handled appropriately. Token rotation, rate limiting, and retries should also be factored in. The solution laid out here serves as a very solid base for implementing this kind of complex integration.
