---
title: "How do I trigger Airflow DAGs programmatically?"
date: "2024-12-23"
id: "how-do-i-trigger-airflow-dags-programmatically"
---

Okay, let's get into programmatic triggering of Airflow DAGs. It’s something I've dealt with quite a bit over the years, especially when moving away from strictly scheduled workflows to more event-driven pipelines. I remember one particular project involving real-time data ingestion where we had to migrate from a fixed cron schedule to a system that dynamically launched DAGs based on upstream data availability. That experience really hammered home the importance of understanding the different mechanisms for programmatic triggers.

At the core, you’re looking to initiate a dag run outside of Airflow's built-in scheduler. This is usually accomplished through one of Airflow’s apis or interfaces. I've consistently found the rest api to be the most flexible and widely applicable. It allows for integration with diverse systems and platforms using standard http requests. There are other options, like directly interacting with airflow’s python apis, but this tends to introduce dependencies which make it harder to decouple processes.

The key point here is that initiating a DAG programmatically involves making an api call specifying the dag id you wish to trigger, and, optionally, some configuration parameters. It’s quite straightforward when you break it down, but there are some best practices that will save you headaches down the road. For instance, authentication and authorization must be handled carefully to ensure no unauthorized access to your airflow instance, especially if you’re operating in a production setting.

Let's delve into the core mechanics of using the rest api for this purpose. Airflow provides several api endpoints, but we're interested in the ones that handle dag runs. Specifically, the endpoint we'll target for triggering a new dag run is typically `/api/v1/dags/{dag_id}/dagRuns`. This is a `post` request that accepts a json payload containing parameters relevant to this dag run.

Now, before I jump into the code snippets, a few notes on airflow’s configuration are pertinent. To use the api, you'll typically need to enable it and configure authentication methods. This usually involves setting parameters in your `airflow.cfg` file, and also likely creating a user with the appropriate permissions. The recommended way to do this in modern airflow is using roles and permissions rather than modifying the `airflow.cfg` file directly, and should be done using the cli or airflow ui. The official documentation covers this extensively so I'll avoid replicating that information here, however you can find details in the "apache airflow documentation: security" section.

Okay, let's look at the code examples:

**Snippet 1: Python using `requests` library**

```python
import requests
import json

def trigger_dag(dag_id, conf=None, airflow_url="http://localhost:8080", auth=("admin", "admin")):
    """Triggers an Airflow DAG using the REST API."""
    url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
    headers = {"Content-Type": "application/json"}
    payload = {}
    if conf:
        payload["conf"] = conf
    
    try:
        response = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload))
        response.raise_for_status() # Raises an exception for bad status codes
        print(f"Dag {dag_id} triggered successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error triggering dag {dag_id}: {e}")

if __name__ == "__main__":
    my_dag_id = "example_python_operator"  # replace with your dag id
    my_conf = {"my_param": "some_value"}
    trigger_dag(my_dag_id, conf=my_conf)
    trigger_dag(my_dag_id) # Trigger without config params
```

This first snippet shows a basic python function using the `requests` library that most developers will be very comfortable with. It encapsulates the http request that sends the trigger message to the airflow api. It also shows handling of the most common exceptions and it prints some diagnostic outputs. Note the optional `conf` parameter; this allows you to pass extra configurations to your dag. The authentication here uses basic auth with username and password, which are, for the sake of brevity, hardcoded (which should *not* be done in a production system, but it's perfectly fine for testing purposes). In production, you'd usually use a more robust mechanism like an api token or external secret management. A good resource for secure api access is "Oauth 2 in action" by Justin Richer and Antonio Sanso.

**Snippet 2: Using `curl` from the command line**

```bash
DAG_ID="example_bash_operator"
AIRFLOW_URL="http://localhost:8080"
AUTH="admin:admin"
CONFIG='{"my_config": "test_value"}'


curl -X POST \
     -H "Content-Type: application/json" \
     -u "$AUTH" \
     -d "{\"conf\": $CONFIG}" \
     "$AIRFLOW_URL/api/v1/dags/$DAG_ID/dagRuns"
```

This second snippet demonstrates how to achieve the same result with a `curl` command. It's highly beneficial for quick debugging or integration into shell scripts. This approach avoids python dependency which can be useful, especially when you’re working with heterogeneous environments. The curl command showcases how to pass configurations using json as well. It should also be mentioned that some terminals may require special quoting to handle json properly within shell strings. Notice that the username and password are also hardcoded here, just like in snippet one, and require updating in a production environment.

**Snippet 3: Example with authentication token in python**

```python
import requests
import json

def trigger_dag_token(dag_id, conf=None, airflow_url="http://localhost:8080", auth_token="your_auth_token"):
    """Triggers an Airflow DAG using the REST API with an auth token."""
    url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {auth_token}"}
    payload = {}
    if conf:
        payload["conf"] = conf
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise exception for bad status codes
        print(f"Dag {dag_id} triggered successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error triggering dag {dag_id}: {e}")

if __name__ == "__main__":
    my_dag_id = "example_python_operator" # Replace with your dag id
    my_conf = {"my_param": "some_value"}
    my_token = "your_auth_token_here"  # Replace with your token
    trigger_dag_token(my_dag_id, conf=my_conf, auth_token = my_token)
    trigger_dag_token(my_dag_id, auth_token = my_token) # Trigger without config params
```

This third example introduces token-based authentication. This approach is significantly more secure than basic authentication, especially when dealing with sensitive credentials. It's best to use a security token issued by airflow. The bearer token is passed in the 'authorization' header. Token authentication usually requires you to generate this token from within the airflow environment, using the cli or ui, and you should store them in a secure vault. The "Practical Cryptography" by Niels Ferguson and Bruce Schneier can help clarify security considerations if you wish to deepen your understanding of the security aspect of token management.

In summary, programmatically triggering airflow dags using the api is a powerful capability for creating event-driven pipelines, and it's relatively simple to achieve, with these snippets acting as a foundational block. However, remember that security is paramount; proper authentication and authorization are vital for any production system. Additionally, make sure to thoroughly test and monitor the triggered dag runs to ensure they behave as expected, especially when introducing new configurations or parameters. As always, consult the official apache airflow documentation to confirm that you are using the latest security practices and functionality.
