---
title: "How can I Trigger an airflow DAG using a post request with parameters?"
date: "2024-12-23"
id: "how-can-i-trigger-an-airflow-dag-using-a-post-request-with-parameters"
---

Alright, let's tackle triggering airflow dags via post requests with parameters; it's a fairly common requirement, and I've certainly encountered it multiple times over the years. I recall a particularly tricky scenario during a project migrating data pipelines from a legacy system. We needed the ability to trigger specific data processing flows dynamically, based on events originating from various external applications. The standard Airflow web UI simply wouldn't cut it for that use case.

The core mechanism involves using Airflow's REST API, which thankfully is quite capable. The key lies in understanding how to structure your POST request, specifically the json payload, to effectively communicate with the Airflow scheduler. You’re essentially crafting a request to instruct Airflow to kick off a new dag run, passing along specific configurations as needed.

The fundamental idea revolves around the `dag_runs` endpoint and using POST method. You provide the dag id you wish to trigger and, crucially, any configurations you need as json. Let’s break down what this involves in practice with some illustrative code examples.

**Code Snippet 1: Simple Trigger without Parameters (Python)**

This first example showcases the basic process of triggering a dag without any dynamic parameters. This would apply if, for instance, you have a dag that always operates on the same data source, or if all the necessary configurations are pre-defined in the DAG itself.

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080"  # Replace with your Airflow webserver URL
dag_id = "my_simple_dag" # Replace with your DAG ID
auth = ("airflow", "airflow") # Replace with your airflow webserver auth or remove this line if no auth required

url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"

headers = {
    "Content-Type": "application/json"
}

payload = {
    "conf": {}, #Empty dictionary since no parameters are needed
    "run_id": None, #optional - leave as None if you want airflow to generate one
}

try:
    response = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload))
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    data = response.json()
    print(f"DAG Run triggered successfully! Dag Run ID: {data['dag_run_id']}")
except requests.exceptions.RequestException as e:
    print(f"Error during request: {e}")
    if hasattr(response, 'text'):
        print(f"Error response: {response.text}")
```
In this example, we use the `requests` library to send a POST request to the specified airflow api endpoint. The `conf` field within the payload, initially an empty dictionary here, is where you’d pass any additional parameters. Note that we also handle potential request exceptions to make it more robust. The optional `run_id` field allows you to specify a custom ID for the dag run if needed. If left as `None`, Airflow generates one.

**Code Snippet 2: Trigger with Parameters (Python)**

Now, let's get to the more useful scenario - triggering a dag with parameters. This is crucial for dynamic behaviour and customization of DAG runs. This is similar to the project I mentioned earlier, where parameters controlled which part of a complex pipeline would run.

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080"  # Replace with your Airflow webserver URL
dag_id = "my_parameterized_dag" # Replace with your DAG ID
auth = ("airflow", "airflow") # Replace with your airflow webserver auth or remove this line if no auth required


url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"

headers = {
    "Content-Type": "application/json"
}


params = {
    "start_date": "2024-01-01",
    "data_source": "s3://my-bucket/input_data/",
    "data_output": "s3://my-bucket/output_data/",
     "priority": "high"
}


payload = {
    "conf": params,
    "run_id": "custom_run_id_123" # A custom run id for demonstrative purposes
}
try:
    response = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload))
    response.raise_for_status()
    data = response.json()
    print(f"DAG Run triggered successfully with parameters! Dag Run ID: {data['dag_run_id']}")

except requests.exceptions.RequestException as e:
    print(f"Error during request: {e}")
    if hasattr(response, 'text'):
        print(f"Error response: {response.text}")
```
Here, the primary change is within the `payload`. We now have a `params` dictionary which holds key-value pairs representing the desired configuration for our DAG run. These parameters are passed into the DAG using the `conf` argument. In your actual dag definition, you can then access these values through `dag_run.conf`. The custom `run_id` field shows how you would set a specific run id, instead of letting airflow autogenerate one.

**Code Snippet 3: Practical example and validation (Python)**

This final example is a bit more elaborate and includes some validations before actually triggering the DAG. Here we check to see if the DAG exists first.

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080" # Replace with your Airflow webserver URL
dag_id = "my_advanced_dag"  # Replace with your DAG ID
auth = ("airflow", "airflow") # Replace with your airflow webserver auth or remove this line if no auth required


headers = {
    "Content-Type": "application/json"
}

params = {
    "config_location": "/path/to/config.json",
    "process_id": 12345,
    "debug_mode": True
}

payload = {
    "conf": params,
    "run_id": None
}

try:
    # Check if DAG exists
    check_url = f"{airflow_url}/api/v1/dags/{dag_id}"
    check_response = requests.get(check_url, headers=headers, auth=auth)
    check_response.raise_for_status()

    url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
    response = requests.post(url, headers=headers, auth=auth, data=json.dumps(payload))
    response.raise_for_status()
    data = response.json()
    print(f"Advanced DAG Run triggered successfully! Dag Run ID: {data['dag_run_id']}")


except requests.exceptions.RequestException as e:
    print(f"Error during request: {e}")
    if hasattr(response, 'text'):
      print(f"Error response: {response.text}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

In this snippet, before triggering the dag we make an additional request to `/api/v1/dags/{dag_id}` using the `get` method. This allows us to perform pre-checks, such as ensuring that the target dag exists within Airflow. If the dag is missing, then the request will fail and the code will not proceed. This is a simple but practical validation step which is very useful to have in production systems.

**Important Considerations**

- **Authentication:** Airflow security is critical, so be sure to configure proper authentication in your environment. The examples use a basic username/password auth, but you’d likely want to use api keys, oauth2, or more robust authentication methods in a production setting. Refer to the airflow documentation for details on how to configure security.
- **Error Handling:** As demonstrated, include error handling to capture connection problems, invalid payloads, or authorization issues. Logging these problems will drastically improve debugging.
- **Parameter Validation:** In your DAG definition, validate parameters in your code before utilizing them. This ensures data integrity and can prevent unexpected issues.
- **Rate Limiting:** In production, consider implementing rate limiting on your application to prevent overloading the Airflow scheduler.
- **JSON Serialization:** Pay attention to the proper serialization of data to json for the API. Especially when working with complex data structures.

**Recommended Resources**

For a deeper understanding of the concepts involved, consider these materials:
* **Apache Airflow Documentation:** This is the primary and most authoritative resource. It is always best to refer to the most up to date information available on the official website.
* **"Programming Apache Airflow" by Bas Harenslak and Julian de Ruiter:** A comprehensive guide covering various aspects of Airflow, including the REST API and its utilization.
* **"Designing Data-Intensive Applications" by Martin Kleppmann:** Although not solely focused on Airflow, this book provides essential knowledge on distributed systems and data pipelines, which is relevant in understanding the context of Airflow.

Implementing these strategies should empower you to flexibly trigger Airflow DAGs via POST requests with custom parameters. Remember to thoroughly test your implementation, and most importantly, to consult the official Airflow documentation for the latest information and best practices. I hope this practical and detailed explanation helps!
