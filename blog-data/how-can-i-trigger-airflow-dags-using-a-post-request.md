---
title: "How can I trigger Airflow DAGs using a POST request?"
date: "2024-12-23"
id: "how-can-i-trigger-airflow-dags-using-a-post-request"
---

Okay, let's talk about triggering Airflow DAGs with POST requests. It’s a topic I've navigated quite a few times over the years, particularly during my stint building a high-throughput data ingestion platform for a large e-commerce operation. We needed to orchestrate a series of complex data transformations based on upstream system events, and polling simply wasn’t cutting it. So, moving to a push model using the Airflow API became essential.

The short answer: Airflow's REST API allows you to trigger DAG runs via http POST requests. However, it isn’t quite as straightforward as sending a simple request to a url. You'll need to authenticate and construct the request body properly. The key aspect here is interacting correctly with the Airflow API endpoint and understanding how to pass configuration parameters.

Before we delve into the code, let me provide some context. Airflow’s security model mandates authentication for API interactions. This means you can't just toss a POST request at an endpoint and expect it to work. You'll typically need either API tokens or some other form of authentication setup, such as basic auth. This is typically configured within your airflow installation’s configuration. The process for obtaining these credentials varies based on your airflow setup; if you’re using the standalone install, generating API tokens is managed through the UI. If it's a more complex setup, such as deploying on Kubernetes, token configuration would involve your infrastructure's secret management. I'm going to assume you have that covered and can generate an access token for this. Let's focus on the core API interaction.

The primary endpoint for triggering DAG runs is:

`/api/v1/dags/{dag_id}/dagRuns`

You'll be sending a `POST` request to this endpoint with a json body containing the configuration parameters for your DAG run. These parameters are crucial; they allow you to customize each run and pass context-specific data to your tasks. Now, let's get into the examples.

**Example 1: Triggering a DAG with no configuration**

This is the simplest scenario: You just need to kick off a DAG with its default configurations. Here is how it is done with python using the requests library.

```python
import requests
import json

def trigger_dag(dag_id, airflow_url, access_token):
  """
  Triggers an airflow dag with a POST request.
  """
  url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
  headers = {
      "Authorization": f"Bearer {access_token}",
      "Content-Type": "application/json",
  }
  payload = {}

  try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    print(f"DAG {dag_id} triggered successfully. Response: {response.json()}")
  except requests.exceptions.RequestException as e:
      print(f"Error triggering DAG {dag_id}: {e}")

if __name__ == '__main__':
    dag_id_to_trigger = "my_example_dag"  # Replace with your DAG id
    airflow_url_base = "http://your-airflow-host:8080" # Replace with your airflow url
    api_token = "your_api_token"  # Replace with your access token

    trigger_dag(dag_id_to_trigger, airflow_url_base, api_token)
```

In this first example, we construct a basic POST request to the appropriate endpoint with the necessary authorization header. The payload is an empty dictionary because we aren't passing any specific configurations. Crucially, `response.raise_for_status()` ensures we catch any error codes returned by the API, making our solution more robust. Remember to replace the placeholder values with your actual DAG ID, Airflow URL, and your API token.

**Example 2: Triggering a DAG with configuration parameters**

Now, let’s look at a common scenario where you need to pass configuration parameters to your DAG run. These parameters can drive conditional logic or provide input data to your tasks. Here's how you'd send a POST request to accomplish that:

```python
import requests
import json

def trigger_dag_with_config(dag_id, airflow_url, access_token, config):
  """
  Triggers an airflow dag with a POST request and configuration.
  """
  url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
  headers = {
      "Authorization": f"Bearer {access_token}",
      "Content-Type": "application/json",
  }
  payload = {"conf": config}
  try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    print(f"DAG {dag_id} triggered with config successfully. Response: {response.json()}")
  except requests.exceptions.RequestException as e:
     print(f"Error triggering DAG {dag_id} with config: {e}")

if __name__ == '__main__':
    dag_id_to_trigger = "my_parameterized_dag" # Replace with your DAG id
    airflow_url_base = "http://your-airflow-host:8080" # Replace with your airflow url
    api_token = "your_api_token" # Replace with your access token
    config_parameters = {
        "start_date": "2024-02-29",
        "dataset_name": "monthly_sales",
        "file_type": "csv"
    }
    trigger_dag_with_config(dag_id_to_trigger, airflow_url_base, api_token, config_parameters)
```

Here, we include a `"conf"` key in our payload. The value associated with this key is a python dictionary. Airflow will transform this json dictionary into a python dictionary, making the parameters available inside your dag using `dag_run.conf`. I often passed parameters such as input file paths, query strings, and other configurations critical to controlling processing logic through this approach. The specific details depend on what you need within the tasks that comprise the DAG.

**Example 3: Triggering a DAG with a specific run id**

Sometimes you may want to specify a custom run id for the DAG, perhaps for better tracking or unique naming in your applications. Here's how you would handle that:

```python
import requests
import json
import uuid

def trigger_dag_with_run_id(dag_id, airflow_url, access_token, run_id):
  """
  Triggers an airflow dag with a POST request, custom run id and configuration.
  """
  url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
  headers = {
      "Authorization": f"Bearer {access_token}",
      "Content-Type": "application/json",
  }
  payload = {"run_id": run_id}
  try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    print(f"DAG {dag_id} triggered with run id {run_id} successfully. Response: {response.json()}")
  except requests.exceptions.RequestException as e:
        print(f"Error triggering DAG {dag_id} with run id {run_id}: {e}")


if __name__ == '__main__':
    dag_id_to_trigger = "my_custom_run_id_dag" # Replace with your DAG id
    airflow_url_base = "http://your-airflow-host:8080" # Replace with your airflow url
    api_token = "your_api_token"  # Replace with your access token

    custom_run_id = f"my_run_{uuid.uuid4()}" #Generating a random run id

    trigger_dag_with_run_id(dag_id_to_trigger, airflow_url_base, api_token, custom_run_id)
```

The example introduces the `run_id` field in the request body, which specifies a custom run identifier. It is important to make sure that the run_id is unique since airflow rejects a run request with a run_id that exists. Generating a UUID is one way to make sure the run_id is unique. I’ve found this incredibly useful for tracing individual runs when dealing with many asynchronous events.

**Key Considerations and Best Practices**

*   **Authentication:** Always secure your Airflow API by using strong authentication mechanisms (API tokens, basic auth with TLS, etc.) and avoid storing credentials directly in your code.
*   **Error Handling:** As the above code examples show, use `response.raise_for_status()` to handle failed API requests gracefully. This allows for easier troubleshooting.
*   **Rate Limiting:** Be mindful of potential rate limits on your Airflow API. Avoid excessive polling or sending large volumes of trigger requests. The specific limits depends on your infrastructure configuration
*   **Idempotency:** Design your DAGs to be idempotent whenever possible, particularly if using custom run ids, so that triggering the same DAG with identical configuration multiple times won't result in unintended effects. This may involve extra logic, but it’s worth it to ensure data consistency.
*   **Monitoring:** Implement monitoring of your DAG trigger requests. Monitor for error cases to ensure data pipelines are not failing silently.
*   **Alternative triggers:** While POST requests offer great flexibility, also consider alternatives like Airflow sensors for polling, which may be more appropriate in some contexts.

**Further Reading**

To gain a deeper understanding of these topics, I recommend the following resources:

*   **The official Apache Airflow Documentation:** This is the best starting point. Pay special attention to the REST API reference and the configuration section. This documentation is the gold standard and is routinely updated.
*  **"Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian Rutger de Ruiter:** This book gives an excellent practical overview of airflow, covering topics like DAG design, best practices, and API interactions, among others.
*  **"Designing Data-Intensive Applications" by Martin Kleppmann:** While not directly Airflow-specific, this book provides essential background on distributed systems design, principles that are vital when working with airflow in complex environments.

Triggering DAGs via the Airflow API is a powerful tool. With careful planning and by implementing the points above, you can build robust, event-driven data pipelines. The key is understanding the API, implementing secure authentication, and crafting your request bodies correctly. These steps should serve as a solid base, feel free to delve into the documentation for more complex scenarios that may involve advanced configurations.
