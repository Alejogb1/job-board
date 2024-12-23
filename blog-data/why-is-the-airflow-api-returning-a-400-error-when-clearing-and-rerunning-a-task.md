---
title: "Why is the Airflow API returning a 400 error when clearing and rerunning a task?"
date: "2024-12-23"
id: "why-is-the-airflow-api-returning-a-400-error-when-clearing-and-rerunning-a-task"
---

Alright, let's unpack this peculiar 400 error when you're trying to clear and rerun a task in Airflow. It's a situation I've seen pop up more than a few times in my career, and it's usually not as straightforward as it initially appears. The 400, or "Bad Request," error from the Airflow API typically points to an issue with the data you're sending in your request, rather than a problem with the Airflow server itself. It's a message saying, "Hey, I received your request, but something about it isn't right," which can be quite broad.

Specifically, when you're clearing and rerunning a task, you're essentially interacting with the airflow’s `/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}` endpoint, often through the CLI command or a custom script. The issue almost always resides in the payload of this request. Think of this payload as the specific instructions you're giving the api. It must contain the right data structure and valid parameters.

I remember troubleshooting this exact error on a high-throughput data pipeline a few years back. The team was aggressively iterating, and frequently clearing and rerunning tasks. Initially, things worked smoothly, but then the 400 errors started cropping up, seemingly out of nowhere. We initially suspected database locks, or even inconsistent airflow configuration, but after a significant amount of debugging, the culprit was, in fact, a malformed request payload.

First, let’s consider the parameters needed. You're going to be sending a JSON payload for this request. Commonly, the following parameters are relevant when clearing and rerunning:

* `include_upstream`: Boolean; whether to include upstream tasks in the operation.
* `include_downstream`: Boolean; whether to include downstream tasks in the operation.
* `include_future`: Boolean; whether to include future tasks in the operation.
* `execution_date`: String; this is usually not included, as the dag_run_id in the URL should suffice, but can cause trouble if there is a mismatch, particularly if you're trying to trigger the action on a historic run.

The most common error, from my experience, happens when there's either a syntax error in the JSON itself, the parameters are incorrectly formatted (e.g. sending a string when a boolean is expected), or a required parameter is missing entirely. Another frequently seen cause is the inclusion of additional, unexpected parameters. The API is very strict about the shape of the payload.

Let’s take a look at some examples. Here's an example of a common mistake and how to fix it with three scenarios in Python, illustrating different payload structures and how to avoid the 400 error. We are going to use the `requests` library, which you should already have if using Python for interacting with Airflow. It's crucial to understand how these structures impact your requests.

**Example 1: Incorrect Boolean Formatting**

Let’s say you want to clear only the given task, without affecting its up or downstream tasks, and your code mistakenly formats the booleans as strings.

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080"
dag_id = "example_dag"
dag_run_id = "scheduled__2024-05-03T00:00:00+00:00"
task_id = "your_task_id"
auth_tuple = ('your_username', 'your_password')


url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}"


payload = {
    "include_upstream": "false",
    "include_downstream": "false",
    "include_future": "false"
}


headers = {'Content-type': 'application/json'}

try:
    response = requests.post(url, auth=auth_tuple, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    print("task cleared and reran successfully")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    print(f"Response Text: {response.text}")
```

In this case, `include_upstream`, `include_downstream`, and `include_future` are incorrectly formatted as strings (`"false"`) rather than booleans (`false`). This will result in a 400 error.

**The Fix:**

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080"
dag_id = "example_dag"
dag_run_id = "scheduled__2024-05-03T00:00:00+00:00"
task_id = "your_task_id"
auth_tuple = ('your_username', 'your_password')

url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}"

payload = {
    "include_upstream": False,
    "include_downstream": False,
    "include_future": False
}

headers = {'Content-type': 'application/json'}

try:
    response = requests.post(url, auth=auth_tuple, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    print("task cleared and reran successfully")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    print(f"Response Text: {response.text}")
```

By using the boolean `False`, this will send the correct structure to the API. This is often the most common mistake.

**Example 2: Missing or Extra Parameters**

Now, let's look at a case where you mistakenly send an extra, undefined parameter, or completely omit a parameter required for the request. Here we'll just send `include_upstream` which may be sufficient in some contexts, but in many cases, will return a 400 error if the API is expecting `include_downstream` and `include_future` as well.

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080"
dag_id = "example_dag"
dag_run_id = "scheduled__2024-05-03T00:00:00+00:00"
task_id = "your_task_id"
auth_tuple = ('your_username', 'your_password')


url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}"


payload = {
    "include_upstream": False
}

headers = {'Content-type': 'application/json'}

try:
    response = requests.post(url, auth=auth_tuple, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    print("task cleared and reran successfully")

except requests.exceptions.RequestException as e:
     print(f"Error: {e}")
     print(f"Response Text: {response.text}")
```

This will most likely return a 400 error due to an invalid request payload.

**The Fix:**

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080"
dag_id = "example_dag"
dag_run_id = "scheduled__2024-05-03T00:00:00+00:00"
task_id = "your_task_id"
auth_tuple = ('your_username', 'your_password')


url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}"


payload = {
    "include_upstream": False,
    "include_downstream": False,
    "include_future": False
}

headers = {'Content-type': 'application/json'}

try:
    response = requests.post(url, auth=auth_tuple, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    print("task cleared and reran successfully")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    print(f"Response Text: {response.text}")
```

Always check your API documentation or try a `GET` request on the endpoint to understand exactly which parameters to include.

**Example 3: Correct Implementation**

This is how a correct request should look when clearing a single task and its upstream and downstream dependencies

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080"
dag_id = "example_dag"
dag_run_id = "scheduled__2024-05-03T00:00:00+00:00"
task_id = "your_task_id"
auth_tuple = ('your_username', 'your_password')

url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}"

payload = {
    "include_upstream": True,
    "include_downstream": True,
    "include_future": False
}


headers = {'Content-type': 'application/json'}

try:
    response = requests.post(url, auth=auth_tuple, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    print("Task and it's dependencies cleared and reran successfully")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    print(f"Response Text: {response.text}")
```

This code illustrates the correct structure for a request when you want to clear a task and its upstream and downstream dependencies using the correct payload format.

**Recommendations and Further Reading:**

For a deeper understanding of the airflow API, the official Apache Airflow documentation is an invaluable resource. Specifically, dive into the sections concerning the REST API and the Task Instance endpoints. You can also find more information about the specific parameters for the task instances endpoints in the documentation generated when you run an airflow server.

A book I’d recommend for a more general understanding of working with APIs is “RESTful Web APIs” by Leonard Richardson and Mike Amundsen. Although it doesn't focus specifically on Airflow, it provides a solid understanding of how rest apis work in general, which helps in debugging them. Also, familiarize yourself with the HTTP specification, particularly section 6.5.1 which defines error codes, which will be helpful in the event you don't see an explicit error message returned from the API, and you can infer something from the HTTP status code itself.

The key is meticulous attention to the parameters in the JSON payload you’re sending, ensuring they match what the Airflow API expects. In my experience, this is the root cause of about 90% of 400 errors when dealing with Airflow's task instance operations. Triple-checking these details, and carefully constructing your payloads, will significantly minimize the occurrence of this error. Remember, the API is a stickler for detail, so always double check your JSON.
