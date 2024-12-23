---
title: "Why can't a Lambda function running in Localstack connect to the Airflow REST API?"
date: "2024-12-23"
id: "why-cant-a-lambda-function-running-in-localstack-connect-to-the-airflow-rest-api"
---

Let's talk about why a Lambda function, humming along in your Localstack environment, might struggle to reach an Airflow REST API. I've bumped into this exact scenario a few times, often late at night, debugging setups that *should* just work, but don’t. It's a classic case of misaligned expectations, often arising from the complexities of networking within a simulated environment.

The core issue revolves around network isolation. Localstack, while impressively simulating aws services, operates within its own contained network space. When you deploy a Lambda function using Localstack’s s3 or cli, it’s executing inside this virtual network, isolated from your host machine and potentially from other containers. This isolation prevents direct, effortless communication between the Lambda function and, let’s say, an Airflow instance running as another container, or even on the host machine outside of docker.

Here's a breakdown of the common culprits, presented from the perspective of someone who's burned their fingers enough times on similar setups:

**1. Incorrect Airflow REST API Endpoint:** This is the most frequent offender. Your Lambda function is configured to target an endpoint that is not actually reachable. This usually stems from the following situations:

    *   **Host Machine Endpoint:** If your Airflow instance is running directly on your host machine (outside docker), the endpoint might be something like `http://localhost:8080/api/v1/dags`. The crucial bit here is `localhost`. From inside a Docker container (where Lambda functions run in Localstack), `localhost` refers to the *container's* localhost, not your host machine's. Therefore, your Lambda function’s attempted connection goes nowhere.
    *   **Incorrect Docker Network Configuration:** If both Airflow and Localstack (and its contained Lambda) are within different Docker containers or networks, simply using `localhost` or even `127.0.0.1` won't work. The container running your Lambda cannot resolve the name or IP of the Airflow container if they are not within the same or a connected Docker network.

**2. Network Routing and DNS Resolution Issues:** Even if you've configured docker-compose to interconnect services via a named network, sometimes the dns resolution within the lambda’s container doesn’t correctly map container names to their ip addresses. This might require explicit mapping of service names to their ips or by leveraging docker's dns configuration, even if you are in the same network. This is especially true if you have complex setups with custom docker networks.

**3. Missing Network Permissions:** Less common, but still plausible, is a lack of network permissions on the Airflow instance or Localstack configuration. Typically, this involves the Airflow webserver being explicitly configured to *only* accept connections from certain IPs (or not listen on the correct port/interface) or misconfigured Localstack network settings, like not exposing the necessary ports correctly. This is a relatively rarer case if your airflow setup is using default configurations.

Let's illustrate with some code examples:

**Example 1: Illustrating the host machine localhost problem:**

Suppose, you have a python Lambda function, `lambda_handler.py`:

```python
import requests

def lambda_handler(event, context):
    try:
        airflow_url = "http://localhost:8080/api/v1/dags" # Problematic URL
        response = requests.get(airflow_url)
        response.raise_for_status()
        return {
            'statusCode': 200,
            'body': response.json()
        }
    except requests.exceptions.RequestException as e:
        return {
            'statusCode': 500,
            'body': f"Error: {e}"
        }
```

And the above is running via lambda deployed using `awslocal lambda deploy` from the localstack cli, while your Airflow webserver is running on your machine directly using `airflow webserver`. The lambda function will be unable to reach the webserver because the `localhost:8080` from the Lambda container, points to itself, not to your host machine.

**Example 2: Demonstrating correct service to service communication in the same docker network:**

Assume both the Lambda function, through Localstack, and Airflow are running in docker containers in the same custom docker network called `my_network`. Assume also that you have configured the docker-compose for both to be connected to `my_network` and the airflow webserver is exposed with the docker-compose service name `airflow`. You could then configure your Lambda function to call the airflow server via:

```python
import requests

def lambda_handler(event, context):
    try:
        airflow_url = "http://airflow:8080/api/v1/dags"  # correct url using service name
        response = requests.get(airflow_url)
        response.raise_for_status()
        return {
            'statusCode': 200,
            'body': response.json()
        }
    except requests.exceptions.RequestException as e:
        return {
            'statusCode': 500,
            'body': f"Error: {e}"
        }
```

This will only succeed if both the lambda container running in localstack and the airflow container are in the same docker network, and airflow service is exposed with the name 'airflow' in that network. Docker's internal DNS system will translate `airflow` to the container's internal IP.

**Example 3: Incorrect resolution using IP instead of service name:**

Building on example 2, let's say that your airflow container’s ip within the `my_network` is `172.18.0.2`. Then, the following code *might* work but it is bad practice:

```python
import requests

def lambda_handler(event, context):
    try:
        airflow_url = "http://172.18.0.2:8080/api/v1/dags"  # hardcoded ip address, bad practice
        response = requests.get(airflow_url)
        response.raise_for_status()
        return {
            'statusCode': 200,
            'body': response.json()
        }
    except requests.exceptions.RequestException as e:
        return {
            'statusCode': 500,
            'body': f"Error: {e}"
        }
```

This will probably work, but it introduces brittleness. IPs can change if containers are restarted or rebuilt, so using service names is much better, because the docker dns will reliably translate the service name to the current ip address.

**Troubleshooting Steps:**

1.  **Verify Network Configuration:** Carefully inspect your `docker-compose.yml` (or similar) to ensure both your Localstack containers (including the Lambda execution environment) and Airflow are on the same docker network and can resolve each other.
2.  **Test Connectivity:** Within the Localstack container that hosts your Lambda function (using `docker exec -it <localstack_container_id> bash`), you can use `curl` or `wget` to attempt to reach the Airflow API URL. This will pinpoint whether the core issue lies in the network layer or within the code.
3.  **Use Service Names:** As much as possible, rely on docker service names for your communication endpoints rather than hardcoded IP addresses.
4.  **Examine logs:** Both Localstack and Airflow will have logs. Careful analysis of these logs can pinpoint whether the issue is in the lambda code, network configuration or airflow configuration.

**Recommended Reading:**

*   **Docker Documentation:** The official Docker documentation is indispensable for understanding network isolation and container communication. Particularly, focus on sections detailing user defined networks, bridge networks, and dns resolution within Docker networks.
*   **Localstack Documentation:** Pay special attention to networking details, especially when deploying services using localstack's cli or s3.
*   **'Programming Kubernetes' by Michael Hausenblas and Stefan Schimanski:** Although focused on Kubernetes, this book provides a deep and beneficial overview of container networking concepts applicable in docker as well, which are the foundation for localstack.
*   **'Docker in Action' by Jeff Nickoloff:** This will give you a solid, very readable foundation in all things docker, which is critical to understanding what is going on under the hood with localstack.

In closing, connecting a Lambda function within Localstack to an Airflow REST API often requires diligent attention to the nuances of network isolation and service discovery in containerized environments. By methodically checking the points discussed and leveraging your understanding of networking, you’ll be able to set up a reliable solution that allows your local Lambda functions to interact with the simulated services within the Localstack environment. It's a problem-solving exercise in microcosm of larger, similar issues one faces in cloud environments, which makes solving this, a worthwhile learning endeavor.
