---
title: "How can AWS Lambda disable Airflow DAGs?"
date: "2024-12-23"
id: "how-can-aws-lambda-disable-airflow-dags"
---

Alright, let's talk about this. I remember back when I was architecting a pretty sprawling data pipeline a few years ago, we ran into this very challenge – needing to programmatically manage airflow dags from within an aws lambda function. It's a situation that, while not immediately obvious, crops up more frequently than you might think once you start intertwining serverless and orchestration.

The core issue here is that aws lambda is, by its nature, an event-driven compute service, while airflow is a centralized workflow orchestration platform. They don't directly speak the same language, meaning lambda can't directly interact with airflow’s internal api to disable dags. We need to bridge that communication gap, and that’s where the magic happens.

Essentially, you need to trigger an airflow api call from your lambda function. Airflow provides a REST api for interacting with its components, and the most robust way to achieve this involves utilizing airflow’s built-in api functionality. However, there are some crucial considerations to make before diving into code: authentication, authorization, and ensuring secure communication.

One approach, which is the one I've found to be most reliable, involves leveraging an airflow api endpoint that allows you to toggle a dag’s activation status. Airflow, generally, is configured with a user system to control access. Your lambda function will need to authenticate against this api. The preferred methods would involve creating a dedicated service account in airflow and using its credentials within your lambda environment (ideally securely via environment variables and parameter store).

Here's the general pattern: your lambda function receives some trigger (say, an event from dynamodb or s3), then using the service account credentials, it constructs a request to the airflow api, specifying the dag id you intend to disable, and submits it. Let me show you a simple example using python’s `requests` library.

**Example 1: Basic Dag Disabling**

```python
import requests
import os
import json

def disable_airflow_dag(dag_id, airflow_endpoint, airflow_user, airflow_password):
    """
    Disables an airflow dag using its rest api.
    """
    url = f"{airflow_endpoint}/api/v1/dags/{dag_id}/paused"
    auth = (airflow_user, airflow_password)
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"is_paused": True})  # set to 'false' to enable.


    try:
        response = requests.patch(url, auth=auth, headers=headers, data=data)
        response.raise_for_status()  # raises exception on bad status
        print(f"successfully disabled dag: {dag_id}")
    except requests.exceptions.RequestException as e:
        print(f"Error disabling dag {dag_id}: {e}")
        raise  # Propagate the exception for caller to handle

def lambda_handler(event, context):
    dag_id_to_disable = os.environ.get("DAG_ID") # Read dag id from env vars
    airflow_api_endpoint = os.environ.get("AIRFLOW_ENDPOINT")
    airflow_user = os.environ.get("AIRFLOW_USER")
    airflow_password = os.environ.get("AIRFLOW_PASSWORD")

    disable_airflow_dag(dag_id_to_disable, airflow_api_endpoint, airflow_user, airflow_password)

    return {
            'statusCode': 200,
            'body': json.dumps('Lambda execution complete.')
        }
```

In this snippet, we're assuming that the airflow api endpoint, user, password, and the target `dag_id` are configured in the lambda's environment variables. It uses the `requests` library to construct a `patch` request to the airflow api's `/dags/{dag_id}/paused` endpoint, setting `is_paused` to true, to disable the dag. Setting this to `false` would enable it. We implement error handling in the try catch block, this is important because network issues can always occur.

Now, while this gets the job done, it’s not necessarily the most elegant or scalable approach. Managing plain-text credentials in environment variables isn’t ideal. I’ve found it much better to use aws secrets manager to store these credentials. Let’s refactor the previous example to use secrets manager.

**Example 2: Using AWS Secrets Manager for Authentication**

```python
import requests
import os
import json
import boto3
from botocore.exceptions import ClientError

def get_secrets(secret_name):
    """
    Retrieves secrets from aws secrets manager.
    """
    region_name = "your-aws-region" # replace with your aws region
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
            return json.loads(secret)  # Assuming secret is in json format

        return None  # Handle binary secrets later
    except ClientError as e:
        print(f"error retrieving secret: {e}")
        raise

def disable_airflow_dag(dag_id, airflow_endpoint, airflow_secrets):
        """
        Disables an airflow dag using its rest api and credentials from aws secrets manager.
        """

        auth = (airflow_secrets['username'], airflow_secrets['password'])
        url = f"{airflow_endpoint}/api/v1/dags/{dag_id}/paused"
        headers = {"Content-Type": "application/json"}
        data = json.dumps({"is_paused": True})

        try:
            response = requests.patch(url, auth=auth, headers=headers, data=data)
            response.raise_for_status() # Raises an exception on a bad status code
            print(f"successfully disabled dag: {dag_id}")
        except requests.exceptions.RequestException as e:
            print(f"error disabling dag {dag_id}: {e}")
            raise # Propagate for handler

def lambda_handler(event, context):
    dag_id_to_disable = os.environ.get("DAG_ID")
    airflow_api_endpoint = os.environ.get("AIRFLOW_ENDPOINT")
    secret_name = os.environ.get("AIRFLOW_SECRETS_NAME")

    airflow_secrets = get_secrets(secret_name)
    if airflow_secrets:
        disable_airflow_dag(dag_id_to_disable, airflow_api_endpoint, airflow_secrets)
    else:
        print("Failed to retrieve airflow secrets from secrets manager.")

    return {
            'statusCode': 200,
            'body': json.dumps('Lambda execution complete.')
        }
```

Here, we've added a `get_secrets` function, utilizing the `boto3` client to interact with aws secrets manager, retrieving our user credentials and decoding the json secret. This makes credential management much more secure and maintainable.

Now you might be thinking, how do we handle more complex scenarios, like disabling multiple dags or filtering based on other criteria? You’d typically need to make use of airflow’s `list_dags` endpoint and then iterate through the results and disable them as required, based on some logic.

**Example 3: Disabling Multiple DAGs Based on Filters**

```python
import requests
import os
import json
import boto3
from botocore.exceptions import ClientError

def get_secrets(secret_name):
  # same as example 2. removed for brevity
  pass

def list_airflow_dags(airflow_endpoint, airflow_secrets, filter_criteria):
    """
    Lists airflow dags based on filter criteria and returns a list of dag ids.
    """

    auth = (airflow_secrets['username'], airflow_secrets['password'])
    url = f"{airflow_endpoint}/api/v1/dags"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, auth=auth, headers=headers)
        response.raise_for_status()
        dags = response.json()["dags"]
        filtered_dags = [dag['dag_id'] for dag in dags if filter_criteria(dag)] #apply our filter
        return filtered_dags
    except requests.exceptions.RequestException as e:
        print(f"error getting list of dags: {e}")
        raise

def disable_airflow_dag(dag_id, airflow_endpoint, airflow_secrets):
        # same implementation as Example 2. Removed for brevity.
        pass

def lambda_handler(event, context):
    airflow_api_endpoint = os.environ.get("AIRFLOW_ENDPOINT")
    secret_name = os.environ.get("AIRFLOW_SECRETS_NAME")
    dag_filter = os.environ.get("DAG_FILTER_CRITERIA") # filter criteria from an environment variable

    airflow_secrets = get_secrets(secret_name)
    if airflow_secrets:
       def filter_function(dag):
          if "contains" in dag_filter:
             return dag_filter.split("contains:")[1] in dag['dag_id'] # example criteria 'contains:my-prefix'
          return True  # default to return all.

       dag_ids = list_airflow_dags(airflow_api_endpoint, airflow_secrets, filter_function)
       for dag_id in dag_ids:
            disable_airflow_dag(dag_id, airflow_api_endpoint, airflow_secrets)
    else:
        print("Failed to retrieve airflow secrets from secrets manager.")

    return {
            'statusCode': 200,
            'body': json.dumps('Lambda execution complete.')
        }
```

In this example, we've added a `list_airflow_dags` function that interacts with the `/dags` endpoint to retrieve all dags. We then use a filter function to select only the dags we wish to disable, and loop through them individually to disable them.

I would highly recommend reading the official apache airflow documentation regarding the rest api endpoints, this will provide all the details for specific versions of airflow you may be using. "airflow documentation" in your preferred search engine should suffice. For a more robust understanding of aws services, especially the lambda and secrets manager, the "aws documentation" is of course, a must. These are great resources for gaining in-depth knowledge in this space.

This should give you a solid foundation for understanding how you can control airflow dags from within lambda. Remember to always prioritize security when dealing with api credentials and network communication, and test your implementations thoroughly. Good luck out there!
