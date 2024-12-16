---
title: "Why can't Portainer pull ECR images or share credentials?"
date: "2024-12-16"
id: "why-cant-portainer-pull-ecr-images-or-share-credentials"
---

Alright, let’s tackle this. It's a recurring issue, and I’ve certainly spent my fair share of late nights debugging similar setups. From what I remember, during a project a couple of years back, we were migrating a bunch of microservices to EKS, leveraging Portainer for container management. It was all going smoothly until we hit the wall with Portainer not playing nice with ECR. The symptoms were exactly what you're describing: Portainer could access the Docker Hub just fine but kept throwing errors when trying to pull images from our private Elastic Container Registry (ECR), or it simply refused to save any authentication data. It’s not uncommon, and the root causes are varied, but let's break down the common culprits and how we can solve them.

Firstly, the core issue usually revolves around authentication. ECR is, by design, far more restrictive than Docker Hub. Portainer, by itself, doesn’t natively handle the complex authentication methods that ECR relies upon. Unlike public registries, ECR requires an aws-cli-generated authorization token that's time-limited and unique to your AWS account and region. This token is crucial for any application or service needing to interact with your ECR repository. When Portainer is configured via the UI, it expects static credentials – username and password pairs – which is why it works with Docker Hub, but fails when presented with the dynamic authentication required for ECR.

The first step in understanding this lies in examining the Portainer architecture. Portainer uses a relatively simple authentication mechanism for its image registry connections. It attempts to store these credentials securely within its database. When trying to access an ECR repository, Portainer doesn’t inherently know to use the aws-cli’s output of a temporary token; it looks for a fixed username and password. This is why simply providing your AWS access key id and secret access key won’t work. Those keys are not the same as the authorization tokens that ECR demands.

Now, let’s get into solutions. There are several ways to enable Portainer to work with ECR, and the method you choose depends heavily on your infrastructure and security policies. I’ll outline three approaches, each with a code snippet to illustrate.

**Approach 1: Using `aws ecr get-login-password` and Static Credential Storage**

This is a less secure method, suitable for testing or development environments, but it's a good starting point for understanding the underlying issue. Essentially, you’re manually retrieving a temporary password using the `aws ecr get-login-password` command, and then attempting to use those generated credentials directly within the Portainer registry setup.

```bash
# First, get the login password:
aws ecr get-login-password --region your-aws-region

# Then, obtain the ECR endpoint (it's part of your registry URI):
aws ecr describe-repositories --region your-aws-region

# Once you have both these, configure the registry in Portainer with the following:
# Username: AWS
# Password: The password outputted from the above command
# Registry URL: The registry URI from the output of the describe-repositories command
```

*Caveat:* Note that the password you get from `aws ecr get-login-password` is valid for a limited period (typically 12 hours), so you would need to generate it again after it expires. Therefore, while this approach demonstrates the token-based auth, it is not production-ready and requires a scheduled task to refresh these credentials. This method essentially forces Portainer to bypass its limitations on ECR’s credential management, as you are using a short-lived token as a static password.

**Approach 2: Using an IAM Role for EKS Nodes (or Other EC2 Instances Running Portainer)**

A much more secure method involves setting up an IAM role that allows the EC2 instance running Portainer to automatically obtain the required ECR permissions. This removes the need for manually-generated temporary tokens. This approach assumes that Portainer is running on an EC2 instance or a Kubernetes node where instance profiles with attached IAM roles are supported.

First, create an IAM role with the following policy attached:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        }
    ]
}
```

Then, attach this role to your EC2 instance or worker node. After that, your Portainer container should be able to pull ECR images without additional configuration or password prompts (assuming it’s running on the same machine or cluster with the proper credentials). Portainer leverages the AWS SDK for Python which should automatically look for the configured IAM role and its associated credentials.

To verify that everything is working correctly, you might need to restart Portainer. However, no additional configuration is needed within Portainer itself, as it will automatically inherit the permissions granted by the instance profile.

*Caveat:* This is a secure way, however, it requires that your Portainer container is running within an AWS environment with an attached role and the proper permissions. This limits this solution to deployment on AWS based resources, not for local setups or deployments outside of AWS.

**Approach 3: Using an AWS Lambda Function to Generate Tokens and a Script or API to Update Portainer**

This is the most sophisticated method and would work across environments where you can call API endpoints, and it's closest to what we ended up implementing. You can create an AWS Lambda function to generate temporary ECR auth tokens and have a scheduled script or API that updates Portainer's registry settings using Portainer’s API.

First, the lambda function:

```python
import boto3
import json

def lambda_handler(event, context):
    ecr = boto3.client('ecr')
    token = ecr.get_authorization_token()['authorizationData'][0]['authorizationToken']
    decoded_token = base64.b64decode(token).decode('utf-8')
    username, password = decoded_token.split(":")
    
    return {
      'username': username,
      'password': password
    }
```

This lambda function uses the boto3 library to retrieve the ECR authorization token, decode it, and output the username ("AWS") and the generated password.

You then need to write a script that calls the lambda function and updates Portainer through its API. Here is an outline:

```python
import requests
import json
import boto3
import os
import base64


def update_portainer_registry(portainer_endpoint, registry_id, username, password, portainer_api_key):

  headers = {
        'X-API-Key': portainer_api_key,
        'Content-Type': 'application/json'
    }

  payload = {
        "username": username,
        "password": password
    }


  url = f'{portainer_endpoint}/api/registries/{registry_id}'

  response = requests.put(url, headers=headers, json=payload, verify=False)
  response.raise_for_status() # Raise an exception for non-200 responses
  print(f"Successfully updated Portainer registry {registry_id}")


def get_ecr_credentials(lambda_function_name, region_name="your-aws-region"):
  client = boto3.client("lambda", region_name=region_name)
  response = client.invoke(
    FunctionName=lambda_function_name,
    Payload='{}' # This is an empty payload
  )
  payload = json.loads(response['Payload'].read().decode())
  return payload['username'], payload['password']


def main():
  portainer_endpoint = os.getenv('PORTAINER_ENDPOINT') # The FQDN/IP and Port number for the Portainer instance
  portainer_api_key = os.getenv('PORTAINER_API_KEY') # API key generated by portainer.
  lambda_function_name = os.getenv("LAMBDA_FUNCTION_NAME") # The lambda function name
  registry_id = os.getenv("REGISTRY_ID") # the ID of the registry in Portainer. 

  username, password = get_ecr_credentials(lambda_function_name)
  update_portainer_registry(portainer_endpoint, registry_id, username, password, portainer_api_key)


if __name__ == '__main__':
  main()
```

This script fetches the temporary ECR credentials from the lambda function and uses Portainer’s API endpoint to update the registry configurations using a provided registry ID.

*Caveat:* While more complex to setup, this approach provides a robust solution where the Portainer endpoint can be updated programmatically. This is important if you are looking to integrate ECR into a broader CI/CD pipeline.

For further reading on AWS IAM, I'd suggest looking at the "AWS Identity and Access Management (IAM)" documentation directly from AWS. This contains everything you need on roles, policies, and best practices around securing your infrastructure. For understanding how AWS Lambda functions work I recommend the AWS documentation pages, specifically the 'Lambda Developer Guide' which is the best place to get started, specifically how to work with boto3. For general container registry and authentication, "Docker in Practice" by Ian Miell and Aidan Hobson Sayers is a worthwhile read, specifically the chapters on registry management and access control.

In summary, Portainer’s inability to directly pull ECR images stems from its reliance on static authentication models, while ECR requires dynamic authorization tokens. We can address this by either manually providing a temporary token (for simple tests) or, more practically, by leveraging IAM roles or a lambda-based token generation method. The right approach depends entirely on the security requirements and sophistication of the environment. Each method described has its pros and cons, but I hope this breakdown gives a clear and actionable path forward.
