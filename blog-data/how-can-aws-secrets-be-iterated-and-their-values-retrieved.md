---
title: "How can AWS secrets be iterated and their values retrieved?"
date: "2024-12-23"
id: "how-can-aws-secrets-be-iterated-and-their-values-retrieved"
---

Alright, let's delve into this. I've seen my share of projects where properly managing secrets in AWS became a bottleneck, especially when dealing with a large number of them. Iterating through those secrets and fetching their values is a pretty common need, and thankfully, aws offers a few good pathways to do it. It isn't always a straightforward process, and it often involves understanding the nuances of the aws sdk and appropriate permissions.

The core challenge here lies in the fact that secrets manager isn't designed for easy, broad sweeps of every secret. It's geared towards retrieving specific secrets you know exist. Consequently, the methods for iterating require a different approach than simply asking for "all the secrets". Primarily, we use a paginated approach, combined with filtering, if needed, to locate our targets.

My first hands-on experience with this was in a previous role at a fintech startup. We were using AWS secrets manager extensively for everything from database credentials to third-party api keys. We started running into a deployment bottleneck where our automated pipelines needed to know the values of a growing number of secrets. Manually pulling secrets became infeasible, and that's where I had to really wrap my head around this problem.

The key is to utilize the `list_secrets` method and then `describe_secret` to gather the metadata, and finally `get_secret_value` to retrieve the actual secrets. Let's break down three example snippets, covering different aspects of this:

**Snippet 1: Basic Iteration with Pagination (Python)**

```python
import boto3

def get_all_secrets():
    secrets_client = boto3.client('secretsmanager')
    all_secrets = {}
    next_token = None

    while True:
        response = secrets_client.list_secrets(MaxResults=100, NextToken=next_token)
        for secret in response['SecretList']:
            secret_name = secret['Name']
            try:
              desc_response = secrets_client.describe_secret(SecretId=secret_name)
              if desc_response.get('RotationEnabled',False) == False:
                secret_value_response = secrets_client.get_secret_value(SecretId=secret_name)
                all_secrets[secret_name] = secret_value_response.get('SecretString') or secret_value_response.get('SecretBinary')
              else:
                print(f"Skipping rotated secret: {secret_name}")
            except Exception as e:
                print(f"Error retrieving secret {secret_name}: {e}")

        next_token = response.get('NextToken')
        if not next_token:
            break
    return all_secrets


if __name__ == '__main__':
    secrets = get_all_secrets()
    for name, value in secrets.items():
        print(f"Secret: {name}: Value: {value[0:20]} ...") #Only print the first 20 characters of the value
```

This first example is the foundational step. It iterates through all your secrets, retrieves the string or binary value (whichever is available), and stores them in a dictionary. Note the use of pagination via `next_token`; this is crucial when you have more than 100 secrets since aws limits the response size. I've also added exception handling within the loop to prevent a single failing secret from breaking the entire operation. The `describe_secret` is added to not retrieve secrets which are set for rotation, as this usually can fail. The output includes the secret name and the first 20 characters of the value, to avoid cluttering the output.

**Snippet 2: Filtering Secrets by Tag (Python)**

```python
import boto3

def get_secrets_by_tag(tag_key, tag_value):
    secrets_client = boto3.client('secretsmanager')
    filtered_secrets = {}
    next_token = None

    while True:
        response = secrets_client.list_secrets(
            Filters=[
                {
                    'Key': 'tag-key',
                    'Values': [tag_key]
                }
            ],
            MaxResults=100,
            NextToken=next_token
        )

        for secret in response['SecretList']:
            secret_name = secret['Name']
            tags_response = secrets_client.describe_secret(SecretId=secret_name).get('Tags',[])
            found_tag = False
            for tag in tags_response:
              if tag['Key'] == tag_key and tag['Value'] == tag_value:
                found_tag = True
                break
            if found_tag:
                try:
                    secret_value_response = secrets_client.get_secret_value(SecretId=secret_name)
                    filtered_secrets[secret_name] = secret_value_response.get('SecretString') or secret_value_response.get('SecretBinary')
                except Exception as e:
                    print(f"Error retrieving secret {secret_name}: {e}")
            
        next_token = response.get('NextToken')
        if not next_token:
            break
    return filtered_secrets


if __name__ == '__main__':
    tag_key = 'Environment'
    tag_value = 'Production'
    secrets = get_secrets_by_tag(tag_key, tag_value)
    for name, value in secrets.items():
      print(f"Secret (tag: {tag_key}={tag_value}): {name}: Value: {value[0:20]} ...")
```

Here, we've added a filtering mechanism. Instead of pulling all secrets, I'm using the `Filters` parameter in the `list_secrets` function to filter based on a specific tag. It's important to note, aws requires you to search based on a 'tag-key'. Therefore, you must iterate through tags of each secret to see if it contains a value. This came in extremely handy when we needed to pull only the production database credentials, allowing us to target the secrets we really needed. The code includes a `describe_secret` call and filters based on a matching key-value pair, then retrieves the secret. Again, output is capped to the first 20 characters.

**Snippet 3: Using a Boto3 Resource (Simplified Iteration)**

```python
import boto3

def get_secrets_resource_approach():
    secrets_manager = boto3.resource('secretsmanager')
    all_secrets = {}
    for secret in secrets_manager.secrets.all():
      try:
        if secret.rotation_enabled == False:
          secret_value = secret.get_secret_value()
          all_secrets[secret.name] = secret_value.secret_string or secret_value.secret_binary
        else:
          print(f"Skipping rotated secret: {secret.name}")
      except Exception as e:
        print(f"Error retrieving secret {secret.name}: {e}")
    return all_secrets


if __name__ == '__main__':
    secrets = get_secrets_resource_approach()
    for name, value in secrets.items():
        print(f"Secret (resource approach): {name}: Value: {value[0:20]} ...")
```

This final example demonstrates how you can use the boto3 resource, which provides a more object-oriented approach and a slightly more concise syntax. While the function is simplified it internally performs a similar process of pagination and iterative retrieval, it does hide these complexities behind a simpler interface. However, you will still need to handle exceptions. Again the output is capped for space reasons.

Remember, managing secrets needs to be handled with great care. Access permissions using IAM roles are absolutely critical. Make sure the execution role has at minimum the `secretsmanager:ListSecrets`, `secretsmanager:DescribeSecret`, and `secretsmanager:GetSecretValue` permissions. Without them, the scripts will fail miserably. For a deep dive on securing and structuring your secrets, i'd recommend reading "Designing Data-Intensive Applications" by Martin Kleppmann, which offers a lot of insight on data management, including how to handle sensitive data securely. Another incredibly helpful book is "AWS Certified Security Specialty Exam Guide" by Ben Piper and David Clinton for specific practices related to security within the AWS ecosystem.

The snippets provided here will definitely get you moving in the right direction. However, always test the changes on a development environment before running any production changes, and follow the principle of least privilege when granting IAM access. Through experience, I've learned that while getting secrets can be easy, ensuring it's done securely and effectively requires thought and planning. It's not just about the code, it's the whole process around it.
