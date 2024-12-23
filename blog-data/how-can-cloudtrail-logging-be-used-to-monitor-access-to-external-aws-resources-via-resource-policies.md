---
title: "How can CloudTrail logging be used to monitor access to external AWS resources via resource policies?"
date: "2024-12-23"
id: "how-can-cloudtrail-logging-be-used-to-monitor-access-to-external-aws-resources-via-resource-policies"
---

Alright, let’s tackle this. I remember back in the early days of a previous project, we had a similar challenge concerning external access policies – it's not always straightforward to see *exactly* who's doing what when you're granting access to external entities. We leaned heavily on CloudTrail, and here's the breakdown of how we managed to get detailed insights into such access:

The core challenge lies in the fact that resource policies, like those attached to s3 buckets or kms keys, govern who *can* do what, but they don’t directly create log events of access. Instead, CloudTrail logs *API calls*. Therefore, to monitor access granted by resource policies, you have to translate the "who" in the logs to the effective permissions resulting from those policies. This is where a good understanding of CloudTrail event structure and policy evaluation logic becomes vital. Essentially, CloudTrail allows you to see when an entity *attempts* an action, and a properly configured system means those attempts will often fail or succeed *because* of the resource policy.

Let's start with the general principle. CloudTrail provides records of aws api calls. When a principal, internal or external, attempts to access a resource, that action triggers an api call. The relevant aspects that are important for us are the `userIdentity` section within each event – which tells us who made the request – and the `eventName`, which details what they tried to do (e.g., `s3:GetObject`, `kms:Decrypt`). Crucially, if the resource policy denies the action, this failure also appears as an API event in CloudTrail; similarly, if the action is allowed, we see that as well. The 'errorCode' will differ in both scenarios, helping differentiate the outcome.

The resource policy itself isn't logged directly. Instead, it's the effect of that policy, observed through API events logged to CloudTrail, that provides your monitoring. To effectively monitor external access you need to analyze the events and infer the permission model. This process is key for ensuring security and compliance because it lets you verify if access is being used as expected, and identify any potentially malicious actions.

Here are a few key strategies to leverage CloudTrail for this specific monitoring scenario:

1. **Filtering and Analysis of Events**: This involves querying and parsing the CloudTrail logs to identify attempts by external principals. We look for entries where the `userIdentity` field indicates an external account (or role) and the `eventName` corresponds to actions on resources we're interested in, such as S3 objects, KMS keys, or sqs queues. Crucially, you should always monitor for 'failure' events. These can be the initial indication of an unwanted permission configuration.

2. **Correlation with Resource Policy Configuration**: We need to periodically analyze resource policies to understand exactly what permissions they grant. CloudTrail events only show the usage, whereas resource policies explicitly state permissions. So, if an allowed action appears that was not anticipated it should be a signal to audit policy settings. The critical piece is to correlate the logged usage with the declared policy – this is essential to understand exactly why actions are permitted or denied.

3. **Automated Alerting based on Events**: Once you have a good understanding of your baseline usage, set up automated alerts based on patterns you discover. For example, an external principal attempting access to a sensitive KMS key would be a red flag. This means setting up systems that can process CloudTrail data – it is too difficult to review manually if the scale is large. This process is important to proactively respond to unauthorized access attempts.

To illustrate this, I will provide code snippets using python and boto3, aws's official python sdk, showing how to perform these steps. Consider that the code below assumes you have set up credentials correctly on your local system for accessing aws resources:

**Snippet 1: Fetch CloudTrail Events for a Specific Resource**

This snippet retrieves CloudTrail events filtered by resource and event type.
```python
import boto3
import json

def get_cloudtrail_events(resource_arn, event_name_prefix, region):
    """Fetches CloudTrail events for a specific resource."""
    client = boto3.client('cloudtrail', region_name=region)
    events = []
    paginator = client.get_paginator('lookup_events')
    pages = paginator.paginate(
        LookupAttributes=[
            {
                'AttributeKey': 'ResourceName',
                'AttributeValue': resource_arn
             },
             {
                 'AttributeKey': 'EventName',
                'AttributeValue': event_name_prefix
             }
         ]
    )
    for page in pages:
        for event in page['Events']:
            events.append(event)
    return events

if __name__ == '__main__':
    resource_arn = "arn:aws:s3:::my-secure-bucket"
    event_prefix = "s3:Get" # search for GET events
    region = 'us-east-1'

    events = get_cloudtrail_events(resource_arn, event_prefix, region)
    for event in events:
        print(json.dumps(event, indent=2))
```
This code provides a basic extraction of cloudtrail logs, this can be used to filter by other parameters or look for different errors, like access-denied events, or those that originate from different regions than your own.

**Snippet 2: Extract User Identity from CloudTrail Event**

This snippet extracts the relevant `userIdentity` information for analysis. This is key to know *who* made a request.

```python
import json

def extract_user_identity(event):
    """Extracts user identity information from a CloudTrail event."""
    if 'userIdentity' in event:
        user_identity = event['userIdentity']
        identity_type = user_identity.get('type')
        if identity_type == 'IAMUser':
            return f"IAMUser: {user_identity.get('userName')}"
        elif identity_type == 'AssumedRole':
            return f"AssumedRole: {user_identity.get('sessionContext').get('sessionIssuer').get('userName')}"
        elif identity_type == 'AWSAccount':
            return f"AWSAccount: {user_identity.get('accountId')}"
        elif identity_type == "FederatedUser":
            return f"FederatedUser: {user_identity.get('userName')}"
        else:
            return f"Unknown Type: {identity_type}"
    else:
        return "No User Identity information"

if __name__ == '__main__':
    # Assuming 'events' is obtained from the previous example
    resource_arn = "arn:aws:s3:::my-secure-bucket"
    event_prefix = "s3:Get"
    region = 'us-east-1'
    events = get_cloudtrail_events(resource_arn, event_prefix, region)

    for event in events:
        user_info = extract_user_identity(event)
        print(f"User: {user_info}")

```

This snippet shows how to extract the identity of a user from the logged events. Note that various options are possible, such as IAM roles, users or externally federated identities.

**Snippet 3: Check for Access Denied Events**

This checks for access denied responses, as these are the main indicators that policies are acting as intended.

```python
import json

def check_access_denied(event):
  """checks if an access denied event exists in cloudtrail log"""
  if event.get('errorCode') == 'AccessDenied':
      return True
  else:
      return False

if __name__ == '__main__':

    resource_arn = "arn:aws:s3:::my-secure-bucket"
    event_prefix = "s3:Get"
    region = 'us-east-1'
    events = get_cloudtrail_events(resource_arn, event_prefix, region)

    for event in events:
        if check_access_denied(event):
            print(f"Access Denied Event: {json.dumps(event, indent=2)}")
```

This illustrates a very basic approach to identifying `accessdenied` results in the cloudtrail logs. Note that other errors exist.

For those wanting a deeper dive, I'd recommend focusing on the official AWS documentation for CloudTrail and resource policies. The AWS Security Blog often features posts about practical security patterns with CloudTrail. Additionally, “AWS Certified Security Specialty Exam Guide” by Ben Piper and David Pelly is an excellent resource for understanding the general security principles and how they interact with the service. The key is to move past the basic understanding of CloudTrail, and learn how to parse and correlate the resulting data. Understanding these concepts thoroughly is absolutely necessary to properly secure your cloud environment. CloudTrail alone isn't a security solution; it's an *auditing* tool that provides critical visibility. You have to be proactive in your analysis of its results to create a functional security system.
