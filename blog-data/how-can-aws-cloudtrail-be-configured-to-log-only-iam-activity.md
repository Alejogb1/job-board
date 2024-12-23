---
title: "How can AWS CloudTrail be configured to log only IAM activity?"
date: "2024-12-23"
id: "how-can-aws-cloudtrail-be-configured-to-log-only-iam-activity"
---

Alright, let's tackle this one. It's a common need, filtering out the noise from CloudTrail to focus specifically on iam actions. From my own experience, working on environments with a high volume of cloud activity, I've often had to fine-tune CloudTrail configurations. A full, unfiltered log can be a firehose of information, making it challenging to isolate the relevant events, particularly for security audits or compliance tracking related to iam.

The default behavior of CloudTrail is to log all api activity across your aws account, or organization, if configured that way. This includes calls made by the aws console, cli, sdks, and other aws services. While comprehensive, it can make focusing on IAM related activity a bit like searching for a needle in a haystack. So, to log *only* iam actions, we need to leverage CloudTrail's "data events" and "management events" filters combined with a bit of understanding how those are structured.

First, let’s distinguish these two event types. Management events are actions performed on the control plane of aws resources; this includes things like provisioning a new ec2 instance, creating an s3 bucket, and, importantly for our case, actions related to iam. Data events, on the other hand, are operations performed on the actual data contained within these resources – for instance, reading or writing to an s3 object. Since iam is about managing identities and access – a control plane activity – we’ll focus on management events primarily.

Now, the key to filtering *only* iam activity relies heavily on the `eventName` field in each log entry. Within the management events, actions like `CreateUser`, `AttachRolePolicy`, `UpdateUser`, and so on, will be represented by their corresponding `eventName` value. To capture *only* these actions, we’d need to filter logs based on this field. CloudTrail doesn't offer a prebuilt, dedicated filter for "iam only", requiring us to be a bit more explicit.

We achieve this filtering via an advanced event selector within the CloudTrail trail configuration. This involves specifying `eventName` as the attribute we want to evaluate and providing a list of desired values associated with iam actions. Let’s look at a basic example to illustrate this using a cloudformation template. I’ve trimmed the noise to focus specifically on the relevant portion.

```yaml
Resources:
  CloudTrailTrail:
    Type: 'AWS::CloudTrail::Trail'
    Properties:
      TrailName: 'iam-activity-trail'
      S3BucketName: 'your-s3-bucket-for-logs' #replace with your bucket
      IsLogging: true
      EventSelectors:
        - DataResources: []
          IncludeManagementEvents: true
          ReadWriteType: 'All'
          # filter for IAM actions
          # This is where the magic happens
          AdvancedEventSelectors:
            - Name: iam-filter
              FieldSelectors:
                - Field: 'eventName'
                  Equals:
                    - 'CreateUser'
                    - 'DeleteUser'
                    - 'UpdateUser'
                    - 'CreateRole'
                    - 'DeleteRole'
                    - 'AttachRolePolicy'
                    - 'DetachRolePolicy'
                    - 'PutUserPolicy'
                    - 'DeleteUserPolicy'
                    - 'CreatePolicy'
                    - 'DeletePolicy'
                    - 'CreateGroup'
                    - 'DeleteGroup'
                    # add more iam actions here as needed
```

This snippet defines a CloudTrail trail and uses an `AdvancedEventSelector` to filter based on specific event names associated with common iam activities. It’s important to note this list isn't exhaustive, and you would likely need to extend it to include all the iam events you want to monitor. For a comprehensive list, refer to the aws documentation on "cloudtrail supported event names." This demonstrates the fundamental principle - defining a list of `eventName` values.

Now, you might be thinking, "that's a lot to type, is there a better way to construct this?" Yes, there are some better practices, for example, using scripting in combination with aws cli. Let's look at a python script using boto3, the aws sdk for python, that allows you to achieve a similar outcome:

```python
import boto3
import json

cloudtrail = boto3.client('cloudtrail')

iam_events = [
    'CreateUser',
    'DeleteUser',
    'UpdateUser',
    'CreateRole',
    'DeleteRole',
    'AttachRolePolicy',
    'DetachRolePolicy',
    'PutUserPolicy',
    'DeleteUserPolicy',
    'CreatePolicy',
    'DeletePolicy',
    'CreateGroup',
    'DeleteGroup'
]

try:
    response = cloudtrail.update_trail(
        Name='your-trail-name', # replace with your trail name
        AdvancedEventSelectors=[
            {
                'Name': 'iam-filter',
                'FieldSelectors': [
                    {
                        'Field': 'eventName',
                        'Equals': iam_events
                    }
                ]
            }
        ]
    )
    print(json.dumps(response, indent=2))

except Exception as e:
    print(f"An error occurred: {e}")

```

This python script utilizes boto3 to interact with cloudtrail api, builds the necessary filter based on our `iam_events` list, and updates an existing trail with those settings. This is advantageous as it is more maintainable than writing lengthy yaml, allowing you to programmatically handle updates. This demonstrates that we can use programmatic approaches for configuration management.

Furthermore, considering a multi-account environment, where you might want centralized logging and filtering across many accounts, you can leverage aws organizations and cloudtrail organization trails. In that setup, the same filtering principles apply but at the organizational level. To illustrate, this involves creating the trail in the management account and applying the filtering via a similar event selector logic – although, the actual configuration process is somewhat different using aws organizations apis or by defining service control policies that mandate settings.

Here is how it would look in the organizations console, which essentially results in a similar filter config as the prior example, just configured from the management account with organization scope:
```json
{
 "Name": "iam-activity-trail-organization",
 "S3BucketName": "organization-logging-bucket",
 "IsLogging": true,
 "IncludeGlobalServiceEvents": true,
  "EventSelectors": [
    {
      "DataResources": [],
      "IncludeManagementEvents": true,
      "ReadWriteType": "All",
      "AdvancedEventSelectors": [
        {
          "Name": "iam-organization-filter",
          "FieldSelectors": [
             {
              "Field": "eventName",
              "Equals":  [
                    "CreateUser",
                    "DeleteUser",
                    "UpdateUser",
                    "CreateRole",
                    "DeleteRole",
                    "AttachRolePolicy",
                    "DetachRolePolicy",
                    "PutUserPolicy",
                    "DeleteUserPolicy",
                    "CreatePolicy",
                    "DeletePolicy",
                    "CreateGroup",
                    "DeleteGroup"
                   ]
             }
           ]
        }
      ]
    }
  ]
}

```
This example demonstrates the organization level filtering approach. It’s similar to our first code snippet but intended for a larger setup. A key takeaway is that the principles remain consistent.

For further study, I recommend consulting the following authoritative resources:

1.  **"AWS CloudTrail User Guide"**: This is the official documentation and provides the most detailed information on configuring CloudTrail trails, events, and advanced event selectors. It’s essential for understanding the nuances of the service.

2. **“AWS Security Best Practices” Whitepaper:** This document, while broader than just cloudtrail, offers a good understanding of security principles, which tie directly to why such granular logging is often desired. Pay particular attention to the section on logging and monitoring.

3. **“Security Logging in AWS” from the aws security blog:** A great resource for best practices around logging, going beyond just CloudTrail into other logging mechanisms and how they integrate.

4.  **"Mastering AWS Security" by Chris Farris and James Bee:** This book offers a deep dive into various aws security mechanisms, including CloudTrail, and can provide valuable insight into the practical implementation of logging and monitoring.

In conclusion, filtering CloudTrail logs for iam activity involves configuring event selectors to target specific `eventName` values, using either cloudformation, sdk scripting, or through the aws console. While CloudTrail is comprehensive, with careful planning and filtering, you can efficiently monitor activities related to iam, leading to better security practices and more efficient auditing procedures. You need a thorough understanding of the event structure and the specific event names, coupled with scripting and config tools to manage them at scale. Remember to always refer back to official documentation and security best practices to build a resilient and well-monitored environment.
