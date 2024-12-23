---
title: "Can cloudtrail logs be written to an S3 bucket in a different AWS account?"
date: "2024-12-23"
id: "can-cloudtrail-logs-be-written-to-an-s3-bucket-in-a-different-aws-account"
---

Alright, let’s tackle this. It’s a common scenario, and I’ve personally navigated it more than a few times, especially in multi-account aws setups where security and centralized logging are paramount. So, can cloudtrail logs be written to an s3 bucket in a different aws account? The short answer is, emphatically, yes. But there are important caveats and configurations to understand for a successful and secure implementation. It’s not just a matter of pointing cloudtrail to a different account's bucket; there’s more to the story.

The primary reason you'd want to do this is for centralized security and auditing. Instead of having numerous s3 buckets scattered across several aws accounts holding cloudtrail logs, it’s significantly easier to manage, monitor, and analyze logs from a single, designated “logging” account. This also strengthens security posture as logs become less accessible to potentially compromised accounts. Let's break down how this works practically and go through the permissions and configurations required.

The core of this setup is establishing the correct permissions. We aren’t simply *copying* logs from one account to another; we are configuring cloudtrail to *directly deliver* logs to a bucket in a different account. This requires bucket policy adjustments on the destination bucket and the role or user that is configured in cloudtrail in the *source* account.

Let's delve into the technical details with examples, demonstrating how this is achieved.

**Example 1: The Destination Bucket Policy**

First, let's look at the bucket policy on the destination s3 bucket, in what we’ll call the “logging account.” This policy *must* explicitly allow cloudtrail from other accounts to write to it. Crucially, the `aws:SourceArn` condition is used to restrict the writes to CloudTrail service principals only.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AWSCloudTrailAclCheck",
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudtrail.amazonaws.com"
            },
            "Action": "s3:GetBucketAcl",
            "Resource": "arn:aws:s3:::your-logging-bucket-name"
        },
        {
            "Sid": "AWSCloudTrailWrite",
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudtrail.amazonaws.com"
            },
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::your-logging-bucket-name/AWSLogs/*",
            "Condition": {
                "StringEquals": {
                    "aws:SourceArn": [
                        "arn:aws:cloudtrail:us-east-1:111122223333:*",  // account-id of source account 1
                         "arn:aws:cloudtrail:us-west-2:444455556666:*"  // account-id of source account 2
                       // Add more as necessary. Always include the source region
                    ]
                },
                 "StringEquals": {
                        "s3:x-amz-acl": "bucket-owner-full-control"
                  }
            }
        }
    ]
}
```
In this example, the bucket named `your-logging-bucket-name` is explicitly allowing `cloudtrail.amazonaws.com` service principals from accounts with ids `111122223333` in region `us-east-1` and `444455556666` in `us-west-2`  to write objects (logs) into the `AWSLogs` subfolder. Important to note here, we specify an `s3:x-amz-acl` condition. CloudTrail requires this to allow it to grant full control to the bucket owner. Without this, CloudTrail will throw an error when attempting to deliver the logs.

**Example 2: Cloudtrail Role or User Permissions**

On the source accounts, the CloudTrail service itself doesn’t assume a role, but you might need to ensure the role/user that *creates* the trail has sufficient permissions. In the source account, you will either configure a CloudTrail role to write these logs, or you will need to ensure that the user configured to use CloudTrail has the necessary permissions to create a trail that delivers logs to a bucket in another account. It’s useful to assume a role to make this process much more controlled and secure. Here's an example of the policy that can be attached to such a role or the user configured to administer CloudTrail:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
               "s3:GetBucketAcl",
               "s3:GetBucketLocation",
               "s3:PutObject"
            ],
            "Resource": [
              "arn:aws:s3:::your-logging-bucket-name",
              "arn:aws:s3:::your-logging-bucket-name/AWSLogs/*"
            ]
        }
     ]
}
```

This policy grants permissions to get the bucket's acl (required by CloudTrail), get the bucket's location, and put the CloudTrail log objects within the logging bucket's `AWSLogs` folder. You will need to replace `your-logging-bucket-name` with the actual bucket name of the logging account's bucket. This permission would be attached to a role, which the source account cloudtrail would be configured to use. If instead you decide to not configure a role for cloudtrail to write the logs using, it must be attached to the user who will configure cloudtrail.

**Example 3: Cloudtrail Configuration on Source Account**

The final step is to actually configure CloudTrail. On the source account, when you configure the trail, you will select "Log to an s3 bucket in another account". You will then provide the arn of the bucket in the logging account, replacing `your-logging-bucket-name` as required.

```
aws cloudtrail create-trail \
    --name my-cross-account-trail \
    --s3-bucket-name arn:aws:s3:::your-logging-bucket-name \
    --s3-key-prefix AWSLogs/ \
    --region us-east-1 \
    --is-logging true \
    --include-global-service-events \
    --no-log-file-validation-enabled \
    --profile source-account-admin-profile
```
This command demonstrates a basic aws cli command to create the trail to write to a logging account. The important part is the `--s3-bucket-name` parameter that provides the s3 arn. The `--profile source-account-admin-profile` is only there as an example of the configuration that is using a profile within the aws cli to assume the appropriate permissions to configure the trail.

**Key Considerations and Best Practices:**

1.  **Encryption**: Always ensure that both your source and destination buckets have server-side encryption enabled to protect the logs in transit and at rest. Consider server-side encryption with kms keys that are also centrally managed in your logging account.
2.  **Bucket Versioning**: Enable s3 bucket versioning on the logging bucket. This protects your logs from accidental deletion and provides a history of changes.
3.  **Lifecycle Policies**: Implement lifecycle rules on your logging bucket to archive and eventually delete older logs. This helps manage costs. Consider sending logs into a service like Amazon S3 Glacier Deep Archive for very long-term storage of logs.
4.  **Centralized Auditing**: Once logs are centralized, utilize tools like amazon athena or cloudwatch logs insights for log analysis and alerting.
5.  **Least Privilege**: Ensure that you grant only the required permissions to both the bucket policy and the roles used in cloudtrail and adhere to the principle of least privilege, granting no more than necessary access to resources. Review these permissions regularly.
6.  **Trail Validation**: While I turned off validation in example three's command, it’s usually a good idea to enable log file validation for cloudtrail, as that prevents unintended modification or deletion of log files. I disabled it here for the purposes of simplicity.

**Further Reading and Resources**

For a deeper understanding of the concepts discussed, I recommend the following resources:

*   *AWS CloudTrail User Guide*: The official AWS documentation for CloudTrail is invaluable. Pay particular attention to the section on “Configuring Logging Across AWS Accounts”
*   *AWS Security Best Practices*: This AWS whitepaper covers a range of security practices that should form your security foundation, including centralized logging strategies.
*   *Mastering AWS Security* by Chris Farris: This is a good practical text that covers CloudTrail, permissions, and the broader landscape of AWS security.

In my experience, setting up cross-account logging takes careful planning and precise configurations. It's not uncommon to encounter permission issues or misconfigurations at first. However, by carefully considering the policy requirements and implementing best practices you’ll be able to configure cross-account logging effectively, which is key to achieving a more robust and manageable security posture. The key is to test your setup thoroughly and iterate. Hopefully, this provides a solid starting point and addresses the core of your question. Let me know if there's anything else you would like to discuss.
