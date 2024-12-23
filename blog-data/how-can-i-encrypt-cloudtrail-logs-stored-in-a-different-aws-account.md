---
title: "How can I encrypt CloudTrail logs stored in a different AWS account?"
date: "2024-12-23"
id: "how-can-i-encrypt-cloudtrail-logs-stored-in-a-different-aws-account"
---

Right, let's talk about securing CloudTrail logs across aws accounts. It's a scenario I've encountered more times than i'd like to recall, often during compliance audits or in environments with robust security requirements. In my experience, the crucial aspect isn't just enabling encryption; it’s doing so correctly while maintaining access for analysis without creating undue administrative overhead.

The core challenge here is that, by default, CloudTrail logs are written to an s3 bucket. If you're using a centralized logging account – which is highly recommended for security and ease of analysis – this means your trail in a separate account needs to deliver those logs securely into a bucket in a *different* aws account. The encryption process then needs to accommodate this cross-account delivery. There are multiple approaches, but I will focus on the most robust and recommended methods using server-side encryption with kms managed keys. This isn't just about checking a box; it's about understanding the full picture of key management and access control.

The most common method, and the one I usually advise, is using server-side encryption with s3-managed keys (sse-s3) for the *initial* delivery, and then, more importantly, enabling server-side encryption using kms keys (sse-kms) when those logs are delivered to the central logging account's s3 bucket. This involves configuring the s3 bucket policy and the kms key policy to allow this cross-account access. Let's consider the technical steps and associated code examples.

**Understanding the Requirements**

The main point to remember is that the encryption doesn't happen at the CloudTrail *service* level but at the s3 bucket level, where the logs are ultimately stored. The process involves two key accounts:

1.  **Source Account:** The account where CloudTrail is recording events and is generating logs. Let's refer to this as `account_a`.
2.  **Central Logging Account:** The account where the s3 bucket for central logging resides. Let's refer to this as `account_b`.

The goal is to ensure logs in the `account_a` CloudTrail are encrypted when they land in the central logging bucket within `account_b`.

**Step 1: Setting up the s3 Bucket in the Central Logging Account (`account_b`)**

First, in account_b, we create the s3 bucket where the logs will land. We'll need a kms key to encrypt the logs server-side. We configure the kms key to allow `account_a` to write to the bucket and encrypt.

```python
import boto3

# Assumes you have already configured your aws cli credentials
# for account_b

region_name = 'us-east-1' #replace with desired region
kms_client = boto3.client('kms', region_name=region_name)

# 1. Create kms key in account_b

response = kms_client.create_key(
    Description='KMS key for encrypting cloudtrail logs in the logging account',
    KeyUsage='ENCRYPT_DECRYPT',
    CustomerMasterKeySpec='SYMMETRIC_DEFAULT'
)

key_arn = response['KeyMetadata']['Arn']
key_id = response['KeyMetadata']['KeyId']

print(f"Created KMS key with ARN: {key_arn}")


# 2. Define the kms key policy to allow account_a to use this key.
# The principal for account_a should be the root account principal
# This principal should look something like "arn:aws:iam::123456789012:root"

account_a_principal = "arn:aws:iam::111111111111:root" #replace with correct account_a arn

policy = {
    "Version": "2012-10-17",
    "Id": "key-default-1",
    "Statement": [
        {
            "Sid": "Enable IAM policies",
            "Effect": "Allow",
            "Principal": {
                "AWS": f"arn:aws:iam::{boto3.client('sts').get_caller_identity().get('Account')}:root"  # Ensure it includes the current account
            },
            "Action": "kms:*",
            "Resource": "*"
        },
         {
            "Sid": "Allow encryption by account a",
            "Effect": "Allow",
            "Principal": {
                "AWS": account_a_principal
            },
            "Action": [
                "kms:Encrypt",
                "kms:GenerateDataKey*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "Allow decryption by account a",
            "Effect": "Allow",
            "Principal": {
               "AWS": account_a_principal
            },
             "Action": [
              "kms:Decrypt",
              "kms:DescribeKey",
              "kms:ReEncrypt*",
              "kms:GenerateDataKey*"
              ],
            "Resource": "*"
        }
    ]
}

kms_client.put_key_policy(
    KeyId=key_id,
    Policy=json.dumps(policy)
)
print(f"KMS key policy updated to allow account_a access.")

# 3. Create the s3 bucket in account_b

s3_client = boto3.client('s3', region_name=region_name)
bucket_name = 'your-central-logging-bucket' #replace with your bucket name
try:
    s3_client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={'LocationConstraint': region_name},
    )
    print(f"Created s3 bucket {bucket_name}")
except s3_client.exceptions.BucketAlreadyExists as e:
     print(f"Bucket {bucket_name} already exists. {e}")
except s3_client.exceptions.BucketAlreadyOwnedByYou as e:
     print(f"Bucket {bucket_name} already owned by you. {e}")


# 4. Configure s3 bucket policy to allow account_a to deliver logs
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AWSCloudTrailAclCheck",
            "Effect": "Allow",
            "Principal": {"Service": "cloudtrail.amazonaws.com"},
            "Action": "s3:GetBucketAcl",
            "Resource": f"arn:aws:s3:::{bucket_name}"
        },
        {
            "Sid": "AWSCloudTrailWrite",
            "Effect": "Allow",
            "Principal": {"Service": "cloudtrail.amazonaws.com"},
            "Action": "s3:PutObject",
            "Resource": f"arn:aws:s3:::{bucket_name}/AWSLogs/{account_a_principal.split(':')[4]}/*", #this path is crucial
            "Condition": {
                "StringEquals": {"s3:x-amz-acl": "bucket-owner-full-control"}
            }
        }
    ]
}

s3_client.put_bucket_policy(
    Bucket=bucket_name,
    Policy=json.dumps(bucket_policy)
)
print(f"S3 bucket policy updated to allow account_a to deliver logs")
```

**Step 2: Configuring the CloudTrail in Source Account (`account_a`)**

Now, in account_a, we need to configure CloudTrail to deliver the logs to the central logging bucket in account_b. Note, the kms key *does not* need to be used here; only the kms key in the logging account is utilized.

```python
import boto3
# Assumes you have already configured your aws cli credentials for account_a

cloudtrail_client = boto3.client('cloudtrail',region_name='us-east-1') #replace with desired region

bucket_name = "your-central-logging-bucket"  # the bucket in account_b
kms_key_arn = "arn:aws:kms:us-east-1:222222222222:key/your-kms-key-id" # replace with actual arn of kms key in account b
logging_account_id = "222222222222" #account_b id

trail_name = 'your-trail-name'#replace with your trail name

cloudtrail_client.update_trail(
   Name = trail_name,
   S3BucketName=bucket_name,
   S3KeyPrefix="AWSLogs/"+ logging_account_id + "/",
   KmsKeyId = kms_key_arn, #this key id is the logging account's kms key
   IncludeGlobalServiceEvents=True,
   IsLogging=True
)

print(f"Updated cloudtrail trail {trail_name} to deliver logs to {bucket_name} with sse-kms encryption.")
```

**Step 3: Validating the Setup**

After setting everything up, it's crucial to check. The way you validate this is by checking the s3 objects in the central logging bucket within `account_b`. If the setup is correct, the server-side encryption type would be `aws:kms`. If it’s `aws:s3`, something has gone wrong, and logs aren't properly encrypted using the kms key within `account_b`. If you need to debug, focus on the kms key policies, bucket policies and also note the prefix `AWSLogs/<account_a id>/*` is vital and it must match the account of the cloudtrail source account in `account_a`.

**Key Considerations**

*   **KMS Key Rotation:** Make sure you have key rotation enabled for your kms keys. This is essential for maintaining good security hygiene and should be part of your kms lifecycle.
*   **Access Control:** The policy examples provided are basic; for production, you would use least-privilege principles, refining resource access.
*   **Cross-Region Considerations:** If your accounts are across different regions, pay attention to region-specific endpoints and resources.
*   **Monitoring:** Implement monitoring and alerting for failed delivery or encryption errors within the central logging account. Cloudwatch logs is extremely useful for this.
*   **Data Analysis:** While the logs are encrypted at rest, make sure any services consuming them, such as athena or other analysis tooling, have the proper kms access.

**Further Reading**

For a deep dive, I highly recommend the following resources:

1.  *AWS Security Best Practices* Whitepaper by AWS - a general overview of best practices for the platform.
2.  *Mastering aws Security* by Chris Farris - This is a technical book with solid practices for security within aws.
3.  The official AWS documentation on CloudTrail, S3, and KMS.

This approach allows for centralized logging with secure encryption using kms keys in your central logging account, providing not just encryption at rest, but also centralized management and security of all CloudTrail logs. Remember, the key is not just to encrypt, but to manage the key lifecycle and access controls diligently to maintain a secure and compliant environment.
