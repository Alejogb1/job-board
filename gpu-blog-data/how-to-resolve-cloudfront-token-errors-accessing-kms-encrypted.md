---
title: "How to resolve CloudFront token errors accessing KMS-encrypted files via OAI?"
date: "2025-01-30"
id: "how-to-resolve-cloudfront-token-errors-accessing-kms-encrypted"
---
CloudFront token errors when accessing KMS-encrypted files stored in S3 via an Origin Access Identity (OAI) typically indicate a permissions mismatch within the resource policy of the KMS key, rather than the CloudFront configuration itself. The problem arises because CloudFront, in its default configuration, lacks direct authorization to decrypt objects encrypted with a server-side KMS key. My experience troubleshooting numerous infrastructure deployments has highlighted this particular issue's recurrence, often stemming from a misunderstanding of how OAI and KMS interact. Addressing this involves explicitly granting the OAI the necessary decryption permissions within the KMS key policy, not just relying on S3 bucket policies.

The root cause lies in the principle of least privilege, which requires that any entity accessing a protected resource must be explicitly granted permission to do so. When dealing with KMS-encrypted S3 objects, both S3 and KMS policies must be configured to allow CloudFront to operate through the OAI. The OAI, a special service principal within AWS, acts as a substitute for your AWS account when CloudFront accesses resources. The S3 bucket policy usually only grants the OAI permission to `s3:GetObject`. However, this is insufficient when the object is encrypted by KMS. The server-side encryption means that before S3 can send the object to CloudFront, the associated KMS key needs to authorize the decryption of that specific object for the OAI principal. Without the proper KMS policy, the decryption step fails, causing CloudFront to return access errors, usually HTTP 403 responses.

To resolve these issues, I’ve found that adding a specific statement to the KMS key’s resource-based policy that authorizes the CloudFront OAI to use the key is the most reliable solution. There’s no single 'magic' setting. Instead, we need to explicitly state the permissions.

Here's how this manifests in practice with three distinct examples. First, consider a basic scenario where a KMS key is used to encrypt files stored in S3, and a CloudFront distribution utilizes an OAI to serve content. The initial KMS policy, without modification, may only allow the root user to perform KMS operations and might be similar to this:

```json
{
  "Version": "2012-10-17",
  "Id": "key-default-1",
  "Statement": [
    {
      "Sid": "Enable IAM User Permissions",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::111122223333:root"
      },
      "Action": "kms:*",
      "Resource": "*"
    }
  ]
}
```

In this initial policy, the `Principal` allows only the root user. The critical missing piece is a statement allowing the CloudFront OAI to perform decryption. Adding the necessary statement requires understanding the specific ARN of the OAI. You can obtain this from the CloudFront distribution settings. Let's assume, for the sake of this example, that the OAI ARN is `arn:aws:iam::cloudfront:user/CloudFront OAI (E1234567890ABCD)`.  We will append an additional policy statement to the existing policy. The complete modified KMS policy will then become:

```json
{
  "Version": "2012-10-17",
  "Id": "key-default-1",
  "Statement": [
    {
      "Sid": "Enable IAM User Permissions",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::111122223333:root"
      },
      "Action": "kms:*",
      "Resource": "*"
    },
     {
      "Sid": "Allow CloudFront OAI Decryption",
      "Effect": "Allow",
      "Principal": {
         "AWS": "arn:aws:iam::cloudfront:user/CloudFront OAI (E1234567890ABCD)"
       },
      "Action": "kms:Decrypt",
      "Resource": "*"
    }
  ]
}
```

This additional policy statement, with the `Sid` "Allow CloudFront OAI Decryption," explicitly allows the specified CloudFront OAI to perform the `kms:Decrypt` action on any resource governed by this KMS key. By adding this, we authorize CloudFront to retrieve the objects encrypted with that specific key.

Now, let’s consider a more complex scenario where we want to limit the scope of the decrypt permissions. Instead of allowing decryption for any resource associated with the key, we can restrict it to resources associated with a specific S3 bucket. This requires the use of the `Condition` block, limiting the permission to only occur when the KMS request originates from a specific S3 bucket that contains the object. Let’s say the S3 bucket's ARN is `arn:aws:s3:::my-encrypted-bucket`. Our new policy entry might look like:

```json
    {
      "Sid": "Allow CloudFront OAI Decryption with Condition",
      "Effect": "Allow",
      "Principal": {
         "AWS": "arn:aws:iam::cloudfront:user/CloudFront OAI (E1234567890ABCD)"
       },
      "Action": "kms:Decrypt",
      "Resource": "*",
       "Condition": {
         "ArnEquals": {
           "kms:EncryptionContext:aws:s3:arn": "arn:aws:s3:::my-encrypted-bucket"
         }
       }
    }
```
In this refined policy, the `Condition` block using `ArnEquals` with the `kms:EncryptionContext:aws:s3:arn` key restricts decryption to requests originating from the specified S3 bucket. This adds another layer of security, ensuring that the OAI cannot use the KMS key to decrypt objects from other S3 buckets that may be using the same KMS key. This condition relies on the encryption context being set on the S3 object which is automatically added by default server-side encryption with KMS (SSE-KMS). It’s crucial to remember the encryption context is only present when using SSE-KMS directly, and not for client-side encryption.

Finally, let's examine a case where the CloudFront distribution is configured with multiple Origins, each potentially with its own OAI. Each OAI should have a unique principal.  Let’s say we have an additional OAI with the ARN: `arn:aws:iam::cloudfront:user/CloudFront OAI (F9876543210ZYXW)`.  The KMS policy must include a separate statement for each OAI that needs to perform decrypt operations:
```json
{
  "Version": "2012-10-17",
  "Id": "key-default-1",
  "Statement": [
    {
      "Sid": "Enable IAM User Permissions",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::111122223333:root"
      },
      "Action": "kms:*",
      "Resource": "*"
    },
     {
      "Sid": "Allow OAI 1 Decryption",
      "Effect": "Allow",
      "Principal": {
         "AWS": "arn:aws:iam::cloudfront:user/CloudFront OAI (E1234567890ABCD)"
       },
      "Action": "kms:Decrypt",
      "Resource": "*"
    },
      {
       "Sid": "Allow OAI 2 Decryption",
       "Effect": "Allow",
       "Principal": {
         "AWS": "arn:aws:iam::cloudfront:user/CloudFront OAI (F9876543210ZYXW)"
        },
       "Action": "kms:Decrypt",
       "Resource": "*"
      }
  ]
}
```

This example shows how multiple OAIs can be granted access individually, providing granular control over who can decrypt resources. This structure should be used over wildcarding, as this can be difficult to maintain in larger systems.

These examples illustrate that effectively addressing CloudFront token errors requires a solid grasp of the interplay between KMS and OAI. The key takeaway is the explicit need to grant `kms:Decrypt` permission to the OAI principal in the KMS policy. When troubleshooting, I typically advise starting with basic access, verifying functionality, then progressively implementing tighter conditions as needed.

For additional information beyond these examples, I recommend exploring the official AWS documentation on Key Management Service, Identity and Access Management, and CloudFront. The AWS white papers on security best practices offer invaluable insights into these areas. Furthermore, thoroughly examining the AWS Well-Architected Framework provides a comprehensive guide for secure and reliable cloud deployments, offering specific recommendations related to permissions and encryption strategies. These resources are designed to expand your comprehension and ability to effectively secure your AWS resources.
