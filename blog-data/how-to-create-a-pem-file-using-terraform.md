---
title: "How to create a PEM file using Terraform?"
date: "2024-12-23"
id: "how-to-create-a-pem-file-using-terraform"
---

Okay, let’s address crafting PEM files within the terraform ecosystem. It's something I've dealt with quite a bit, especially when deploying infrastructure that relies on certificate authentication. It’s more common than you might think, particularly when you're setting up things like mutual tls (mtls), vpn gateways, or even specific service configurations that require signed certificates. While terraform doesn’t directly generate certificates, it’s often our tool of choice to manage the final deployment, including the creation of the necessary pem files that encapsulate keys and certs.

The core issue revolves around the fact that pem files are essentially text-based containers holding encoded data—usually in base64—of private keys and certificates. Terraform excels at managing infrastructure as code, which in this instance includes the ability to create files with arbitrary content. Therefore, our approach isn't about terraform *generating* the cryptographic material, but rather about managing and deploying it. You’ll typically have either pre-generated keys and certificates or use another mechanism, like vault or a dedicated certificate authority, to provide the content.

Essentially, you're often dealing with variables in your terraform configuration. These variables will hold the base64 encoded content, which we then use to construct the pem file.

Let’s break it down with some code examples, keeping in mind that while the actual content here would be a placeholder, in a real-world scenario you would be pulling this data from a secure source.

**Example 1: Basic Private Key PEM File Creation**

This first example creates a pem file containing a private key. We're assuming the existence of a variable called `private_key_base64` that holds the base64 encoded private key.

```terraform
variable "private_key_base64" {
  description = "Base64 encoded private key"
  type        = string
  sensitive   = true  # Always mark sensitive data
}

resource "local_file" "private_key_pem" {
  content  = <<EOF
-----BEGIN PRIVATE KEY-----
${base64decode(var.private_key_base64)}
-----END PRIVATE KEY-----
EOF
  filename = "private_key.pem"
  file_permission = "0600" # restrict access for security
}
```

Here, `local_file` is a resource provided by the terraform `local` provider. It’s excellent for managing local files, particularly during development.  The `content` attribute uses a here-document (<<EOF...EOF) to form the complete pem file structure, inserting the decoded variable using the `base64decode()` function. Setting `file_permission` is crucial for security, ensuring that the key is readable only by the user running the terraform process. This would typically be root or a dedicated service account, depending on your deployment strategy. I have personally seen production outages caused by overly permissive key files; it's an area to pay close attention to.

**Example 2: Certificate PEM File Creation**

Building on the first example, here's how to create a certificate pem, which often contains multiple certificates in a chain. This example uses a base64 encoded certificate chain held in a variable called `certificate_chain_base64`.

```terraform
variable "certificate_chain_base64" {
    description = "Base64 encoded certificate chain"
    type        = string
    sensitive   = true
}

resource "local_file" "certificate_pem" {
  content  = <<EOF
${base64decode(var.certificate_chain_base64)}
EOF
  filename = "certificate.pem"
 file_permission = "0644"  # usually publicly readable
}
```

This follows a similar structure to the first example, using `local_file`. The `content` is populated with the decoded certificate chain content, and the resulting file is named `certificate.pem`. The `file_permission` here is set to "0644", meaning publicly readable as certificates are generally not sensitive material. This permission is a standard practice, but you should always check what best suits your requirements.

**Example 3: Combining Key and Certificate into a Single PEM File**

Often, it's beneficial to combine the private key and certificate (and any intermediate certificates) into a single pem file. This is particularly common for client-side authentication. This example uses two variables: `private_key_base64` and `full_certificate_chain_base64`

```terraform
variable "private_key_base64" {
  description = "Base64 encoded private key"
  type        = string
  sensitive   = true
}


variable "full_certificate_chain_base64" {
    description = "Base64 encoded full certificate chain (including leaf and intermediate)"
    type        = string
    sensitive   = true
}

resource "local_file" "client_credentials_pem" {
 content = <<EOF
-----BEGIN PRIVATE KEY-----
${base64decode(var.private_key_base64)}
-----END PRIVATE KEY-----
${base64decode(var.full_certificate_chain_base64)}
EOF
  filename = "client_credentials.pem"
  file_permission = "0600" # keep private
}
```

This example shows how to concatenate the private key and the full certificate chain into one file. The final file will contain first the private key, followed by the certificate(s). It’s crucial that when providing this file to other applications or systems, they’re expecting this format. Once again, the file permission is set to `0600` as it includes sensitive key data. In production, managing and distributing this sort of file securely is paramount.

**Important Considerations and Further Reading:**

Several things need careful thought when implementing this pattern in practice. First, **security is non-negotiable**. Don’t embed your secrets directly in your terraform configuration. Instead, leverage secure parameter stores like aws ssm parameter store, azure key vault, or hashicorp vault. These systems offer secure storage and retrieval of sensitive data. There are terraform providers for each that allow secure retrieval of these variables at runtime. This dramatically reduces the risk of exposure.

Second, consider how the content is generated and maintained. If you have a certificate authority, then integrating terraform with it using its api will be a more effective approach than manually fetching and encoding the certificate.

Lastly, when generating certificates, the process needs to be well-understood and secure. I’ve seen teams unknowingly generate certificates with weak keys, misconfigured extensions, or improperly managed certificate lifetimes. Ensure you adhere to best practices for certificate and key management.

To further your knowledge on this, I recommend:
*   **"Cryptography Engineering: Design Principles and Practical Applications" by Niels Ferguson, Bruce Schneier, and Tadayoshi Kohno**: An excellent book for understanding the underlying principles of cryptography and secure key management.
*   **The official terraform documentation on the ‘local’ provider and variable management**: The terraform documentation always remains a valuable, up-to-date resource for specifics regarding its features.
*   **RFC 7468: Textual Encodings of PKIX, PKCS, and CMS Structures**:  For precise understanding of the pem encoding format.

In summary, creating pem files with terraform isn’t about generating cryptographic material, it's about orchestrating and deploying it securely, efficiently, and following best practices. These examples provide a foundation; the key to success lies in implementing it within a comprehensive security framework and adapting the techniques to your specific needs.
