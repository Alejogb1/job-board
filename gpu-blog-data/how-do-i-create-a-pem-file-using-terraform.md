---
title: "How do I create a PEM file using Terraform?"
date: "2025-01-26"
id: "how-do-i-create-a-pem-file-using-terraform"
---

Creating a Privacy Enhanced Mail (PEM) file with Terraform often involves manipulating cryptographic keys or certificates which, while not a core Terraform resource itself, is facilitated through its powerful provisioner system. I've personally encountered this scenario multiple times while automating infrastructure deployments where secure communications and identity management were paramount. The key to understanding how Terraform helps in this process isn't about generating the actual key material – those steps belong elsewhere like the command line using tools like `openssl`, or dedicated Key Management Systems – but rather in safely handling and injecting those pre-generated artifacts into resources that require them, frequently as part of authentication or encryption workflows.

The core principle involves utilizing Terraform’s `local-exec` provisioner or similar to interact with the operating system of the targeted machine and perform file manipulations. A second approach involves injecting PEM data directly into a resource argument by making use of the `file` or `templatefile` function in Terraform. The specific technique applied depends heavily on the desired outcome, be it writing a PEM file directly to disk or embedding it as configuration data. A third method, relevant in some cloud environments, revolves around leveraging secrets managers to store the PEM data securely and then retrieve it as needed using Terraform data sources.

Let’s first address the scenario of writing a PEM file to disk using the `local-exec` provisioner. This is common when deploying servers or virtual machines that need specific keys for services like web servers or database access. Here’s a generalized example:

```terraform
resource "null_resource" "create_pem_file" {
  provisioner "local-exec" {
    command = <<-EOT
      echo "${var.pem_content}" > ${var.pem_path}
      chmod 400 ${var.pem_path}
    EOT
  }

  depends_on = [null_resource.generate_key_pair] # Optional key generation dependency

}

variable "pem_content" {
    type        = string
    description = "The content of the PEM file."
  }
variable "pem_path" {
    type        = string
    description = "The desired file path for the PEM file."
    default = "/tmp/my_key.pem"
}
```

In this example, a `null_resource` is utilized as a placeholder because the focus is file manipulation on the local machine. The `local-exec` provisioner executes a shell command. The command utilizes the `echo` command to write the content from the `pem_content` variable into the file specified by `pem_path`. Crucially, the subsequent command changes the file permissions to 400, a common practice for private key files preventing accidental exposure by other users. The `depends_on` attribute introduces an optional prerequisite that could involve a key generation process that might be executed before this file is created. For instance, `null_resource.generate_key_pair` might be a resource executing a `local-exec` block that uses `openssl` to generate an RSA key pair. The variables `pem_content` and `pem_path` provide flexibility, making the module reusable. It’s important to emphasize the `pem_content` variable would contain the actual string representation of the PEM-formatted data. This approach assumes the PEM content is already available and just needs to be written to the system via Terraform.

Next, consider the case where the PEM content is to be injected directly into a resource. Here's an example involving an AWS EC2 instance initialization where a PEM-encoded certificate is placed within the user data:

```terraform
resource "aws_instance" "example" {
  ami = "ami-xxxxxxxx"
  instance_type = "t2.micro"
  key_name = "your_key_pair"

  user_data = templatefile("${path.module}/user_data.tpl",{
       certificate_pem  = file("${path.module}/cert.pem")
     })

   tags = {
    Name = "example-instance"
   }
}
```

In this scenario, a template file, `user_data.tpl`, is used in conjunction with the `templatefile` function.  The `file` function reads the content of the `cert.pem` located in the same module directory, and it passes that content as a variable (`certificate_pem`) into the template file. Within `user_data.tpl`, which could look like:

```bash
#!/bin/bash
echo "-----BEGIN CERTIFICATE-----
${certificate_pem}
-----END CERTIFICATE-----" > /tmp/server.crt
```

The `certificate_pem` variable’s value, the content of the file `cert.pem`, gets inserted into this bash script. This script is executed during the startup of the EC2 instance. The crucial point here is that the PEM data is never directly stored in the Terraform code, instead, it’s read from a file and passed through. This helps keep the sensitive data separated from your infrastructure-as-code definitions. This technique avoids the need for a `local-exec` provisioner to write a file to disk separately. Instead, the PEM data is directly used by a Terraform resource.

A third, and more robust, approach uses a secrets manager to secure the PEM data. Suppose we use AWS Secrets Manager:

```terraform
data "aws_secretsmanager_secret_version" "cert" {
  secret_id = "arn:aws:secretsmanager:us-west-2:123456789012:secret:my-certificate-xxxxx"
}

resource "aws_api_gateway_domain_name" "example_domain" {
   domain_name = "api.example.com"
  certificate_arn = aws_acm_certificate.example.arn
  certificate_body = data.aws_secretsmanager_secret_version.cert.secret_string
 }
```

Here, the secret, containing the PEM encoded certificate, is stored in AWS Secrets Manager. The `aws_secretsmanager_secret_version` data source reads the secret version from the specified ARN. The value of `data.aws_secretsmanager_secret_version.cert.secret_string` contains the string content of the secret, which in this example is the PEM data, and that content becomes part of the API Gateway domain certificate configuration. The secret is loaded as part of the terraform plan execution. This approach ensures that the certificate material never exists in the raw format within the code or locally. The data is directly loaded from the central key manager when deploying the resource and is not hard-coded into the template file or variables. The `certificate_body` field of the `aws_api_gateway_domain_name` resource expects the raw content of the certificate in PEM format. The use of a secrets manager to hold sensitive PEM data follows best practices.

In my experiences, I've found that choosing the correct method depends on various factors. If the PEM file is needed directly on a local system or a server, the `local-exec` approach is suitable. However, this can lead to security risks if the local machine is not secure.  Direct injection via `templatefile` is handy for cloud resources, keeping the code cleaner and easier to audit. Finally, using secret managers to store and provide sensitive PEM data adds an important layer of security and manages the lifecycle of the certificate.

For further study and a more comprehensive grasp of these techniques, I recommend consulting the official Terraform documentation on provisioners, data sources, and the `templatefile` function. AWS’s documentation on Secrets Manager and ACM is useful when you are using their respective resources. Books and online courses focusing on infrastructure as code practices can further enhance understanding of best practices surrounding key management and sensitive data handling within the Terraform environment. Always focus on the principle of least privilege and ensure that sensitive data, like the contents of PEM files, are handled as securely as possible throughout the development and deployment process.
