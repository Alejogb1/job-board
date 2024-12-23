---
title: "Why can't I pull an image from ECR and why can't Portainer seem to share credentials on EC2?"
date: "2024-12-23"
id: "why-cant-i-pull-an-image-from-ecr-and-why-cant-portainer-seem-to-share-credentials-on-ec2"
---

Alright, let's unpack this. It’s a scenario I've encountered more times than I care to count, and the frustration is certainly understandable. You're attempting to pull a container image from Amazon Elastic Container Registry (ECR) on an EC2 instance, managed perhaps with something like Portainer, and the whole thing is falling apart because of credential issues. This isn't an uncommon hurdle. Let's dive into why this occurs and how to systematically address it.

Essentially, the core problem boils down to authorization and authentication—or rather, a failure of either. When an EC2 instance tries to access ECR, it needs to prove its identity, demonstrating it has the necessary permissions. It's not automatically trusted simply because it's running within AWS's infrastructure. Additionally, Portainer, while extremely useful, isn't magically sharing credentials across your system, and relies on properly configured authentication mechanisms.

The first key piece is understanding how AWS authentication works. It often defaults to utilizing *instance profiles* when running on EC2. An instance profile is an IAM role associated with your EC2 instance. This role should ideally have permissions allowing access to ECR. If you’ve not explicitly assigned the role or granted the correct permissions, ECR will reject your requests.

Secondly, consider that the Docker daemon running on your EC2 instance, which is used to pull the image, needs to know how to authenticate with ECR. It doesn't inherit AWS credentials by osmosis. Usually, this is done by getting temporary credentials (that refresh automatically), based on that instance's associated IAM role, and storing it securely to be used by the Docker runtime. When this process fails – and it can be due to several reasons that I’ll cover in a moment – the `docker pull` operation will be denied.

Portainer adds an extra layer here. It needs to be configured to be aware of how to authenticate with ECR; it won’t automatically use the instance’s IAM role if it's not configured to. If you are using Portainer to manage Docker on your EC2 instance, then Portainer itself needs to have access to those credentials.

Let’s get into the specifics with code examples. Assume we have an EC2 instance with an IAM role named `ec2-container-role`. First, we need to ensure the role has sufficient permissions. This part isn't code you execute on the EC2 instance itself, but part of the configuration in the AWS console or through the AWS CLI. You can check it through the console's IAM section, or through the CLI with something like:

```bash
aws iam get-role --role-name ec2-container-role
```
The result of that will include the attached policies. Make sure, at minimum, it has the `AmazonEC2ContainerRegistryReadOnly` policy or equivalent. You can create a custom policy if you prefer, but `AmazonEC2ContainerRegistryReadOnly` grants read access needed to pull images.

Now, the EC2 itself. If you are simply trying to pull an image via the docker CLI, it relies on the AWS CLI credentials provider, by default. To verify that is working, you can use the AWS CLI.

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com
```

Here, `us-east-1` should be replaced with your desired region, and `<your-account-id>` with your AWS account ID. This command obtains temporary credentials and uses them to log in to ECR. If this works, then you can pull images, but not directly with the `docker pull` command unless you configure the credential helper as described below.

If the login fails with an authentication error, it strongly indicates that your IAM role is missing permissions or that the AWS cli credentials provider is not set up properly. This can be configured using the awscli. Alternatively, you can use a docker credential helper as described below.

Now, let's consider a Portainer scenario. Portainer, ideally, should be configured to use the same instance profile mechanism on EC2. Portainer utilizes the Docker daemon on the underlying OS, and that daemon *must* be authenticated. Here’s how you might configure Docker to use instance profile credentials by setting up a credential helper. This is a critical part that is often overlooked.

```bash
# Ensure you have awscli installed: sudo apt install awscli (or yum install)
# Install docker-credential-ecr-login: pip3 install docker-credential-ecr-login
mkdir -p ~/.docker
cat <<EOF > ~/.docker/config.json
{
  "credsStore": "ecr-login"
}
EOF
chmod 600 ~/.docker/config.json
```
The above snippet will create a `config.json` file that instructs docker to use the `ecr-login` credential helper, which handles the temporary credential fetching automatically. You need to ensure that `docker-credential-ecr-login` is available in the path of your docker runtime, or specify its full path in the configuration file (not shown for clarity). This method also eliminates the need to run `aws ecr get-login-password` manually and then login to the ecr registry manually. Docker now handles the authentication to the ECR registry automatically in the background. If Portainer is using the docker cli on your system to manage docker, then it would use these credentials as well.

If Portainer is running as a container itself, make sure that the portainer container is also launched with the instance role and configured to use the credential helper. The instance profile is granted to the container runtime, not just to the machine.

Finally, some common pitfalls to be aware of. Sometimes you have older EC2 instances that were created *before* the IAM role was set. In these scenarios, the instance won't refresh its assumed role until it's restarted, or the metadata service is queried to reload it. Ensure your instance's metadata settings are set appropriately to allow for retrieving credentials from IAM. Another frequent cause is when the instance role is changed but then the EC2 instance is not restarted or reloaded properly.

As for recommended resources, I’d strongly suggest the *AWS Identity and Access Management (IAM)* documentation on their site, specifically focusing on Instance Profiles and how to assign them. For a deeper understanding of how the docker credential helpers work, I recommend referring to the official Docker documentation on credential stores and helpers. Understanding the nuances of how Docker handles credentials will solve many issues down the line. Also, consider exploring the AWS SDK documentation if you need to programmatically authenticate to ECR from an application (not covered here, but important). Further, the documentation for `docker-credential-ecr-login` can provide detailed information on its usage and capabilities.

In short, the inability to pull an image from ECR coupled with Portainer's inability to use the credentials stems from misconfigurations in AWS IAM roles, Docker daemon authentication mechanisms, and Portainer's reliance on properly configured credentials within the Docker environment. By meticulously checking permissions, employing the docker credential helper, and understanding the propagation of IAM roles across your system, you should be able to get your container images pulling successfully from ECR. This process usually resolves a vast majority of these types of issues.
