---
title: "How do I deploy with Octopus on an ECS cluster?"
date: "2024-12-23"
id: "how-do-i-deploy-with-octopus-on-an-ecs-cluster"
---

Right then, let’s tackle this. I've spent a fair chunk of my career navigating the intricacies of deployment pipelines, and getting Octopus Deploy to play nice with an Amazon ECS cluster is a challenge I've encountered multiple times – each with its own nuanced quirks. It's not a simple 'plug and play' situation, but it's absolutely achievable with a systematic approach. We're essentially orchestrating interactions between a deployment platform designed for broad application management (Octopus) and a container orchestration service tailored for containerized workloads (ECS). The crux lies in bridging that gap.

First, understand that Octopus doesn't directly "deploy" to ECS the way it might to, say, a virtual machine. Instead, it orchestrates the necessary actions within AWS to *cause* a deployment to happen on ECS. This typically involves: building container images, pushing them to a repository (like ECR), and updating the ECS service definition to point to the new image, which then triggers a rolling update in ECS.

My first run at this, about five years ago, involved a rather complex microservices setup where we had multiple ECS services, each with its own deployment cadence. It was a valuable learning experience, largely in that I realized the importance of keeping configurations separate and using parameterization. We initially tried to cram everything into one Octopus project, which quickly became a maintenance nightmare. Lesson one: organization matters.

Now, let's break down how you'd generally configure this.

**The Core Components:**

1.  **Octopus Project:** This is where you'll define the deployment process. It's essential to have a clear understanding of the variables you need – things like image tag, cluster name, service name, etc. I found it effective to use Octopus variables extensively to avoid hardcoding specific environment values. This promotes reusability across different environments.

2.  **Docker Build Process:** Generally, you'll initiate a docker build process either locally or in a CI server (like Jenkins, GitHub Actions). This is a separate step *before* Octopus gets involved. Ensure your build process produces a tagged image that includes relevant build metadata (e.g., a timestamp or Git commit hash).

3.  **Image Push:** The Docker image, once built, is pushed to an accessible container registry like Amazon Elastic Container Registry (ECR). Your Octopus project needs credentials to be able to interact with this registry.

4.  **ECS Task Definition Update:** This is the critical step where Octopus interacts with ECS. You'll use the AWS CLI or AWS SDK to modify the task definition associated with your ECS service. This essentially involves updating the `image` parameter to use the newly built image from ECR.

5.  **ECS Service Update:** Finally, the ECS service itself needs to be updated. This will cause ECS to initiate a rolling update, replacing old container instances with new ones, which are running the updated image.

Now for some practical examples. Let's consider three stages.

**Example 1: Pushing the Docker Image to ECR**

This step assumes your docker image has already been built and tagged locally. We’ll use the AWS CLI in an Octopus step.

```powershell
# Octopus Variable: $DockerImageTag, $AwsAccountId, $AwsRegion, $EcrRepositoryName

#Login to AWS ECR (Ensure appropriate AWS credential is configured in Octopus)
aws ecr get-login-password --region $AwsRegion | docker login --username AWS --password-stdin $AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com

# Push the image to ECR
docker push $AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com/$EcrRepositoryName:$DockerImageTag
```

This is a basic powershell script you would configure as a “Run a Script” step in your Octopus deployment process. We are using variable substitution in Octopus with variables like `$DockerImageTag` and `$EcrRepositoryName` that would need to be defined within the project. Make sure you have an AWS account and an access key setup as a “cloud account” in Octopus.

**Example 2: Modifying the Task Definition**

We will be generating the task definition JSON from our current ECS task definition and update the image tag, then create a new task definition revision. Note that this assumes the use of `jq`, a JSON processor, which you'll need to include in the build environment.

```powershell
# Octopus Variables: $TaskDefinitionName, $AwsRegion, $DockerImageTag, $AwsAccountId, $EcrRepositoryName

# Get the current task definition
$TaskDefinition = aws ecs describe-task-definition --task-definition $TaskDefinitionName --region $AwsRegion | ConvertFrom-Json

# Construct the new image URI
$NewImageUri = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com/$EcrRepositoryName:$DockerImageTag"

# Update the image field in container definitions using jq
$UpdatedTaskDefinition = ($TaskDefinition.taskDefinition | ConvertTo-Json) | jq --arg newimage $NewImageUri '.containerDefinitions[].image = $newimage' | ConvertFrom-Json

# Register the new task definition revision
aws ecs register-task-definition --cli-input-json $UpdatedTaskDefinition  --region $AwsRegion
```

Here, we first fetch the existing task definition, then use `jq` to modify the `image` property within the `containerDefinitions` array. Finally, we register this revised task definition. This script would also be executed in a "Run a Script" Octopus step.

**Example 3: Updating the ECS Service**

This script uses the new task definition ARN created in the previous step to perform a service update:

```powershell
# Octopus Variables: $ServiceName, $AwsRegion, $TaskDefinitionName

# Get the latest revision of the task definition
$LatestTaskDefinition = (aws ecs describe-task-definition --task-definition $TaskDefinitionName --region $AwsRegion | ConvertFrom-Json).taskDefinition.taskDefinitionArn

# Update the ECS Service
aws ecs update-service --service $ServiceName --task-definition $LatestTaskDefinition --force-new-deployment  --region $AwsRegion
```

This script updates the ECS service by referencing the latest task definition ARN and adding the `--force-new-deployment` flag, which enforces a rolling update of the ECS containers. This is another "Run a Script" Octopus step.

**Important Considerations and Best Practices:**

*   **IAM Permissions:** Ensure that the Octopus deployment user or role has the necessary AWS permissions to interact with ECR, ECS, and potentially other services you might be using. This involves creating a suitable IAM policy and attaching it to your deployment principal.
*   **Blue/Green Deployments:** For zero-downtime deployments, you would look at implementing a blue/green deployment strategy, which is more involved. This involves having two sets of environments and shifting traffic between them after the deployment is complete. This might involve the use of AWS CodeDeploy in conjunction with ECS and Octopus.
*   **Secrets Management:** Avoid hardcoding any secrets (passwords, tokens, etc.). Use Octopus’s built-in secrets management or a dedicated secrets manager like AWS Secrets Manager.
*   **Idempotency:** Ensure your deployment steps are idempotent, meaning running them multiple times won't cause unintended side effects. This is particularly important when working with cloud resources. The AWS CLI helps a lot in this regard, but you must double check that your script will behave as you expect.
*   **Error Handling:** Include comprehensive error handling and logging within your Octopus deployment scripts. If a script fails, you need to be able to pinpoint what went wrong and why. I typically configure Octopus to send me an email on failure so I’m immediately notified when something doesn’t go according to plan.

**Recommended Resources:**

*   **"Programming Amazon EC2"** by Pratik Patel and Eric B. Brown: This book provides a solid understanding of how to work with AWS infrastructure and services, including ECS. It's very detailed, but may not be up-to-date with the most recent services and tools.
*   **"Docker Deep Dive"** by Nigel Poulton: A very practical book, this provides insight into the nuances of containerization. It’s really helpful for understanding how the core technologies (like docker) work before attempting to use a container orchestration platform.
*   **The AWS CLI documentation**: The AWS CLI is how you will interact with your AWS infrastructure within your Octopus deployment process. Familiarity with it is key for automating your infrastructure within Octopus.
*   **The Octopus Deploy documentation**: You should understand Octopus’s deployment process and variable substitution model before trying to automate ECS deployments.

In closing, deploying to ECS using Octopus isn't a one-size-fits-all solution. It requires a clear understanding of both platforms and how they interact. By breaking the process into manageable steps, parameterizing your configurations, and being meticulous with error handling, you can achieve reliable and repeatable deployments. Remember the key is orchestration – Octopus is the conductor, ECS is the orchestra.
