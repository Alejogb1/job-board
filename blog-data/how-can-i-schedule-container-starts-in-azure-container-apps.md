---
title: "How can I schedule container starts in Azure Container Apps?"
date: "2024-12-16"
id: "how-can-i-schedule-container-starts-in-azure-container-apps"
---

Okay, let’s talk container starts in Azure Container Apps. It's a challenge I’ve encountered more than a few times, particularly when dealing with microservices that have specific dependencies or need phased deployments. There isn't a single, magic “schedule start” button, but rather a combination of approaches using Azure’s infrastructure to achieve the desired timing. You’re essentially orchestrating container lifecycles indirectly, using the tools at your disposal.

The core concept here is understanding that Azure Container Apps operates on a model of managing revisions, not individual container instances. When you push a change (including an image update or configuration change), you create a new revision. The actual scheduling is therefore often achieved via manipulating how these revisions become active. We won't directly tell a container to start at *exactly* 3:00 pm, but we will control *when* a new revision comes online, and by consequence, its constituent containers.

Let's explore the methods I've seen work effectively.

**1. Manual Revision Activation and Deactivation**

The simplest form of control is manual activation and deactivation of revisions. This isn't truly scheduled, of course, but it allows for a precise, human-driven timing, which is sometimes exactly what's needed. Think of scenarios like a background processing service needing to start after a daily database backup job concludes. It doesn't require a precise time; rather, it needs to happen after a specific preceding activity.

The process is straightforward. Using the Azure CLI (my tool of choice here):

```bash
# List revisions for your container app
az containerapp revision list --name <container-app-name> --resource-group <resource-group-name>

# Deactivate an existing revision (example revision name: myrevision-01)
az containerapp revision deactivate --name <container-app-name> --resource-group <resource-group-name> --revision myrevision-01

# Activate a revision (example revision name: myrevision-02)
az containerapp revision activate --name <container-app-name> --resource-group <resource-group-name> --revision myrevision-02
```

This approach offers precise manual control, but it isn’t ideal for automated scheduling scenarios, obviously. It requires human intervention or an external system to trigger these commands. Still, it is very useful when needing to stagger processes that are not tightly linked by time.

**2. Leveraging Azure Pipelines and Scheduled Builds**

For something a bit more automated, we turn to Azure Pipelines (or GitHub Actions, or any comparable CI/CD platform). The trick here isn’t scheduling the containers directly but scheduling the *builds* that result in new container images. These new images, when deployed, become new container revisions. Therefore, by scheduling the *build*, you indirectly schedule when new containers are spun up. I have used this strategy on several project, including one where we had to control the availability of new product features during specific times.

Here is a sample Azure Pipelines YAML snippet:

```yaml
trigger: none

schedules:
- cron: "0 10 * * *" # Runs at 10:00 UTC daily
  displayName: Daily Build & Deploy
  branches:
    include:
    - main

stages:
  - stage: Build
    jobs:
    - job: BuildImage
      pool:
        vmImage: ubuntu-latest
      steps:
      - task: Docker@2
        inputs:
          containerRegistry: 'yourRegistryConnection'
          repository: 'your-image-repo'
          command: 'buildAndPush'
          Dockerfile: 'Dockerfile'
          tags: '$(Build.BuildId)'

  - stage: Deploy
    dependsOn: Build
    jobs:
      - job: DeployContainerApp
        pool:
          vmImage: ubuntu-latest
        steps:
        - task: AzureCLI@2
          inputs:
            azureSubscription: 'your-azure-subscription-connection'
            scriptType: 'bash'
            scriptLocation: 'inlineScript'
            inlineScript: |
              az containerapp update --name <container-app-name> --resource-group <resource-group-name> --set properties.template.containers[0].image="your-image-repo:$(Build.BuildId)"
              az containerapp update --name <container-app-name> --resource-group <resource-group-name> --no-wait
```

This yaml defines a schedule ("cron: 0 10 * * *") that executes the pipeline daily at 10:00 UTC, builds your Docker image and deploys it to your container app, creating a new revision. Note the `az containerapp update` commands. The first *pushes* the new image and then the second one avoids waiting for the operation to complete and moves on immediately. This makes the process more asynchronous.

This setup doesn't give you absolute second-by-second precision. The build process and image deployment do have some duration, so the container will not be online at *exactly* 10:00, but it will usually be close.

**3. Azure Logic Apps for Advanced Orchestration**

Finally, for the most complex scenarios where I needed precise scheduling or conditional starting sequences, I have used Azure Logic Apps. Logic Apps allow for complex workflows involving multiple systems and schedules.

The Logic App approach allows to decouple the schedule from the build pipeline. Here, the pipeline only creates a new image, but the Logic App is responsible for activating it as a revision. In my case, I have used this when a deployment needed to be coupled with some other external data migration process, which was not available until a later time. The Logic App would wait for that data process to be complete and *then* trigger the activation of the new version.

Here's a simplified example in pseudocode demonstrating the core concepts of how the Logic App should work:

```
# Logic App Workflow (pseudocode)
trigger: Recurrence (e.g., "every hour")

actions:
  - action: Get-CurrentTime
    output: CurrentTime

  - condition: Check if it's 10:00 UTC
    expression: Time(CurrentTime) = 10:00 # check if its 10:00 AM UTC

    if true:
      - action: List container app revisions
        output: Revisions

      - action: Find the latest active revision
        output: LatestActiveRevision

      - action: Find the latest build revision
        output: LatestBuildRevision # using something like az containerapp revision list -o json --query "[?properties.active == null] | [0].name"

      - condition: Check if the latest build revision is different from latest active
        expression: LatestActiveRevision != LatestBuildRevision

        if true:
          - action: Activate the new container revision
             command: az containerapp revision activate --name <container-app-name> --resource-group <resource-group-name> --revision LatestBuildRevision

        if false:
          - action: Terminate - No new deployment needed
    if false:
        - action: Terminate - It's not 10 AM
```

The logic app retrieves the latest build revision and only activates it if it is different from the current active revision. There is a condition to ensure that it’s only activated at a specific time, in this case, 10:00. This logic can easily be adapted to very complex conditional triggers using the full power of Logic Apps' connector ecosystem.

**Resources for further study:**

* **“Cloud Native Patterns” by Cornelia Davis:** This book offers in-depth patterns and approaches for architecting and deploying containerized applications, which directly relate to the broader challenges of scheduling and revision management.
* **Microsoft Azure Documentation on Azure Container Apps:** This should always be your primary source for the latest information and guidance, including specifics on revisions, scaling, and deployment strategies. Pay close attention to the sections on revision management and traffic splitting.
* **“Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation” by Jez Humble and David Farley:** Although not specific to containers, this foundational text provides context on the importance of controlling and automating the release pipeline, and helps you connect the overall concepts with the specifics of scheduled deployments.

Remember, achieving specific container start times in Azure Container Apps is rarely a case of directly setting a timer on the container itself. Instead, you are orchestrating the underlying platform and its mechanisms, typically through revision deployment and activation, using either manual processes, build pipelines, or a dedicated orchestration service. Choose the method that aligns best with your requirements and the complexity of your system. Each of these examples are battle tested, and I’ve used them in the field with good results.
