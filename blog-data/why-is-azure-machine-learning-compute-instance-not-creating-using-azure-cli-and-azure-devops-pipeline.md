---
title: "Why is Azure Machine Learning Compute Instance Not Creating using Azure CLI and Azure DevOps Pipeline?"
date: "2024-12-23"
id: "why-is-azure-machine-learning-compute-instance-not-creating-using-azure-cli-and-azure-devops-pipeline"
---

Alright, let's unpack this. I've seen my fair share of Azure Machine Learning compute instance creation failures, and it's rarely a straightforward single cause. Usually, it's a combination of factors that need careful examination, particularly when using Azure CLI within an Azure DevOps pipeline. The fact that it’s failing in this automated scenario points to issues beyond just a faulty local CLI command, which would be much easier to troubleshoot.

Let me recount a project where we were deploying a complex MLOps pipeline that heavily relied on compute instances for training. We were consistently running into a similar issue – compute instances refusing to materialize via the pipeline, and initially we suspected code issues, but the local CLI commands worked. It took a deep dive to reveal the real culprits, and I'll share some of that learning here.

The first thing to establish is that Azure DevOps pipelines run under service principals, or managed identities, not user accounts like when you're executing commands locally. This subtle but crucial difference means we need to examine the permissions being granted to the service principal in your pipeline. Your pipeline’s service principal needs the necessary roles granted at the resource group level, or better at the workspace level, and, critically, permissions to create compute instances specifically for your azure machine learning workspace. A common error is granting only ‘contributor’ role to the resource group, or even the machine learning workspace, and not realizing this doesn't grant explicit permission to create compute instances. You need the "machine learning contributor" role or a custom role that includes the necessary permissions. Often the service principal we were using had "contributor," but not the "machine learning contributor" role, which resulted in the very issue you are facing.

Secondly, the underlying Azure CLI command itself, while seemingly basic, requires meticulous parameter management, especially when being passed through DevOps. Here's a breakdown of the typical command and potential pitfalls:

```bash
az ml compute create --name my-compute-instance \
    --workspace-name my-ml-workspace \
    --resource-group my-resource-group \
    --type computeinstance \
    --size standard_ds3_v2 \
    --vnet-resource-group my-vnet-resourcegroup \
    --subnet my-subnet-name \
    --no-wait
```

The `--no-wait` parameter is essential for pipeline execution. If this is removed the pipeline would wait indefinitely for creation completion. Now, some problems I've witnessed stem from inconsistencies in these parameters. For instance, I saw that an error occurred once because the workspace name and the resource group name didn't match the environment the pipeline was deploying to in one of our environments. A typo, a missing variable substitution, or an improperly configured environment variable in the pipeline definition can easily derail the process. Remember that the pipeline parameters are evaluated at runtime, which makes debugging a bit trickier. Also verify the chosen compute size is available in the chosen region.

Now let's delve a bit into resource configuration, specifically virtual networks (vnets) and subnets. Specifying the `--vnet-resource-group` and `--subnet` parameters means your compute instance will be created within this network, which is crucial for secure network access and internal communication. However, the subnet must have sufficient capacity. If the subnet's address range is too small or has no available IP addresses, creation will fail. Also verify the service principal has sufficient permissions on the vnet resource group. Here's a simple example snippet of the azure cli command to check a subnet address space:

```bash
az network vnet subnet show \
   --name my-subnet-name \
   --vnet-name my-vnet-name \
   --resource-group my-vnet-resourcegroup \
   --query "{name:name, addressPrefix:addressPrefix, availableAddressPrefixes:availableAddressPrefixes}"
```

You can use the result of this command to ensure that your subnet is configured as intended. It’s essential to verify that your subnet has sufficient IP address space.

Furthermore, resource locks can also be an issue. If there's a resource lock placed on either the resource group, the machine learning workspace, or the vnet resource group, compute instance creation could be prevented. Resource locks are easily missed, especially when they've been put in place by another team or through a policy. We’ve had instances where a temporary lock, placed for a specific maintenance task, was unintentionally left active, which then stalled our compute creation. Review your resource group’s lock configurations and identify the source of the lock.

Another scenario I encountered was a pipeline that wasn’t utilizing a properly configured Azure DevOps self-hosted agent. If the agent doesn’t have the Azure cli installed, or doesn’t have the correct authentication configuration, compute instance creation will fail. You need to make sure the agent is properly installed and configured correctly.

Finally, let’s discuss concurrency limits. Azure has default quotas in place regarding the number of compute instances you can have within a region or within a subscription. In one of our projects we reached the limit on compute instances. In the DevOps pipeline we were only creating, but not removing, compute instances. We reached the Azure limits quickly. If you consistently run into creation failures, verify that you haven't hit these limits. Also remember to delete resources when they are not needed. Azure will helpfully show the limits when you run a failed job to create a compute resource. The following example will help check the available quota for your account:

```bash
az ml workspace show --resource-group my-resource-group --name my-ml-workspace --query "{location:location}" -o tsv |  xargs -I {} az vm list-usage --location {} --query "[?contains(name.value, 'standard_ds3_v2')]"
```

This snippet will extract your workspace region and then use it to query your quota in that region for the `standard_ds3_v2` size. You can modify this to check any other size. You should be able to use similar logic to check for other resource types.

To get a better grasp of these concepts I would recommend the official Azure documentation, which is a good place to start, but I would strongly recommend the book "Programming Microsoft Azure" by Haishi Bai et al. which offers an excellent introduction to Azure resource management, and the book "Microsoft Azure Administrator Certification Training: Study Guide with 2 Practice Tests" by Jason R. Alba for an in-depth understanding of service principals and resource permissions. Also, for more information on virtual network configurations, I suggest "Mastering Azure Network Services" by Saurabh Sharma. These resources, combined with meticulous attention to your environment configurations, should be enough to tackle your compute instance creation problems.

In summary, always begin with validating permissions for the pipeline’s service principal, check pipeline parameters against your target environments and verify network configurations and capacity. Don't forget to look for resource locks and double-check Azure quota limits. Careful examination of those areas has solved every compute instance creation issue I've ever faced.
