---
title: "How can AZ information be passed from a Terraform VPC module?"
date: "2025-01-30"
id: "how-can-az-information-be-passed-from-a"
---
The selection and propagation of Availability Zone (AZ) information from a Terraform VPC module often presents a challenge in infrastructure-as-code deployments, particularly when striving for flexibility and resilience. A common pitfall is hardcoding AZ selections, leading to brittle setups and making it difficult to adapt to regional capacity changes or specific placement requirements. I've encountered numerous instances where rigid AZ assignments in the VPC module cascade into downstream resources, making refactoring painful. To address this, a robust solution typically involves passing dynamic AZ information as outputs from the VPC module and consuming them within the resources that require them.

The fundamental principle is to allow the VPC module to determine *available* AZs within the target region and then to expose a *selection* of those AZs as outputs. Downstream modules and resources should subsequently rely on these outputs rather than making assumptions about AZ names or numbers. This approach allows for greater portability of your infrastructure across regions and for more adaptable deployments within a given region.

Here is a breakdown of the process:

1.  **Discovery within the VPC Module:** The VPC module first needs a mechanism to discover available AZs. The Terraform AWS provider offers the `aws_availability_zones` data source. This data source queries the AWS API and returns a list of available AZs for the configured provider region.

2.  **AZ Selection Logic:** Once the available AZs are known, the VPC module should implement logic to select the desired AZs. This selection can be as simple as choosing the first 'n' available zones or could involve more complex logic based on tags or other attributes returned by the data source. I often prefer picking the first two, which generally provides a good balance of resilience and keeps the management surface relatively simple.

3.  **Outputting AZ Information:** The selected AZ names are then exposed as outputs of the VPC module. These outputs will be string arrays. It’s vital that output names are descriptive, such as `selected_az_names` or `private_subnet_azs` for clarity.

4.  **Consumption in Downstream Modules:** Downstream modules, such as those provisioning EC2 instances, RDS databases, or EKS clusters, consume the AZ outputs using Terraform input variables. These variables ensure that resources are distributed across the selected AZs from the VPC module.

5. **Flexibility Through Indexing:** Leveraging Terraform’s indexing features, particularly with the `count` meta-argument, is very useful. You can iterate through the provided list of AZs to create multiple subnets or instances, each within a distinct zone. This approach inherently promotes HA and simplifies configuration management.

Let me illustrate with code examples.

**Example 1: VPC Module - AZ Discovery and Selection**

```terraform
data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  selected_az_names = slice(data.aws_availability_zones.available.names, 0, 2)  # Select first two AZs
}

output "selected_az_names" {
  value = local.selected_az_names
  description = "List of selected availability zone names."
}
```

In this example, the `aws_availability_zones` data source retrieves all available AZs in the current region. The `slice` function selects the first two elements of the resulting names list. The chosen AZs' names are stored in the local variable `selected_az_names` and exposed as an output. This ensures that the module is not restricted to specific AZ names, and the selection is performed programmatically. It also has the advantage of being region agnostic.

**Example 2: Subnet Module - Consuming AZ Information**

```terraform
variable "az_names" {
  type = list(string)
  description = "List of availability zone names."
}

resource "aws_subnet" "private_subnet" {
  count = length(var.az_names)
  availability_zone = var.az_names[count.index]
  cidr_block = cidrsubnet("10.0.0.0/16", 8, count.index) #Example
  vpc_id = var.vpc_id  # VPC ID variable passed into the subnet module
  tags = {
      Name = "Private Subnet ${count.index + 1}"
    }
}

output "private_subnet_ids" {
  value = aws_subnet.private_subnet[*].id
  description = "List of private subnet IDs"
}

```
Here, the module expects a list of AZ names as an input through the `az_names` variable. The `aws_subnet` resource utilizes the `count` meta-argument to create multiple private subnets. The subnets' location is driven by the `az_names` variable at a given index within the loop. This guarantees that one subnet is created per selected AZ, and each subnet is properly tagged for ease of identification. A `cidrsubnet` function shows an example of how to subdivide the initial VPC CIDR. This output can then be used by modules that require the subnets ids.

**Example 3: EC2 Instance Module - Leveraging AZ Indexing**

```terraform
variable "az_names" {
  type = list(string)
  description = "List of availability zone names."
}

resource "aws_instance" "example_instance" {
    count = length(var.az_names)
    ami = "ami-xxxxxxxxxxxxx"  # Replace with valid AMI
    instance_type = "t3.micro"
    subnet_id = var.subnet_ids[count.index]
    availability_zone = var.az_names[count.index] # Use the same list as subnet module
    tags = {
      Name = "Example Instance ${count.index + 1}"
    }
}


```

In this example, similar to the subnet module, the `az_names` variable is used to place instances across each selected AZ. It’s expected that the `subnet_ids` input will be provided from the output of the subnet module. The `count` loop creates one instance in each AZ. It is essential to align the used AZ list with the location of the subnet being used. This ensures that the instance has a subnet available in the same AZ and that HA is achieved.

The use of `count.index` is very important here and in the previous example as it ensures each item in the list is accessed in order and is consistent across resources using the same list. This allows you to create multiple resources that are correlated to the selected AZs.

**Resource Recommendations:**

For a deeper dive, I recommend exploring the official Terraform documentation for the following:

*   **`aws_availability_zones` data source:** This provides detailed information on usage and attributes.
*   **`count` meta-argument:** This is essential for creating multiple resources based on a list of values.
*   **`slice` function:** A very powerful function for list manipulation and selection.
*   **General Terraform documentation on variables and outputs**: Helps in structuring modules effectively.
*   **AWS documentation on Regions and Availability Zones**: This gives the best conceptual grounding of what you're actually building.

In conclusion, handling AZ information from a VPC module dynamically promotes robust and adaptable infrastructure. By using data sources, appropriate selection logic, and output mechanisms, you can avoid hardcoded configurations, leading to greater portability and resilience in your cloud deployments. Remember that the `count` argument paired with a list of AZ names will always produce an HA system, so make good use of it. Leveraging these techniques has proven very valuable in ensuring the durability of systems I have designed.
