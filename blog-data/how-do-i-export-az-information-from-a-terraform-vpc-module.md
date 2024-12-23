---
title: "How do I export AZ information from a Terraform VPC module?"
date: "2024-12-23"
id: "how-do-i-export-az-information-from-a-terraform-vpc-module"
---

Okay, let's talk about extracting availability zone (AZ) information from a Terraform vpc module. It's a common requirement, and I've bumped into this more times than i'd care to recall, usually when cascading modules or when another team needs to build resources dependent on a specific subnet. It’s less straightforward than a simple output value, mostly because AZs can change and hardcoding them is a recipe for pain.

The heart of the matter lies in how Terraform handles dynamic values. You can't just reach inside a VPC module and grab them directly because those values are often determined at apply time, and outputs only become available after the module has been successfully created. This is where thinking about indirect outputs and data sources come in. I've seen this trip up even experienced folks.

When building a highly available application across AZs, dynamically figuring out which subnets exist in each AZ becomes critical. What you *don't* want is to have your infrastructure depend on specific AZ names, because these can change between AWS accounts or regions, and even within an account at times. Instead, we need a way to programmatically get the available AZs associated with the VPC created by the module.

Let’s walk through a practical approach using a few code examples. The first, and probably the most basic, way is to use a `data` block to query all available AZs in a given region, then use the `cidrsubnet` function on each subnet within our vpc module, and make an output. The second approach is a bit more advanced and will involve utilizing `for_each` loops with `locals` to transform the data in a way that's more useful.

Let's get started with our first approach. Say we have a simplified vpc module as follows:

```terraform
# vpc_module/main.tf

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public_a" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
  availability_zone = "us-west-2a"
  tags = {
    Name = "public-subnet-a"
  }
}

resource "aws_subnet" "public_b" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.2.0/24"
  availability_zone = "us-west-2b"
  tags = {
    Name = "public-subnet-b"
  }
}
```

Now, in the module consuming it, we can use the `data` block and some creative looping:

```terraform
# main.tf

module "vpc" {
  source = "./vpc_module"
}

data "aws_availability_zones" "available" {
  state = "available"
}

output "vpc_az_info" {
  value = [
    for az in data.aws_availability_zones.available.names: {
      name = az
      subnet_ids = [ for subnet in module.vpc.aws_subnet[*].id:
        subnet if substr(subnet.availability_zone, length(subnet.availability_zone)-1, 1) == substr(az, length(az)-1, 1)
      ]
    }
  ]
}

```

Here, the key is `data.aws_availability_zones.available`, which provides us with a list of available AZs in the current region. Then, we loop through them, looking for subnets created in the `vpc_module` with matching availability zones. Note we use `substr` to just grab the last character of the availability zone. I've often found this pattern to be useful when needing a quick mapping. This returns a list of objects each including a name and list of subnet ids.

However, this approach has a limitation, it requires us to create subnets with specific availability zone parameters. Sometimes, you might want subnets to be created based on a flexible number of AZs. This requires a more dynamic approach using `for_each` and `locals`. Let’s see a different approach:

First, the vpc module needs to have a way of accepting an az list:

```terraform
# vpc_module/main.tf

variable "azs" {
  type = list(string)
  description = "list of availability zones to use"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public" {
  for_each = toset(var.azs)
  vpc_id     = aws_vpc.main.id
  cidr_block = cidrsubnet(aws_vpc.main.cidr_block, 8, index(var.azs, each.value) )
  availability_zone = each.value
  tags = {
    Name = "public-subnet-${each.key}"
  }
}

output "subnet_ids" {
    value = aws_subnet.public[*].id
}
```

Here, we use `for_each` to create subnets based on input `azs`. Next, in our consumer module, we transform the data to provide a more useful output:

```terraform
# main.tf

module "vpc" {
    source = "./vpc_module"
    azs = data.aws_availability_zones.available.names
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
    az_map = { for az in data.aws_availability_zones.available.names : az => {
        subnets = [ for id in module.vpc.subnet_ids : id if substr(id.availability_zone, length(id.availability_zone)-1, 1) == substr(az, length(az)-1,1) ]
        }
    }
}

output "vpc_az_info" {
  value = local.az_map
}
```
Here, `local.az_map` uses a `for` loop to create a map using the az names as keys. It then filters the subnet ids list by checking the last character of their respective availability zones.

Finally, let’s try a third approach, which uses the `tolist()` function on subnets to be more explicit about our data transformations. This might come in handy if you are handling complex lists. We are going to modify the output block in the first code example to use `tolist()`:

```terraform
# main.tf

module "vpc" {
  source = "./vpc_module"
}

data "aws_availability_zones" "available" {
  state = "available"
}

output "vpc_az_info" {
  value = [
    for az in data.aws_availability_zones.available.names: {
      name = az
      subnet_ids = tolist([ for subnet in module.vpc.aws_subnet[*].id:
        subnet if substr(subnet.availability_zone, length(subnet.availability_zone)-1, 1) == substr(az, length(az)-1, 1)
      ])
    }
  ]
}
```

The only change here is the addition of `tolist()`, but it’s a useful change for clarity, especially when dealing with potentially null or empty lists, ensuring you get an actual list back, even if it's empty.

Which approach is better? It depends on the complexity of your situation. If you're dealing with explicitly defined subnets with fixed availability zones, the first or third method might suffice. However, for dynamic configurations where the number of AZs or their names might vary, the second approach with the `for_each` loop and local transformations is more flexible.

Regarding further learning, I’d recommend diving into the official Terraform documentation, particularly the sections on `data` sources, `for_each` loops, and `local` variables. Additionally, the book "Terraform: Up and Running" by Yevgeniy Brikman is an excellent resource for understanding more complex Terraform scenarios. For a more in-depth theoretical understanding of infrastructure-as-code concepts, consider studying papers on declarative programming and distributed systems, these principles are fundamentally what drive tools like terraform. It might seem a bit academic, but will drastically improve your understanding and allow you to apply the lessons more broadly. These resources, combined with practice, will put you in a position to handle these issues with confidence.
