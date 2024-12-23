---
title: "How can Terraform's `for_each` be used to extract resource IDs from a map of objects?"
date: "2024-12-23"
id: "how-can-terraforms-foreach-be-used-to-extract-resource-ids-from-a-map-of-objects"
---

Alright,  I remember vividly a project a few years back where we were managing a complex infrastructure involving dozens of virtual machines, each with unique configurations defined in a large, cumbersome json file. Moving this into a more manageable, code-driven terraform setup quickly became paramount, and, naturally, `for_each` became our go-to tool. So, the core problem is how to gracefully retrieve specific resource identifiers from a map of objects using `for_each`. It might sound trivial, but when you have dozens of resources with interdependent relationships, the structure of that map can become quite a headache if not handled precisely.

The challenge stems from the inherent nature of `for_each`. It iterates over *keys* of a map or the *elements* of a set, not the values directly within the map if those values are objects. The goal isn't just iteration; it's about using those iterations to accurately define resource relationships by referencing the specific identifiers we need. This frequently involves dealing with a map of object, where each object might hold properties relevant to a given resource. Let me clarify that using specific examples and some snippets that represent what I’ve seen in similar projects.

Essentially, what we are trying to do is use a map where the *keys* represent some form of identifier for each of our objects in the map. These objects themselves have various attributes, and one of the attributes is the resource identifier (e.g. an instance id, a database id etc) that we need to reference elsewhere in our Terraform configuration.

Let’s start with a basic example, which I'll expand on. Suppose we have a map representing database configurations:

```terraform
locals {
  database_configurations = {
    "db_primary" = {
      engine       = "postgres"
      version      = "14"
      instance_id = "i-0abcdef1234567890"
      storage_size = 100
    },
    "db_secondary" = {
      engine       = "mysql"
      version      = "8.0"
      instance_id = "i-0zyxwvu9876543210"
      storage_size = 50
    },
        "db_replica" = {
      engine       = "postgres"
      version      = "14"
      instance_id = "i-98765abcdef012345"
      storage_size = 100
    }
  }
}
```

Now, suppose we need to create some sort of access controls that depend on these database instance ids. We need a way to loop through this `database_configurations` map, extract the `instance_id` from each object, and pass these ids into our `aws_security_group` resource block. This is where `for_each` shows its real strength. Here is the code:

```terraform
resource "aws_security_group" "database_access" {
  for_each = local.database_configurations

  name        = "database_access_${each.key}"
  description = "Allow access to database instance ${each.key}"

  ingress {
    from_port   = 5432 # postgres default port
    to_port     = 5432
    protocol    = "tcp"
     cidr_blocks = ["10.0.0.0/16"]
      }
   ingress {
    from_port   = 3306 # mysql default port
    to_port     = 3306
    protocol    = "tcp"
     cidr_blocks = ["10.0.0.0/16"]
      }

   tags = {
      Name = "database_access_${each.key}"
    }
}
resource "aws_security_group_rule" "database_access_rule" {
  for_each = local.database_configurations
  type             = "ingress"
  from_port        = each.value.engine == "postgres" ? 5432 : 3306
  to_port          = each.value.engine == "postgres" ? 5432 : 3306
  protocol         = "tcp"
  security_group_id = aws_security_group.database_access[each.key].id
  cidr_blocks = ["10.0.0.0/16"]


}

resource "aws_instance" "example" {
  for_each = local.database_configurations

   ami = "ami-0c55b78937ad1c380" # ubuntu default ami in us-east-1
    instance_type = "t2.micro"
  subnet_id = "subnet-01234567890abcdef" # replace with your subnet
 vpc_security_group_ids = [aws_security_group.database_access[each.key].id]
   tags = {
    Name = each.key
  }
}

```

In this snippet, `for_each = local.database_configurations` iterates over each key-value pair in our `database_configurations` map. The `each.key` gives us access to "db_primary", "db_secondary", and "db_replica", and `each.value` gets us the *object* associated with that key; e.g. the entire object containing `engine`, `version`, `instance_id`, and `storage_size`. Then, in the `aws_security_group` resource, we can access attributes of the object like `each.value.instance_id` but, even more importantly in our case, we can use the `each.key` to uniquely reference each resource we created in our configuration. In the security group rule resource we used conditional logic based on `each.value.engine` to determine the port. We then reference the `aws_security_group` object using `aws_security_group.database_access[each.key].id`.

The key point here is that `each.value` gives access to the object and allows us to pull out the `instance_id` or any other attributes as needed; `each.key` allows us to correctly reference each unique resource created in our `for_each` loop.

Sometimes, our objects may not contain the resource identifier *directly*, but rather contain information that allows us to dynamically create that identifier via a computation. Consider this scenario:

```terraform
locals {
  application_configurations = {
    "app_backend" = {
      env   = "prod"
      region = "us-east-1"
    },
     "app_frontend" = {
      env   = "dev"
      region = "us-west-2"
    }
  }
}

resource "aws_iam_role" "app_roles" {
  for_each = local.application_configurations

    name = "${each.value.env}_${each.key}_role"
    assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Effect = "Allow"
        Sid = ""
      },
    ]
  })

}
resource "aws_iam_role_policy" "app_policy" {
   for_each = local.application_configurations
 role = aws_iam_role.app_roles[each.key].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:*"
        ]
        Effect   = "Allow"
        Resource = ["arn:aws:s3:::*"]
      },
    ]
  })
}


```
Here, instead of an explicit resource id, we have the `env` and `key` which allow us to generate the `name` attribute using `${each.value.env}_${each.key}_role`. Critically, and in line with what was discussed previously, we can still refer to the correct `aws_iam_role` resource in our `aws_iam_role_policy` resource by using `aws_iam_role.app_roles[each.key].id`. The identifier may have been constructed instead of being pre-existing in the object, but the same principles still apply. The power lies in the iteration logic coupled with the `each.key` and `each.value` variables to maintain accurate references.

In essence, using `for_each` effectively in situations like this relies on understanding that `each` provides both a key and the object itself, giving you flexibility in how you extract or generate resource identifiers. We’re leveraging the *key* not just as an arbitrary identifier but as a stable and uniquely identifying key to access the correct resources in our state.

For more in-depth treatment on the intricacies of Terraform’s `for_each` and resource management, I'd strongly recommend reviewing “Terraform Up & Running” by Yevgeniy Brikman. This book does an excellent job of clarifying these concepts and offers a more complete look at the various nuances of using these features in production environments. Similarly, the official Terraform documentation for `for_each` is an invaluable resource, always worth going back to and refreshing your knowledge on.

Remember that, as your maps become more complicated, careful planning on how to structure them is vital, and always keep in mind the need for stable identifiers, particularly when managing interdependent resources with `for_each`. It's a powerful tool, but precision in how you use it is paramount.
