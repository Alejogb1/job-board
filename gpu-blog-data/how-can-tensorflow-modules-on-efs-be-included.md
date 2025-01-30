---
title: "How can TensorFlow modules on EFS be included in Lambda function dependencies on AWS?"
date: "2025-01-30"
id: "how-can-tensorflow-modules-on-efs-be-included"
---
TensorFlow, with its substantial size, presents a challenge when deployed in the ephemeral environment of AWS Lambda. Direct inclusion within the deployment package quickly exceeds the 50MB limit, and even the larger zipped package limit of 250MB is often inadequate for practical use cases. Leveraging Amazon Elastic File System (EFS) offers a solution by allowing Lambda functions to access shared file storage, effectively accommodating TensorFlow and its dependencies without package size constraints.

My experience in developing machine learning inference services on AWS has consistently led me to this approach. The tight coupling between Lambda's execution environment and its dependencies necessitates creative solutions. Attempting to brute-force package TensorFlow often introduces deployment failures and unacceptably long cold starts. A more refined approach involves mounting an EFS file system to the Lambda function, storing TensorFlow libraries within, and updating the Python path accordingly during Lambda initialization.

This process involves several key steps: first, creating an EFS file system, second, installing the required TensorFlow libraries on an EC2 instance (or similar compute resource) and transferring them to the EFS file system. Third, configuring the Lambda function to mount the EFS file system, and finally modifying the Lambda handler to correctly import the libraries from their new location.

The fundamental principle here is to treat EFS as a shared network drive that Lambda functions can access. When a Lambda function is invoked, the EFS mount is presented to the function's execution environment at a specified mount point. We leverage this mechanism to make the large TensorFlow library available during function execution, circumventing Lambda's limitations on deployment package size.

Here are three code examples to demonstrate this process:

**Example 1: Lambda Handler Modification**

This first example illustrates the changes needed within the Python Lambda handler. Crucially, we need to adjust the Python path to tell Python where to find the TensorFlow libraries, given they are not located in the standard path, but on the mounted EFS directory. Additionally, I've included error handling to verify the libraries were loaded correctly. This is critical during debugging.

```python
import os
import sys
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
  try:
    # Modify the python path. 'efs_mount' is the mount point we will use
    efs_path = "/mnt/efs"
    sys.path.append(efs_path)

    # Attempt to import Tensorflow from the EFS directory
    import tensorflow as tf

    logger.info(f"Tensorflow version {tf.__version__}") # Validate import
    
    # Placeholder operation using TensorFlow for demonstration
    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    result = tf.linalg.det(tensor)

    logger.info(f"Determinant of tensor {tensor} is {result.numpy()}")
    
    return {
      "statusCode": 200,
      "body": f"Successfully ran with TensorFlow: {tf.__version__}. Result:{result.numpy()}"
        }

  except Exception as e:
    logger.error(f"Error during execution: {e}")
    return {
      "statusCode": 500,
      "body": f"Error: {str(e)}"
      }
```

The core of this function is the modification of the `sys.path`. The `/mnt/efs` path must match the location to which the EFS file system is mounted within the Lambda function's execution environment. Then, the `import tensorflow as tf` statement will locate the TensorFlow library from the specified path. Without the path modification, a ModuleNotFoundError exception will be triggered. Logging is used to provide visibility during function execution, which is helpful in validating the import process.

**Example 2: Terraform Configuration of Lambda Function (Partial)**

This example utilizes Terraform, as it is a common infrastructure-as-code tool. This snippet is focused on how to configure an AWS Lambda resource to connect to an existing EFS access point using a lambda function resource. The full Terraform configuration would also include resources for the EFS file system and the mount target. The important parameters here are the `file_system_config` block, which specifies the access point ID and mount path for EFS.

```terraform
resource "aws_lambda_function" "example_lambda" {
  function_name = "example-tensorflow-function"
  handler = "lambda_function.lambda_handler"
  runtime = "python3.11"
  role    = aws_iam_role.lambda_exec.arn
  filename = "lambda_function.zip"
    # other lambda configurations including environment variables
  
    file_system_config {
      arn              = aws_efs_access_point.example_access_point.arn
      local_mount_path = "/mnt/efs" # same mount path used in lambda_function.py
    }
  
}
```

The `file_system_config` block is essential for mounting the EFS file system into Lambda's execution environment. The `arn` parameter specifies the EFS access point’s identifier, and `local_mount_path` defines where the file system is available within the function’s file system. This setting corresponds to the `/mnt/efs` directory used in Example 1. These two code snippets should be treated as complementary. I have often seen these components not sync up and lead to confusion.

**Example 3: Python Script for Pre-deployment (EFS Population)**

This final example showcases a simple Python script used for populating EFS with the required libraries. This is done before the Lambda function deployment. This script assumes that you are running it on an EC2 instance that has EFS mounted. This script involves a simple pip install, which installs TensorFlow and related libraries into a local directory and then copies those libraries onto the EFS file system.

```python
import os
import subprocess
import shutil

def populate_efs(efs_mount_path):
    temp_dir = "temp_package"

    # Create a temporary directory to install libraries
    os.makedirs(temp_dir, exist_ok=True)
    
    # Install libraries into temporary directory
    subprocess.check_call(["pip3", "install", "tensorflow", "--target", temp_dir])
    
    # copy to efs mount point
    shutil.copytree(temp_dir, efs_mount_path, dirs_exist_ok=True)

    # clean up temp directory
    shutil.rmtree(temp_dir)
    print("EFS populated with TensorFlow libraries.")
    
if __name__ == "__main__":
    efs_mount_point = "/mnt/efs" # Must be the same as above
    populate_efs(efs_mount_point)

```

This script demonstrates how to programmatically populate EFS with the necessary dependencies. It first creates a temporary directory, installs the `tensorflow` package using pip specifying a local target, then copies the contents of this local directory into EFS. The final step cleans the local directory. This operation should occur before the Lambda deployment, ensuring the required libraries are present when Lambda is invoked. The EFS mount path used here must mirror the mount path in the Terraform configuration and Lambda handler.

Implementing this approach allows Lambda to scale without the constraint of the deployment package size. As long as the file system is available, the Lambda function will find and use the libraries, resolving issues that come from packaging large libraries.

For those interested in further exploration, the following resources are valuable. AWS documentation on Lambda function configuration, specifically on EFS mounting, provides detailed specifications on setup and behavior. In-depth blog posts detailing specific challenges in serverless machine learning deployment on AWS provide best practices for this pattern. Finally, source code repositories that provide examples of Lambda, EFS, and Terraform configurations are great for hands on learning. Understanding the nuances of each component will ensure effective use of this approach.
