---
title: "How can TagUI RPA be run as a Lambda function?"
date: "2025-01-30"
id: "how-can-tagui-rpa-be-run-as-a"
---
The core challenge in deploying TagUI for Robotic Process Automation (RPA) as an AWS Lambda function stems from Lambda's ephemeral and stateless nature, juxtaposed with TagUI's reliance on a persistent file system and local browser instances. Successfully running TagUI in this environment necessitates addressing these fundamental incompatibilities through careful configuration and strategic use of supporting AWS services.

A Lambda function, by its design, operates within a container that is initialized only upon invocation and terminated shortly after. This means any local files written, browser instances spawned, or processes maintained do not persist between executions. TagUI, conversely, needs a working directory to store scripts, downloaded files, and configurations. Additionally, TagUI traditionally launches a local browser for UI interaction, which is problematic within Lambda's serverless environment. Therefore, our approach requires abstracting these needs through containerization and utilizing a headless browser.

My experience with this setup involved extensive experimentation. Early attempts failed due to the limitations mentioned above. Successfully deploying TagUI as a Lambda function required the following steps: First, encapsulating TagUI and its dependencies within a custom Docker image. Second, using a headless browser compatible with Lambda's execution environment. Third, adjusting TagUI's execution parameters to operate within the Lambda context. Finally, ensuring the necessary AWS permissions were assigned.

To illustrate, consider the common scenario of a web scraping task. We will execute a TagUI script that navigates to a website and extracts specific data. This process breaks down into the following code examples, demonstrating how we must adjust the standard TagUI approach to work under Lambda.

**Code Example 1: Dockerfile for TagUI Lambda Function**

```dockerfile
FROM public.ecr.aws/lambda/python:3.9

# Install TagUI dependencies
RUN yum install -y unzip wget \
  && wget https://github.com/kelaberetiv/TagUI/archive/v6.12.tar.gz \
  && tar -zxvf v6.12.tar.gz \
  && rm v6.12.tar.gz \
  && mv TagUI-6.12 /opt/tagui \
  && chmod +x /opt/tagui/tagui

# Install necessary system packages for headless Chrome
RUN yum install -y pango libXcomposite libXdamage libXext libXfixes libXi libXrandr \
libXrender fontconfig cairo-devel alsa-lib atk cups-libs -y \
  && wget https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm \
  && yum install -y google-chrome-stable_current_x86_64.rpm \
  && rm google-chrome-stable_current_x86_64.rpm

# Define TagUI working directory and user
RUN mkdir /mnt/tagui
WORKDIR /mnt/tagui
RUN groupadd -r tagui && useradd -r -g tagui tagui
USER tagui

# Copy Lambda handler and TagUI script
COPY lambda_function.py .
COPY example.tag .

# Set entry point for Lambda execution
CMD ["lambda_function.handler"]
```

This Dockerfile forms the foundation of our deployment. We start with the official AWS Lambda Python image, then install TagUI using a wget command, unpacking the archive to `/opt/tagui`. Critically, we then install the packages required by headless Chrome which is essential as the standard Chrome installation isn't designed for headless operation. We download and install Google Chrome, setting our working directory and adding a ‘tagui’ user. We copy our python handler function and the tagui script, setting up the entry point for Lambda. The use of a distinct ‘tagui’ user enhances security and prevents accidental modification of root-owned system files.

**Code Example 2: Python Lambda Handler `lambda_function.py`**

```python
import os
import subprocess
import json
import tempfile

def handler(event, context):
    try:
        # Create a temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
          os.chdir(temp_dir)
          
          # Execute the TagUI script
          tagui_path = "/opt/tagui/tagui"
          script_path = "example.tag"

          command = [tagui_path, script_path, '-headless']
          process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          stdout, stderr = process.communicate()

          # Handle errors or timeouts
          if process.returncode != 0:
              return {
              "statusCode": 500,
              "body": json.dumps({"error": stderr.decode('utf-8')})
              }

          # Return the captured output
          return {
              "statusCode": 200,
              "body": json.dumps({"output": stdout.decode('utf-8')})
          }
    except Exception as e:
         return {
             "statusCode": 500,
             "body": json.dumps({"error": str(e)})
         }
```

The Python handler (`lambda_function.py`) is the entry point for our Lambda execution. It creates a temporary directory, changes to it, and executes the TagUI script using `subprocess.Popen`. Critically, we invoke TagUI with the `-headless` option, enabling it to operate without a display server. We also capture the standard output and error streams, crucial for debugging and understanding execution results. The return value is a JSON-formatted response, standard for Lambda functions. Error handling is included at the `try-except` level. This provides a layer of robust handling for possible runtime errors.

**Code Example 3: Example TagUI Script `example.tag`**

```tagui
// Example TagUI Script
https://example.com
wait 5
snap page to output.png
```

This simplified TagUI script demonstrates basic web navigation and screenshot functionality. It opens `https://example.com`, waits for 5 seconds, and saves a snapshot of the page to ‘output.png’. While basic, it shows TagUI interacting with a web page. The headless browser, running within the Lambda container, manages this. The output image file, while created, won't persist beyond the execution. Therefore, further processing would be required to retrieve such outputs (e.g. saving to S3).

Successful deployment requires building the Docker image and uploading it to Amazon Elastic Container Registry (ECR). The Lambda function is then configured to use this image. It's also essential to configure the Lambda function with sufficient memory, execution timeout, and necessary IAM permissions. IAM permissions, specifically, need to grant the Lambda function access to write to storage if we want to persist outputs from the TagUI runs. This ensures that the necessary resources are available, and interactions with other AWS services are authorized.

Further improvements could include integrating the Lambda function with Amazon API Gateway for easier invocation, using AWS Systems Manager Parameter Store for managing configuration parameters, and integrating with cloud monitoring tools for tracking execution metrics and potential errors. Regarding debugging, cloud watch logs can help trace execution paths within the lambda and spot errors quickly.

For resources, I recommend consulting the official TagUI documentation, as well as AWS documentation on Lambda, Docker, and ECR. Tutorials are readily available on using Docker containers with AWS Lambda. Furthermore, research into headless browser best practices will be beneficial. Experimentation will be crucial for mastering the nuances of running TagUI in such a constrained environment.

Through this process, I have observed that while seemingly challenging, TagUI can operate successfully in a Lambda context. It requires a carefully considered design incorporating Docker for containerization, headless browsing for UI interaction, and specific Lambda configuration for managing environment limitations.
