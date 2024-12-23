---
title: "How can TagUI RPA be run as a Lambda function?"
date: "2024-12-23"
id: "how-can-tagui-rpa-be-run-as-a-lambda-function"
---

Let's get into this; executing TagUI RPA as a serverless lambda function presents a fascinating, albeit complex, challenge. Over my years of dabbling with robotic process automation, I’ve seen numerous attempts to bridge the gap between traditional, often locally executed, RPA and cloud-native architectures. It’s a transition that, while seemingly straightforward, throws up quite a few nuances.

The primary hurdle arises from the nature of TagUI itself. It's designed for local execution, heavily reliant on a persistent operating system, a local browser instance, and a file system for script storage and output. Lambda, on the other hand, is ephemeral, stateless, and operates in a highly constrained environment. We need to reconcile these differences to achieve our goal. So, let's break it down into practical steps and examine the core problems we need to address.

First, let’s tackle the issue of dependencies. TagUI, being built on node.js, relies on a myriad of npm packages and system-level binaries such as Google Chrome or PhantomJS. These dependencies need to be packaged within the Lambda deployment artifact. This isn’t as simple as zipping your local node_modules folder; Lambda needs dependencies to be architecture-specific. It means carefully building your TagUI environment within a Docker container that matches the lambda execution environment’s architecture (usually Amazon Linux 2). It's crucial to build this container in a way that includes not just your npm modules but also the correct chrome binaries, all properly linked. This can be a bit fiddly, and it’s an area where people often stumble.

Second, persistent storage becomes a challenge. Lambda functions have a temporary `/tmp` directory, but it's limited and not guaranteed to persist across invocations. TagUI scripts, by nature, often generate output (screenshots, logs) and may need to read external data files. We’ll need a mechanism to persist this output and manage input data. Options include S3 for storing input scripts and output logs, or even a database for more structured data handling.

Third, consider how we're going to invoke TagUI from within the Lambda function. TagUI needs to be called as a command-line process. This means using Node.js's child_process module to execute the TagUI binary with the specific script. We'll need to ensure proper error handling and return codes are passed from the TagUI process. I’ve seen plenty of situations where uncaught errors in the child process lead to silent failures within lambda, making debugging incredibly difficult.

Let’s look at some example code snippets to demonstrate these ideas:

**Example 1: Dockerfile for Packaging TagUI for Lambda**

```dockerfile
FROM amazonlinux:2

# Install required packages
RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    make \
    wget \
    unzip \
    tar \
    git

# Install Node.js and npm (adjust version as needed)
RUN curl -sL https://rpm.nodesource.com/setup_18.x | bash -
RUN yum install -y nodejs

# Install Chromium (headless)
RUN yum install -y fontconfig
RUN wget https://github.com/SpartanGeek/chromium/releases/download/v117.0.5938.149/chromium-headless-linux-v117.0.5938.149.zip
RUN unzip chromium-headless-linux-v117.0.5938.149.zip && mv chrome-linux /opt/chrome

# Set executable
RUN chmod +x /opt/chrome/chrome

# Install TagUI globally (adjust version as needed)
RUN npm install -g tagui@6.11

# Create workspace
RUN mkdir /app
WORKDIR /app

# Copy lambda handler and scripts (if any)
COPY lambda_handler.js ./
COPY tagui_scripts ./tagui_scripts

# Set environment variables
ENV CHROME_BIN=/opt/chrome/chrome
ENV HEADLESS=true

# Entry point for the container (you can overwrite this in lambda configuration)
CMD [ "node", "lambda_handler.js" ]

```

This Dockerfile sets up a suitable environment, installs required tools, and copies over the essential lambda handler. Notice the inclusion of chromium, carefully setting the `CHROME_BIN` environment variable, and installing TagUI globally. This Dockerfile is a foundational step – the built image will serve as our deployment package.

**Example 2: Lambda Handler (Node.js)**

```javascript
const { exec } = require('child_process');
const fs = require('fs').promises;
const AWS = require('aws-sdk');

const s3 = new AWS.S3();

exports.handler = async (event) => {
  try {
    const scriptName = event.scriptName || 'test.tag';
    const bucketName = process.env.INPUT_BUCKET;

    if (!bucketName) {
      throw new Error("Input bucket not configured.");
    }
    // download the script
    const s3params = {
        Bucket: bucketName,
        Key: `tagui_scripts/${scriptName}`
    };

    const script = await s3.getObject(s3params).promise();
    await fs.writeFile(`/tmp/${scriptName}`, script.Body);

    const taguiCommand = `/usr/local/bin/tagui /tmp/${scriptName}  `;
    const taguiPromise = new Promise((resolve, reject) => {
      exec(taguiCommand, (error, stdout, stderr) => {
        if (error) {
          console.error(`Error executing TagUI: ${error.message}`);
          console.error(`stderr: ${stderr}`);
          reject(new Error(`TagUI Execution Error: ${error.message}`));
          return;
        }
        resolve({ stdout, stderr });
      });
    });

    const results = await taguiPromise;
    console.log('TagUI Output:', results.stdout);
    console.error('TagUI Errors:', results.stderr);

    // upload the result log
    const loguploadparams = {
      Bucket: process.env.OUTPUT_BUCKET,
      Key: `results/${scriptName.replace('.tag', '.log')}`,
      Body: results.stdout + "\n" + results.stderr
    };

    await s3.upload(loguploadparams).promise();

    return {
      statusCode: 200,
      body: JSON.stringify({ message: 'TagUI script executed successfully', logkey : `results/${scriptName.replace('.tag', '.log')}` }),
    };
  } catch (error) {
    console.error('Error processing request:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ message: 'Error executing TagUI', error: error.message }),
    };
  }
};
```
This is a rather simplified handler but outlines the core functionality. It retrieves a TagUI script from S3, writes it to the `/tmp` directory, executes it using `child_process`, captures outputs and uploads logs to s3. We use environment variables to configure the location of the input scripts and output logs.

**Example 3: Example TagUI script (test.tag) for reference:**

```tagui
//simple tagui script for testing
init chrome headless
  url https://www.google.com
  type searchbox Hello World
  click Google Search
  wait 3
  snap page results.png
  dump results.txt
end
```

This demonstrates a simple tagui script that loads google, searches for "Hello world" takes a screenshot and dumps the result page.
This script can be uploaded to s3 and referenced via the `scriptName` when invoking the lambda.

For deeper understanding, I recommend studying *Operating System Concepts* by Silberschatz et al., particularly sections on process management and virtualization, which are pertinent when you are wrapping a system-level automation tool like tagui in an environment such as lambda. Also, *AWS Lambda in Action* by Danilo Poccia and Matthew Wilson is incredibly useful for grasping the nuances of serverless function development, including packaging and deployment.

Integrating TagUI with Lambda is not a quick fix. It requires a solid understanding of both systems and careful management of dependencies, storage, and process execution. But it's entirely achievable and, in many scenarios, can unlock significant advantages in scalability and maintainability.
