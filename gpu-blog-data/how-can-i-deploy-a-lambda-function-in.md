---
title: "How can I deploy a Lambda function in AWS when its package size is too large?"
date: "2025-01-30"
id: "how-can-i-deploy-a-lambda-function-in"
---
Deployment of AWS Lambda functions exceeding the 50MB unzipped size limit necessitates a shift from the conventional deployment method involving direct package upload.  I've encountered this limitation frequently in my work developing large-scale data processing functions, particularly those incorporating extensive third-party libraries or substantial pre-trained machine learning models.  The solution involves leveraging AWS's S3 storage service as an intermediary.  This approach circumvents the size restriction by having Lambda retrieve the function's code from S3 at runtime, rather than embedding it within the function's deployment package.

**1. Clear Explanation of the S3-Based Deployment Method**

The core principle behind this method involves creating a deployment package that's sufficiently small to satisfy Lambda's upload restrictions. This package, instead of containing the entire function's code and dependencies, will simply include a minimal bootstrap script.  This script's sole purpose is to download the complete function package, which resides in an S3 bucket, and then execute the main application code. This effectively decouples the Lambda function's deployment size from its runtime size.

Crucially, access to the S3 bucket must be properly configured to ensure your Lambda function possesses the necessary permissions to download the code. This involves creating an IAM role for the Lambda function granting it read access to the specific S3 bucket and object.  Failure to correctly manage these permissions will result in a runtime error.   Furthermore,  consider the implications of increased cold start times, which are inherent to this approach, due to the added step of downloading the code before execution. Strategies for mitigating cold starts, such as provisioning concurrency, will likely be necessary for functions requiring low latency.

**2. Code Examples with Commentary**

Here are three code examples illustrating the process, progressively demonstrating greater complexity and robustness:


**Example 1:  Simple Bootstrap (Python)**

This example showcases the most basic implementation.  It's suitable for simple functions with minimal dependencies.  Note the reliance on `boto3`, the AWS SDK for Python.

```python
import boto3
import os
import subprocess

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket_name = 'my-lambda-bucket'
    object_key = 'my-function.zip'
    temp_file = '/tmp/my-function.zip'

    s3.download_file(bucket_name, object_key, temp_file)

    subprocess.check_call(['unzip', '-o', temp_file, '-d', '/tmp'])
    os.chdir('/tmp') # Navigate to unzipped directory

    # Execute your main application code from here, for instance:
    os.system('python3 main.py')
    # or python main.py if using python2
    # Replace 'main.py' with your main script name.

```

**Commentary:** This code directly downloads the zip archive, unpacks it, and then executes the main Python file. Error handling is minimal; production-ready code requires more robust exception handling and logging. The `subprocess` module is used for executing external commands.


**Example 2: Enhanced Bootstrap with Error Handling (Node.js)**

This example expands upon the first, providing basic error handling and improved file management in Node.js.

```javascript
const AWS = require('aws-sdk');
const fs = require('fs');
const unzip = require('unzip');
const { spawn } = require('child_process');

const s3 = new AWS.S3();

exports.handler = async (event, context) => {
  const bucketName = 'my-lambda-bucket';
  const objectKey = 'my-function.zip';
  const tempFilePath = '/tmp/my-function.zip';
  const extractPath = '/tmp';


  try {
    const data = await s3.getObject({ Bucket: bucketName, Key: objectKey }).promise();
    fs.writeFileSync(tempFilePath, data.Body);

    const unzipStream = fs.createReadStream(tempFilePath).pipe(unzip.Extract({ path: extractPath }));

    unzipStream.on('close', () => {
        const child = spawn('node', ['./main.js'], { cwd: extractPath}); // Execute your main script. Replace './main.js' accordingly
        child.stdout.on('data', (data) => console.log(`stdout: ${data}`));
        child.stderr.on('data', (data) => console.error(`stderr: ${data}`));
        child.on('close', (code) => {
            console.log(`child process exited with code ${code}`);
        });
    });

  } catch (error) {
    console.error('Error downloading or extracting code:', error);
    throw error;
  }
};
```

**Commentary:** This example utilizes Node.js's built-in `fs` module for file system operations and an external `unzip` package for archive handling.  The use of promises and `async/await` improves code readability and facilitates better error handling. The child process execution allows for better control and error handling over directly calling system commands as seen in Example 1.


**Example 3:  Layered Architecture with Separate Bootstrap and Application (Go)**

This example demonstrates a more sophisticated approach, separating the bootstrap logic from the core application code. This improves maintainability and organization, especially for larger projects.  It's presented in Go, highlighting the adaptability of this method across various programming languages.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/aws/aws-lambda-go/lambda"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

func handler(ctx context.Context) error {
	sess := session.Must(session.NewSession())
	svc := s3.New(sess)

	bucket := "my-lambda-bucket"
	key := "my-function.zip"
	tempFile := "/tmp/my-function.zip"

	// Download the code
	_, err := svc.GetObjectWithContext(ctx, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	}, &aws.WriteAtBuffer{})

	if err != nil {
		return fmt.Errorf("failed to download code: %w", err)
	}
	//Unzip and execute the code. (Simplified for brevity)

	cmd := exec.Command("unzip", "-o", tempFile, "-d", "/tmp")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if err != nil {
		return fmt.Errorf("failed to unzip code: %w", err)
	}
    // Execute your main application. Replace 'myapp' accordingly.
	cmd = exec.Command("./myapp")
	err = cmd.Run()
    if err != nil {
		return fmt.Errorf("failed to run app: %w", err)
	}
	return nil
}

func main() {
	lambda.Start(handler)
}
```

**Commentary:** This Go example leverages the AWS SDK for Go and utilizes a more structured approach to downloading, unpacking, and executing the application code. It demonstrates cleaner error handling through Go's built-in error handling mechanisms.

**3. Resource Recommendations**

For further exploration, consult the official AWS Lambda documentation, particularly the sections on IAM roles, S3 integration, and best practices for Lambda function development.  Study the documentation for your chosen programming language's AWS SDK.  Review materials on serverless application architecture and deployment strategies.  Familiarize yourself with concepts like cold starts and techniques to optimize their impact on performance.  Investigate advanced deployment methodologies such as using deployment tools or CI/CD pipelines for automated deployment.
