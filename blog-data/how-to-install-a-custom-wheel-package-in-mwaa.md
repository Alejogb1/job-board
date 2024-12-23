---
title: "How to install a custom wheel package in MWAA?"
date: "2024-12-23"
id: "how-to-install-a-custom-wheel-package-in-mwaa"
---

Let's dive into this; I remember tackling a similar challenge back in my days optimizing data pipelines at 'Aetheria Dynamics'. Getting custom wheel packages deployed reliably in managed environments like Amazon Managed Workflows for Apache Airflow (MWAA) isn't always straightforward, but it's definitely manageable with the right approach. It's less about magic and more about understanding the underlying mechanics.

The crux of the matter is that MWAA environments, by their nature, are somewhat isolated. They don’t have direct access to the internet by default to just grab packages from PyPI. Therefore, you need to pre-package your custom wheels and then make them available during environment creation or updates. Forget trying to `pip install` directly in your DAG definitions – that’s not the way this system is structured. Here's how I’ve consistently handled it, and you should probably consider these approaches as well.

Firstly, you need to create your wheel package. Assuming you have your python project already structured with a setup.py or pyproject.toml (I strongly prefer pyproject.toml these days, but that’s another discussion), the command is usually something along the lines of:

```bash
python -m build --wheel
```

This will generate a `.whl` file typically inside the `dist` folder. For this example, let's suppose we have a custom package named `my_custom_lib` that contains functionality we need in our DAGs. The built wheel will then be something like `my_custom_lib-1.0.0-py3-none-any.whl`. This is our deliverable.

Now, here’s where we handle the MWAA integration. You have, broadly, two primary routes for deployment: S3 and Requirements.txt. I've found S3 to be more robust when handling complex dependency trees or when a versioned approach is required. Requirements.txt, on the other hand, works well for smaller, relatively static sets of packages.

Let's start with the S3 method, which I generally prefer:

**Method 1: Using an S3 Bucket**

The process here goes something like this: you upload your generated wheel file(s) to an S3 bucket which your MWAA environment has access to. During MWAA environment creation or update (using the AWS console, CLI, or infrastructure-as-code tools like CloudFormation or Terraform), you configure your environment to look in that S3 bucket for custom packages.

Here's what that typically looks like in Terraform configuration (since that’s my primary IAC tool):

```terraform
resource "aws_mwaa_environment" "example" {
  # ... other configurations ...
  source_bucket_arn = aws_s3_bucket.mwaa_bucket.arn
  requirements_s3_path = "requirements.txt"
  plugins_s3_object_version  = "your_optional_plugins_version"
  requirements_s3_object_version = "your_optional_requirements_version"
  
  dag_s3_path = "dags"

  # these will be read from the `requirements.txt` in your s3 bucket.
  
  environment_class = "mw1.small"
  
  logging_configuration {
    dag_processing_logs {
        enabled = true
        log_level = "INFO"
    }
    scheduler_logs {
        enabled = true
        log_level = "INFO"
    }
    task_logs {
        enabled = true
        log_level = "INFO"
    }
    webserver_logs {
        enabled = true
        log_level = "INFO"
    }
    worker_logs {
        enabled = true
        log_level = "INFO"
    }
  }
  
  
  
}


resource "aws_s3_bucket" "mwaa_bucket" {
  bucket = "your-mwaa-s3-bucket-name"
  acl    = "private"
}


resource "aws_s3_bucket_object" "custom_wheel" {
  bucket = aws_s3_bucket.mwaa_bucket.id
  key    = "my_custom_lib-1.0.0-py3-none-any.whl"
  source = "./dist/my_custom_lib-1.0.0-py3-none-any.whl" # path to your local wheel file
}

resource "aws_s3_bucket_object" "requirements_file" {
  bucket = aws_s3_bucket.mwaa_bucket.id
  key    = "requirements.txt"
  source = "./requirements.txt" # path to your local requirements.txt file
}
```

Important details: you provide an `s3_source_bucket_arn` which is the bucket your wheels will live in. You also specify a requirements file. For instance, that `requirements.txt` file should include a line similar to:

```
my_custom_lib==1.0.0
```

When creating or updating your MWAA environment, MWAA will look in this bucket, find your `requirements.txt`, and then fetch the specified wheels (as long as the wheel itself is in the bucket too). The `requirements_s3_object_version` allows for updating requirements if desired.

**Method 2: Using a Requirements.txt File Directly in S3**

The second route employs just the requirements file and uses it to explicitly reference wheel files in the same S3 bucket. This approach can be cleaner if you’re working with just custom wheels and don’t have a large number of additional standard PyPI packages.

In this method, instead of just `my_custom_lib==1.0.0`, your `requirements.txt` will have a path to the wheel file within s3:

```
s3://your-mwaa-s3-bucket-name/my_custom_lib-1.0.0-py3-none-any.whl
```

The terraform for creating the s3 bucket and object for the requirements file remains the same as above, only the contents of your local requirements file change. The corresponding `aws_mwaa_environment` resource configuration, is identical to the above approach in Method 1. MWAA fetches `requirements.txt`, and the paths within point directly to the .whl files.

**A Note on Dependencies**

A critical point to consider is handling dependencies of your custom wheel. If `my_custom_lib` depends on `requests` (or some other package), you have a few options:

1.  **Include Dependencies in Your Wheel:** You can potentially create a "fat" wheel that includes the dependencies. This is achievable via some more advanced configuration in `setup.py` or `pyproject.toml`, but I wouldn't recommend this approach for a production setup, it gets messy quickly.
2.  **Explicitly List Dependencies:** You can add all needed dependencies to your `requirements.txt` file alongside the custom wheel specification.

Let's exemplify using explicit dependency specification:

```python
# requirements.txt (using Method 2 for example)
s3://your-mwaa-s3-bucket-name/my_custom_lib-1.0.0-py3-none-any.whl
requests==2.28.1
pandas==2.0.0
```

So, you'd also need to upload your custom wheel to your s3 bucket. This way, all requirements will be installed when MWAA initializes the environment.

**Code Example in a DAG**

Regardless of the method chosen, once deployed, using your custom library in your DAG is as straightforward as importing and using any other Python package.

```python
# my_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from my_custom_lib.my_module import my_custom_function # Example import. Assume `my_module.py` exists within your library
import os

def my_task_function():
    print("Starting my task")
    result = my_custom_function(10, 5) # Using function from `my_custom_lib`
    print(f"Custom function result: {result}")

with DAG(
    dag_id="my_custom_lib_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["example"],
) as dag:
    t1 = PythonOperator(
        task_id="use_my_custom_library",
        python_callable=my_task_function,
    )
```

In conclusion, the key to successfully incorporating custom wheel packages into your MWAA environment is to understand that you need to make those packages accessible within MWAA's isolated ecosystem. Using S3 and the `requirements.txt` as detailed above provides a reliable and flexible approach. For more advanced scenarios, it is beneficial to dive into the documentation of specific IaC tools and familiarize yourself with aspects like the `plugins.zip` and `requirements.txt`. Regarding resources, I suggest checking out the *Amazon Managed Workflows for Apache Airflow (MWAA) Developer Guide* and *Effective Python* by Brett Slatkin for general good Python development practices and configuration management. Additionally, the documentation for your chosen IaC tool (e.g., Terraform AWS Provider documentation) will be invaluable.
