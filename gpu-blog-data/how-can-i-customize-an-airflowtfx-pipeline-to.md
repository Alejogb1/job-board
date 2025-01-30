---
title: "How can I customize an Airflow/TFX pipeline to upload files to S3?"
date: "2025-01-30"
id: "how-can-i-customize-an-airflowtfx-pipeline-to"
---
A critical aspect of deploying machine learning models using Airflow and TFX is the effective management of intermediate and final artifacts, often requiring integration with cloud storage. I've found that direct manipulation of pipeline components often proves cumbersome. Instead, using the flexibility of Airflow combined with custom TFX components offers a robust solution for uploading files to S3. This approach addresses the inherent limitation in TFX pipelines, which primarily focus on structured data and model assets, not arbitrary file uploads.

Specifically, TFX's core design centers around orchestrating TensorFlow-based workflows. While TFX facilitates model saving and metadata tracking, its built-in functionalities do not directly support uploading arbitrary files such as processed data, reports, or configuration files to S3. This necessity frequently arises when building more complex systems, where post-processing or external reporting are critical. My past experiences involved pipeline extensions to not only export models but also push processed output to data lakes on S3 for consumption by downstream analysis teams. I will outline how this can be achieved via customized components injected directly into an Airflow pipeline orchestrating a TFX pipeline.

To customize an Airflow/TFX pipeline for S3 uploads, I leverage the flexibility of Airflow's PythonOperator, integrated with a custom TFX component. The TFX pipeline's structure remains largely intact, allowing us to use familiar TFX components such as ExampleGen, Trainer, and Evaluator as usual. Then, a PythonOperator executes a Python function which implements a custom TFX component that interacts directly with the AWS SDK for Python (Boto3). Within the TFX pipeline, the custom component acts as a bridge, facilitating the transmission of local files to S3.

This solution allows the user to leverage existing infrastructure and does not require the user to manage any additional services besides Airflow and the S3 service. This approach further decouples the core TFX logic from the infrastructure layer by abstracting the S3 functionality into a reusable Python function.

**Code Example 1: Custom TFX Component**

```python
import apache_beam as beam
from tfx.dsl.component.experimental.decorators import component
from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact
from tfx.types.channel import Channel
import boto3
import os
from typing import Dict, List, Optional

@component
def FileUploader(
    input_files: Channel[standard_artifacts.String],
    s3_bucket: str,
    s3_prefix: str,
    s3_region: str,
    output_upload_paths: Channel[standard_artifacts.String]
) -> None:

  """Custom TFX component to upload files to S3."""

  s3 = boto3.resource('s3', region_name=s3_region)

  def upload_file(file_path):
    """Uploads a single file to S3"""
    file_name = os.path.basename(file_path)
    s3_key = os.path.join(s3_prefix, file_name)
    s3.Bucket(s3_bucket).upload_file(file_path, s3_key)
    return f"s3://{s3_bucket}/{s3_key}"

  @beam.ptransform_fn
  def _upload_to_s3(files):
     uploaded_paths = []
     for file_artifact in files:
         uploaded_paths.append(upload_file(file_artifact.uri))
     return uploaded_paths

  uploaded_paths = input_files.get().pipe(_upload_to_s3)
  output_upload_paths.set(uploaded_paths)
```

This example defines a custom TFX component, `FileUploader`, designed for file uploads to S3. It accepts a TFX `Channel` of file paths as input, along with S3 bucket and prefix information, including the S3 region. The component leverages Boto3 to create an S3 resource and implements a `upload_file` function to upload each file in the input `Channel` to S3, generating an output `Channel` containing the S3 paths. A key consideration here is the use of Beam’s `ptransform_fn`, allowing us to iterate over the input `Channel` and execute the upload operation for each element in the channel within the Beam context.

**Code Example 2: Airflow PythonOperator**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.proto import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.types.channel import Channel

def create_tfx_pipeline_and_run(**kwargs):
    """Creates and executes the TFX pipeline with custom S3 upload."""

    # Placeholder file generator for demonstration
    def create_dummy_files():
        files = []
        for i in range(2):
           with open(f"dummy_file_{i}.txt", "w") as f:
               f.write(f"This is dummy file {i}")
           files.append(f"dummy_file_{i}.txt")
        return files

    # Call the file generator to create the example files
    dummy_files = create_dummy_files()

    # Initialize the TFX pipeline components (omitted for brevity)
    # TFX pipeline components such as ExampleGen, Trainer, Evaluator would be
    # declared here, but we're going to create dummy components for this example.

    @component
    def DummyArtifactGenerator(output_example: Channel[standard_artifacts.String]):
      output_example.set(dummy_files)

    # Create dummy component artifact
    dummy_output_artifact = Channel(type=standard_artifacts.String)

    # Create a dummy component to generate an artifact
    dummy_artifact_generator = DummyArtifactGenerator(output_example=dummy_output_artifact)

    #Create custom component from example 1
    file_uploader = FileUploader(
        input_files=dummy_output_artifact,
        s3_bucket="my-s3-bucket",
        s3_prefix="my-s3-prefix",
        s3_region="us-west-2",
        output_upload_paths = Channel(type=standard_artifacts.String),
    )

    components = [
        dummy_artifact_generator,
        file_uploader
    ]

    # TFX pipeline definition
    pipeline_definition = pipeline.Pipeline(
       pipeline_name="s3_upload_pipeline",
       pipeline_root="/tmp/tfx_pipeline",
       components=components,
       enable_cache=False,
    )

    # Airflow configuration for the TFX pipeline
    pipeline_config = AirflowPipelineConfig()

    # Run the TFX pipeline using Airflow
    dag_runner = AirflowDagRunner(config=pipeline_config)
    dag_runner.run(pipeline_definition)

with DAG(
    dag_id="tfx_s3_upload_dag",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    run_tfx_pipeline = PythonOperator(
        task_id="run_tfx_pipeline",
        python_callable=create_tfx_pipeline_and_run,
    )
```

This code demonstrates how to embed the custom TFX component within an Airflow DAG. A key part is creating a custom `PythonOperator` named `run_tfx_pipeline` that houses a `create_tfx_pipeline_and_run` function. Inside this function, a TFX pipeline is created. We replace the other traditional TFX components to save space with a dummy component that just generates a file, whose output we then feed into the `FileUploader` component we defined earlier. This demonstrates how to hook in the custom component in a broader TFX pipeline. We also pass S3 bucket name, prefix, and the S3 region as arguments to the custom `FileUploader` component. Airflow executes the function, which then triggers the TFX pipeline.

**Code Example 3: Custom component configuration.**

```python
import os
from typing import Dict, List, Optional

import apache_beam as beam
import boto3
from tfx.dsl.component.experimental.decorators import component
from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact
from tfx.types.channel import Channel


@component
def FileUploader(
    input_files: Channel[standard_artifacts.String],
    s3_bucket: str,
    s3_prefix: str,
    s3_region: str,
    output_upload_paths: Channel[standard_artifacts.String],
    s3_config: Optional[Dict] = None,
) -> None:
    """Custom TFX component to upload files to S3 with config."""

    if s3_config:
        s3 = boto3.resource(
            "s3",
            region_name=s3_region,
            aws_access_key_id=s3_config.get("aws_access_key_id"),
            aws_secret_access_key=s3_config.get("aws_secret_access_key"),
            endpoint_url=s3_config.get("endpoint_url"),
        )
    else:
      s3 = boto3.resource("s3", region_name=s3_region)

    def upload_file(file_path):
      file_name = os.path.basename(file_path)
      s3_key = os.path.join(s3_prefix, file_name)
      s3.Bucket(s3_bucket).upload_file(file_path, s3_key)
      return f"s3://{s3_bucket}/{s3_key}"

    @beam.ptransform_fn
    def _upload_to_s3(files):
      uploaded_paths = []
      for file_artifact in files:
        uploaded_paths.append(upload_file(file_artifact.uri))
      return uploaded_paths

    uploaded_paths = input_files.get().pipe(_upload_to_s3)
    output_upload_paths.set(uploaded_paths)
```
This revised example adds a configurable element to the component. Now the `FileUploader` accepts an optional `s3_config` dictionary. This allows for configurations other than simply using environment variables. This becomes critical when one needs to access S3 through specific security roles, such as those typically used for CI/CD pipelines. The inclusion of an `endpoint_url` as an optional configuration allows users to leverage custom endpoints, such as those provided by local S3 emulators or specific S3 implementations.

**Resource Recommendations:**

For deepening your understanding of TFX, the official TensorFlow Extended documentation is invaluable. Focus specifically on understanding TFX components, pipelines, and concepts such as channels and artifacts. For Airflow, the Apache Airflow official documentation provides in-depth details on task creation, DAG structuring, and workflow orchestration, specifically the section on PythonOperators. The Boto3 documentation is essential for mastering AWS SDK interactions. Thoroughly reviewing these resources will provide a solid foundation to further develop custom components and integrate them within your machine learning pipelines. Familiarity with Apache Beam’s programming model and concepts will also allow one to better grasp how TFX manages dataflow within a TFX pipeline.
