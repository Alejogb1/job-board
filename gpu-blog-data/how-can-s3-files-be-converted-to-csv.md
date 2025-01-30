---
title: "How can S3 files be converted to CSV using an Airflow task?"
date: "2025-01-30"
id: "how-can-s3-files-be-converted-to-csv"
---
The challenge of transforming large, unstructured S3 datasets into usable CSV formats using Apache Airflow is often encountered in data engineering pipelines. A core requirement is efficient, scalable processing that minimizes memory usage and integrates seamlessly within the Airflow ecosystem. I have personally faced this exact problem, moving large volumes of JSON log files from S3 into a relational database; thus, this response draws from that experience.

The process involves several key steps, primarily leveraging Airflow’s ability to orchestrate external processing rather than performing the transformations directly within the scheduler. Directly converting within the Airflow task itself is generally not scalable or efficient when dealing with larger datasets. Therefore, the strategy centers on triggering external processing resources, usually through tools like AWS Glue or Spark running on EMR. My preferred method has evolved to using AWS Glue due to its serverless nature and ease of integration within the AWS ecosystem, and I will focus on that approach in this response.

The solution hinges on the following core components: an Airflow DAG, an Airflow operator to trigger a Glue job, and the Glue job itself which performs the actual conversion. The Airflow DAG’s role is to orchestrate the job and ensure the processing occurs at the appropriate time, while the Glue job contains the core conversion logic. The Glue job will: 1) read the data from the specified S3 location, 2) perform the transformation from the source format (e.g., JSON, Parquet) to CSV, and 3) write the CSV files back to another S3 location. The advantage of this approach is that it leverages the powerful compute capabilities of Glue without requiring persistent infrastructure management.

Here are three examples demonstrating how this can be achieved in practice:

**Example 1: Basic CSV Conversion from JSON with AWS Glue**

This example demonstrates a basic scenario of converting JSON data, where each JSON object represents a row, to CSV using a PySpark Glue job. This utilizes the `aws_glue_job_operator` available in the `airflow.providers.amazon.aws.operators.glue`.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.glue import AwsGlueJobOperator
from datetime import datetime

with DAG(
    dag_id="s3_json_to_csv_glue",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    s3_json_to_csv_task = AwsGlueJobOperator(
        task_id="s3_json_to_csv",
        job_name="json-to-csv-converter-job", # Replace with your glue job name
        region_name="your-aws-region",        # Replace with your aws region
        aws_conn_id="aws_default", # Replace with your AWS connection ID or remove if using default connection.
    )
```

**Commentary:**

*   This snippet establishes a minimal Airflow DAG utilizing the `AwsGlueJobOperator`.
*   The `job_name` parameter points to the AWS Glue job defined within the AWS ecosystem. Note that the Glue job itself must be created *separately* within the AWS environment. The creation of Glue jobs themselves would be out of scope for this question.
*   `region_name` specifies the AWS region where the Glue job is located.
*   `aws_conn_id` specifies the Airflow connection for AWS access. Ensure an appropriate connection is configured within Airflow beforehand.
*   This DAG does not involve any complex dependencies or scheduling logic, illustrating how it simply initiates the external processing within AWS Glue. It is expected that your Glue job itself contains the core transformation logic using PySpark to read JSON and write CSV.

**Example 2: More Complex CSV Conversion with Arguments**

In this example, we expand the previous example by incorporating job arguments which allows for more dynamic configuration of the Glue job. This is useful for specifying input and output paths at runtime.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.glue import AwsGlueJobOperator
from datetime import datetime

with DAG(
    dag_id="s3_json_to_csv_glue_args",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    s3_json_to_csv_task = AwsGlueJobOperator(
        task_id="s3_json_to_csv",
        job_name="json-to-csv-converter-job", # Replace with your glue job name
        region_name="your-aws-region", # Replace with your aws region
        aws_conn_id="aws_default",  # Replace with your AWS connection ID or remove if using default connection.
        arguments={
            "--source_s3_path": "s3://your-source-bucket/your-input-path/",  # Replace with input path
            "--destination_s3_path": "s3://your-destination-bucket/your-output-path/", # Replace with output path
        },
    )
```

**Commentary:**

*   The `arguments` dictionary allows the passing of dynamic parameters to the Glue job.
*   Inside your Glue job's PySpark code, you can access these arguments using `sys.argv`. For example, `source_path = sys.argv[1]` will access the value assigned to `--source_s3_path`.
*   This adds another layer of flexibility, allowing the DAG to specify which S3 locations are being processed without modifying the Glue job itself. This makes it reusable for multiple datasets and increases the adaptability of the entire system.
*   It is crucial that the arguments are properly named and the Glue job expects these parameters. The argument names are Glue specific and pre-fixed by double dashes.

**Example 3: Handling Partitioned Data and Different File Formats**

This example illustrates how a Glue job can handle data that might be partitioned and not in a flat JSON format. Here, we assume the input data is Parquet, which could result from a preceding step and is commonly partitioned by date. This expands the complexity of the conversion process. The Airflow setup would remain almost identical, but the Glue job would involve slightly different processing logic.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.glue import AwsGlueJobOperator
from datetime import datetime

with DAG(
    dag_id="s3_parquet_to_csv_glue_partitioned",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    s3_parquet_to_csv_task = AwsGlueJobOperator(
        task_id="s3_parquet_to_csv",
        job_name="parquet-to-csv-converter-job", # Replace with your glue job name
        region_name="your-aws-region",  # Replace with your aws region
        aws_conn_id="aws_default",  # Replace with your AWS connection ID or remove if using default connection.
        arguments={
            "--source_s3_path": "s3://your-source-bucket/your-partitioned-parquet-path/", # Replace with your input path
            "--destination_s3_path": "s3://your-destination-bucket/your-output-path/", # Replace with your output path
            "--partition_column": "date",
        },
    )
```
**Commentary:**

*   The crucial difference in this case lies within the AWS Glue job script. The PySpark code within Glue would use `.read.parquet()` instead of `.read.json()`, and leverage the `--partition_column` argument to handle the partitioned data correctly.
*   The Glue job logic would typically load the Parquet files, read the data with respect to partitioning, and then write it out as a single or multiple CSV files depending on the overall data volume and requirements. This is an extension beyond the base conversion.
*   This demonstrates a scalable approach for handling large volumes of partitioned data, where Glue parallelizes the reading and conversion of individual partitions using PySpark.

**Resource Recommendations:**

For those aiming to gain deeper understanding of this topic, I recommend focusing on the following areas:

1.  **Apache Airflow Documentation:** The official documentation for Airflow, especially the section pertaining to AWS integration and operators, is essential. Pay close attention to the `AwsGlueJobOperator` and its available configurations.
2.  **AWS Glue Documentation:** Thoroughly reviewing AWS Glue’s documentation is a must. Focus on how to create Glue jobs (including PySpark and Glue Studio), understand job arguments, and configure Spark configurations for optimal performance.
3.  **PySpark Tutorials and Guides:** A strong grasp of PySpark is needed for developing efficient and scalable data transformation within your Glue jobs. Concentrate on I/O operations (reading and writing different formats), data transformation techniques, and proper handling of partitioning.
4.  **S3 best practices:** Ensure you are following S3 best practices. This includes optimizing the structure of the bucket, appropriate security, and understanding the cost implications of storage and API calls.

In summary, orchestrating S3 to CSV conversion with Airflow and AWS Glue is a scalable and robust solution. The Airflow DAG serves as the scheduler, while the heavy lifting of data transformation occurs within AWS Glue, utilizing PySpark for parallel processing. Understanding the core principles behind data engineering patterns and proper configuration of tools is vital for achieving successful outcomes. The provided code examples provide a solid foundation for developing tailored solutions.
