---
title: "How can great expectations be integrated into an Airflow project?"
date: "2024-12-23"
id: "how-can-great-expectations-be-integrated-into-an-airflow-project"
---

Alright,  I remember back in 2019, when I was working on that massive data pipeline for a financial institution – we had a real struggle with data quality. The pipeline was moving petabytes of information, and a single corrupted field could cascade into all sorts of headaches. That's when I truly understood the importance of integrating robust data validation early on. Integrating something like great expectations into Airflow became less of a 'nice to have' and more of a 'mission critical' component.

So, how do you weave great expectations into an Airflow setup? It’s not about slapping it on as an afterthought; it’s about architecting it to be an integral part of your data processing workflow. You're effectively building a verification layer that runs alongside your usual transformations.

First, let's understand the fundamental principle: we use great expectations to define expectations about our data — column types, ranges, null values, uniqueness, and all sorts of other conditions — and then we configure Airflow tasks to check if the data meets these expectations. These checks happen before significant processing or, as I often prefer, immediately after data ingestion or any major transformations. This ensures you stop bad data at its source, preventing downstream failures.

We typically break this down into three main parts in our Airflow DAGs:

1.  **The Data Context Setup:** This involves configuring great expectations and establishing the necessary connections to your data stores. I'd usually do this once during the initial setup of the Airflow project or as part of a specific initialization DAG that runs infrequently.
2.  **The Data Validation Task(s):** These are the core tasks that execute the checks defined by great expectations. They extract data, perform the validation, and return success or failure based on the outcome.
3.  **The Post Validation Handling:** This is about how your DAG responds to a failed expectation. Do you immediately stop the pipeline? Send out alerts? Log the issue for review? This is where the orchestration comes into play, allowing you to handle exceptions gracefully.

Let's look at some illustrative code examples using python and a simplified version of an Airflow DAG to demonstrate. For simplicity, I'll assume you already have a great expectations setup with the necessary configurations and defined expectations. In real-world scenarios, you'll have more complex configurations and data connections.

**Example 1: A simple data validation task using a PythonOperator**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import great_expectations as ge
import pandas as pd

def validate_data(**kwargs):
    context = ge.DataContext("./great_expectations") # Assumes great_expectations directory is in the same location
    validator = context.get_validator(
        batch_request=context.get_batch_request(
            datasource_name="my_pandas_datasource",
            data_asset_name="my_data_asset",
        ),
    )

    validation_result = validator.validate()

    if not validation_result["success"]:
        raise Exception("Data validation failed. Check Great Expectations reports for details.")

    print("Data validation succeeded.")
    return True

with DAG(
    dag_id='data_validation_example',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )
```

In this example, `validate_data` loads the data context from where your expectations are defined, retrieves the validator configured for your specific data asset, and performs the validation. If the validation fails, an exception is raised, and Airflow will mark the task as failed. This is a rudimentary example, but it illustrates the core concept. The key here is the utilization of the great expectations' context and validator within the python operator.

**Example 2: Reading data from a data source and performing validation within a PythonOperator**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import great_expectations as ge
import pandas as pd

def fetch_and_validate_data(**kwargs):
    # Simulate fetching data from a source using Pandas
    data = {'col1': [1, 2, 3, 4, 5], 'col2': ['a', 'b', 'c', 'd', 'e']}
    df = pd.DataFrame(data)

    context = ge.DataContext("./great_expectations")
    batch = ge.dataset.PandasDataset(df)

    # Adding a custom expectation here for demonstration purposes, it can be configured in the json file.
    batch.expect_column_values_to_be_in_set("col2", ["a", "b", "c", "d", "e"])

    result = context.run_checkpoint(
        checkpoint_name="my_checkpoint",
        batch_request = {
            "runtime_batch_identifier" : "mydataset",
            "batch_data" : batch
        }
    )

    if not result["success"]:
      raise Exception("Data validation failed. Check Great Expectations reports for details.")

    print("Data validation succeeded.")
    return True


with DAG(
    dag_id='data_validation_dynamic',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    validate_task = PythonOperator(
        task_id='fetch_and_validate',
        python_callable=fetch_and_validate_data,
    )
```

Here, we're demonstrating how you could fetch data (using a dummy Pandas dataframe in this example, but it could be any source, like a database query) and then directly create a batch using the PandasDataset class before utilizing a defined checkpoint to execute validation. We also add an expectation during runtime. Remember, defining your expectations beforehand with the `great_expectations init` commands is generally the recommended approach for larger projects. This example illustrates a more direct way of interacting with GE and is especially useful when handling dynamic data sources in a DAG.

**Example 3: Integrating a data quality check after a transformation task**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime
import great_expectations as ge
import pandas as pd

def transform_data(**kwargs):
    # Simulate a data transformation
    data = {'col1': [1, 2, 3, 4, 5], 'col2': [10, 20, 30, 40, 50]}
    df = pd.DataFrame(data)
    df['col3'] = df['col1'] * df['col2']
    kwargs['ti'].xcom_push(key='transformed_df', value=df.to_json())


def validate_transformed_data(**kwargs):
    transformed_data_json = kwargs['ti'].xcom_pull(key='transformed_df')
    df = pd.read_json(transformed_data_json)

    context = ge.DataContext("./great_expectations")
    batch = ge.dataset.PandasDataset(df)
    batch.expect_column_values_to_not_be_null('col3')

    result = context.run_checkpoint(
        checkpoint_name="my_checkpoint",
         batch_request = {
            "runtime_batch_identifier" : "transformedDataset",
            "batch_data" : batch
         }
    )

    if not result["success"]:
       raise Exception("Data validation failed. Check Great Expectations reports for details.")

    print("Data validation succeeded after transformation.")
    return True

with DAG(
    dag_id='data_validation_transformation',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    start = DummyOperator(task_id='start')
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
    )
    validate_task = PythonOperator(
        task_id='validate_transformed_data',
        python_callable=validate_transformed_data,
    )
    end = DummyOperator(task_id='end')

    start >> transform_task >> validate_task >> end
```

This last example highlights the importance of performing data quality checks after transformations. Here, `transform_data` performs a rudimentary transformation, storing the result in XCom to be used by the validation task. This mirrors real-world pipelines where multiple processing steps are involved, and you often need validation after each of them.

For more detailed information about great expectations, I highly recommend exploring the official documentation on their website, which provides extensive examples and tutorials. For deeper understanding of data quality strategies and data validation approaches within data pipelines, “Building Data Pipelines with Apache Airflow” by Bas Harenslak and “Fundamentals of Data Engineering” by Joe Reis and Matt Housley can provide a solid theoretical framework alongside the practical implementations. Also, the paper "Data Quality: A Brief Overview" by Wang, R.Y., and Strong, D.M. (1996), though older, offers foundational insights into the importance of data quality management.

In essence, integrating great expectations into Airflow is about embedding data quality as a core part of your workflow, not an optional extra. This approach allows you to catch errors early, ensure the reliability of your data, and improve the overall robustness of your data pipelines. It's something I've found to be incredibly valuable in my own projects.
