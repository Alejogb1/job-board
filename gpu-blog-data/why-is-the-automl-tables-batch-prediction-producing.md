---
title: "Why is the AutoML Tables batch prediction producing an empty table in BigQuery?"
date: "2025-01-30"
id: "why-is-the-automl-tables-batch-prediction-producing"
---
After spending the last few weeks troubleshooting several issues with AutoML Tables batch predictions, I've observed a common pitfall that often results in an empty output table in BigQuery: incorrect or misconfigured input data specifications. Specifically, if the schema of your input data does not precisely match the schema expected by the AutoML model, even if the data *appears* correct at a high level, the prediction job may fail to generate output. This failure, however, is often silent, not accompanied by explicit error messages related to schema mismatch.

The core of the problem lies in how AutoML Tables manages feature expectations. When an AutoML model is trained, it infers a schema from the training data. This schema includes not only column names but, critically, the data types associated with each column. During prediction, the system expects input data to adhere rigorously to this inferred schema. Discrepancies, such as an integer column being supplied as a string, or a date column using a different formatting than expected, will cause issues. AutoML, in its current iteration, doesn't automatically handle these subtle differences in the batch prediction pipeline. It simply fails to generate predictions for inputs that do not conform to its schema, and the batch prediction process returns an empty result.

Let's illustrate this with a few practical examples and associated code snippets. Assume that our AutoML model was trained on a dataset that includes a column named `order_date`, which is stored internally by AutoML as a date type in YYYY-MM-DD format.

**Example 1: Incorrect Date Format**

Suppose our input data, for batch prediction, stores the `order_date` column as a string but in the format MM/DD/YYYY.

```python
from google.cloud import bigquery
from google.cloud import automl_v1beta1 as automl

# Assume project_id, dataset_id, table_id, and model_id are defined

client = bigquery.Client(project=project_id)

query = f"""
    CREATE OR REPLACE TEMP TABLE temp_input_table AS
    SELECT 
        '12345' as order_id,
        '12/25/2023' as order_date, 
        'Product A' as product_name,
        12.99 as price
    """

client.query(query).result()


input_config = automl.BatchPredictInputConfig(
    bigquery_source=automl.BigQuerySource(
        input_uri=f"bq://{project_id}.{dataset_id}.temp_input_table"
    )
)


output_config = automl.BatchPredictOutputConfig(
    bigquery_destination=automl.BigQueryDestination(
        output_uri=f"bq://{project_id}.{dataset_id}.prediction_results"
    )
)


prediction_client = automl.PredictionServiceClient()
name = prediction_client.model_path(project_id, location_id, model_id)


batch_predict_request = automl.BatchPredictRequest(
    name=name,
    input_config=input_config,
    output_config=output_config,
)

operation = prediction_client.batch_predict(request=batch_predict_request)
operation.result()

print("Prediction completed!")
```

In this example, the prediction job will complete successfully (as seen in the print statement), but the `prediction_results` table will remain empty. The issue is that the input `order_date` column, supplied as '12/25/2023', does not align with the YYYY-MM-DD format expected by the model, resulting in no successful predictions.

**Example 2: Data Type Mismatch**

Let’s assume that our model was trained with ‘price’ as a FLOAT64. If we now supply it as a string during batch prediction, it will likely result in the same issue, an empty output table.

```python
from google.cloud import bigquery
from google.cloud import automl_v1beta1 as automl

# Assume project_id, dataset_id, table_id, and model_id are defined

client = bigquery.Client(project=project_id)

query = f"""
    CREATE OR REPLACE TEMP TABLE temp_input_table AS
    SELECT 
        '12345' as order_id,
        '2023-12-25' as order_date,
        'Product A' as product_name,
        '12.99' as price
    """

client.query(query).result()


input_config = automl.BatchPredictInputConfig(
    bigquery_source=automl.BigQuerySource(
        input_uri=f"bq://{project_id}.{dataset_id}.temp_input_table"
    )
)


output_config = automl.BatchPredictOutputConfig(
    bigquery_destination=automl.BigQueryDestination(
        output_uri=f"bq://{project_id}.{dataset_id}.prediction_results"
    )
)


prediction_client = automl.PredictionServiceClient()
name = prediction_client.model_path(project_id, location_id, model_id)


batch_predict_request = automl.BatchPredictRequest(
    name=name,
    input_config=input_config,
    output_config=output_config,
)

operation = prediction_client.batch_predict(request=batch_predict_request)
operation.result()

print("Prediction completed!")
```

Here, although we are now using the correct YYYY-MM-DD format for the `order_date` column, we are still supplying 'price' as a string rather than a numerical data type.  Again, the job will complete, but `prediction_results` will remain empty. AutoML expects a numerical value, not a string representation of one, even though they might seem similar at first glance.

**Example 3: Correct Schema Input**

Let's modify the example by ensuring that the `order_date` and `price` columns match what the model expects.

```python
from google.cloud import bigquery
from google.cloud import automl_v1beta1 as automl

# Assume project_id, dataset_id, table_id, and model_id are defined

client = bigquery.Client(project=project_id)

query = f"""
    CREATE OR REPLACE TEMP TABLE temp_input_table AS
    SELECT 
        '12345' as order_id,
        DATE('2023-12-25') as order_date,
        'Product A' as product_name,
        12.99 as price
    """

client.query(query).result()


input_config = automl.BatchPredictInputConfig(
    bigquery_source=automl.BigQuerySource(
        input_uri=f"bq://{project_id}.{dataset_id}.temp_input_table"
    )
)


output_config = automl.BatchPredictOutputConfig(
    bigquery_destination=automl.BigQueryDestination(
        output_uri=f"bq://{project_id}.{dataset_id}.prediction_results"
    )
)


prediction_client = automl.PredictionServiceClient()
name = prediction_client.model_path(project_id, location_id, model_id)


batch_predict_request = automl.BatchPredictRequest(
    name=name,
    input_config=input_config,
    output_config=output_config,
)

operation = prediction_client.batch_predict(request=batch_predict_request)
operation.result()

print("Prediction completed!")
```

In this final example, we now explicitly use `DATE('2023-12-25')` to ensure that the `order_date` is in the correct format and type, and we supply `12.99` directly as a numerical value. This time, the prediction job should generate output and populate the `prediction_results` table with results.

It is crucial to stress that schema adherence also extends to the presence or absence of columns. If your prediction input table includes additional columns not present during training, it generally does not cause errors. However, the order of columns should match the original training set; AutoML expects the column order of the input data to correspond to the order in which the features were used during training.

To avoid these issues, I have found these steps essential:

1.  **Inspect the training data schema:** Before initiating a batch prediction, carefully examine the schema of the data used to train your AutoML model. Pay special attention to the column data types, especially DATE and numerical types. You can do this within the AutoML UI or via the API.

2.  **Explicitly cast input data:** When preparing data for batch prediction, use explicit type casting (like the `DATE()` function in BigQuery) to ensure that your input data's data types are identical to those expected by the model.

3.  **Test with small batches:** Begin with a very small batch of data (e.g. 5-10 rows) for a test prediction run. This enables you to identify problems quickly and inexpensively. Examine the output table to see if data appears in the first place before investing in a larger prediction run.

4.  **Leverage error logging:** While AutoML doesn't give direct schema mismatch errors, it is beneficial to examine the job details using the Cloud Logging console associated with the project, searching for errors and warning related to the batch prediction job. Sometimes, underlying issues might be logged here even if not surfacing directly via the API.

To learn more about debugging and addressing this kind of behavior, I would recommend consulting the official Google Cloud documentation on AutoML Tables, which frequently updates and contains troubleshooting guides. There are also many community tutorials and examples available online regarding BigQuery and AutoML integration, which can help when trying to diagnose problems. Lastly, using the GCP status dashboard is always helpful to verify if underlying infrastructure issues are causing problems and not your code or data. By attending to these points, you should significantly reduce the incidence of empty tables when running batch predictions with AutoML Tables.
