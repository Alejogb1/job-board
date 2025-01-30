---
title: "How can I use Python to integrate Vertex AI predictions into BigQuery?"
date: "2025-01-30"
id: "how-can-i-use-python-to-integrate-vertex"
---
Accessing machine learning predictions directly within BigQuery offers significant advantages in terms of data analysis efficiency, particularly when dealing with large datasets. My experience with a high-volume recommendation engine at my previous role highlighted the necessity of streamlined access to predictions. We shifted from manual export and upload cycles to a direct, query-based approach, reducing latency and complexity dramatically. Specifically, integrating Vertex AI predictions into BigQuery is achievable through several methods, each with its trade-offs concerning real-time performance versus cost and complexity. I'll focus on two primary approaches: using custom functions (UDFs) and, where appropriate, the Vertex AI BigQuery integration capabilities.

The first and potentially most flexible method involves creating a User-Defined Function (UDF) in BigQuery that invokes a Vertex AI endpoint. This approach is particularly useful when needing to execute custom logic or handle non-standard prediction requests. The core idea is to write a Python function that takes data from a BigQuery table, sends it to your Vertex AI model endpoint, and returns the prediction result, formatted for insertion back into BigQuery. The advantage here is customizability, accommodating any pre- or post-processing, as well as complex input and output data structures. The disadvantage is the increased complexity in managing UDF deployments and the potential for latency due to network calls.

Let's consider a basic example where a table in BigQuery contains numerical features that we want to use as inputs for a simple classification model on Vertex AI. Here's the Python code defining the UDF:

```python
import google.auth
from google.cloud import aiplatform
import json

def predict_with_vertex(data_str, project_id, location, endpoint_id):
  """
  Takes a stringified JSON payload from BigQuery, makes a Vertex AI prediction,
  and returns the prediction as a stringified JSON.

  Args:
      data_str (str): Stringified JSON of feature data from BigQuery.
      project_id (str): Google Cloud project ID.
      location (str): Region where the endpoint is located.
      endpoint_id (str): ID of the Vertex AI endpoint.
  Returns:
      str: Stringified JSON of predictions.
  """
  credentials, project = google.auth.default()
  aiplatform.init(project=project_id, location=location, credentials=credentials)

  try:
    data = json.loads(data_str)
    instances = [data] # Vertex AI expects a list of instances

    endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}")
    response = endpoint.predict(instances=instances)
    predictions = response.predictions
    return json.dumps(predictions)

  except Exception as e:
     return json.dumps({"error": str(e)})

```

This UDF, named `predict_with_vertex`, receives a JSON string representation of the input features from BigQuery. It then authenticates with Google Cloud, initializes the Vertex AI client, and sends the input data to the specified endpoint. Crucially, error handling is incorporated to capture issues during prediction, returning a JSON response indicating any potential problems. Finally, it returns a stringified JSON containing prediction results from the Vertex AI model. To use it, the UDF needs to be deployed to BigQuery and then called inside a SQL query.

The deployment within BigQuery looks like this:

```sql
CREATE OR REPLACE FUNCTION
  `your-project.your_dataset.vertex_prediction`(data STRING)
  RETURNS STRING
  LANGUAGE js AS """
  return predict_with_vertex(data, 'your-project', 'your-region', 'your-endpoint-id');
  """
  OPTIONS (
    library=["gs://your-bucket/your-udf.zip"]  -- Path to your zipped Python dependencies
  );
```

This statement creates a UDF in your BigQuery dataset. The javascript wrapper allows us to call Python code. The `library` option provides the path to a ZIP archive containing the dependencies, which would include the above `predict_with_vertex` function alongside `google-cloud-aiplatform` and `google-auth` and any other required packages.

Finally, you would use it in a query like:

```sql
SELECT
    *,
    `your-project.your_dataset.vertex_prediction`(TO_JSON_STRING(STRUCT(feature1, feature2, feature3))) AS predictions
  FROM
   `your-project.your_dataset.your_table`;
```

This SQL statement retrieves all rows from `your_table`, constructs a JSON object of feature values, calls the UDF (`vertex_prediction`) on each row, and adds the returned predictions to each row as a new column. This method works reliably for batch predictions when there is a reasonable throughput required and an added benefit of custom transformation within the UDF itself. It is suitable for most batch-oriented applications. The primary challenge with this approach is maintaining dependencies and ensuring the UDF execution environment is properly configured.

The second approach, leveraging the built-in BigQuery integration, offers a more direct and arguably streamlined experience, though it might lack some of the flexibility of UDFs. This integration utilizes BigQuery ML (BQML) and its remote model feature. This mechanism is better suited for scenarios where the model input format directly matches the table schema or where post-processing can occur directly within a SQL query. This integration works by defining a BQML model that corresponds to a Vertex AI Endpoint. BQML handles the communication with Vertex AI, removing the need for manual function deployment and management.

For instance, let us assume that you have a regression model that outputs a prediction based on features already present in a BigQuery Table called `my_regression_table`. To configure BigQuery to use the Vertex AI Endpoint you must define a `REMOTE MODEL`. The query syntax would be:

```sql
CREATE OR REPLACE MODEL `your-project.your_dataset.my_remote_model`
  REMOTE WITH CONNECTION `your-project.your_region.vertex_connection`
  OPTIONS(
    endpoint = "projects/your-project/locations/your-region/endpoints/your-endpoint-id",
    input_label_columns = ['feature1', 'feature2', 'feature3']
  );
```

This SQL command creates a BQML model named `my_remote_model`. The `CONNECTION` refers to the connection established between BigQuery and the Vertex AI project to enable communication. Note that this connection has to be preconfigured with the required permissions. The `endpoint` parameter tells BQML where to reach Vertex AI to get predictions. The `input_label_columns` parameter maps the columns in the query that must be passed to the Vertex AI endpoint as features.

Now, to retrieve predictions using BQML, you can use the `ML.PREDICT` function:

```sql
SELECT
    *,
    ML.PREDICT(MODEL `your-project.your_dataset.my_remote_model`, (SELECT AS STRUCT feature1, feature2, feature3 from t) ).predicted_value
  FROM
    `your-project.your_dataset.my_regression_table` as t;
```

This query selects all the columns from `my_regression_table` and creates a new column `predicted_value`. This new column is the result of invoking the `my_remote_model` and passing the `feature1`, `feature2` and `feature3` columns as input to the model. Note that there is no need for JSON serialization or deserialization as it is done implicitly with the BQML prediction call. This is the primary advantage of this method. The prediction response comes back directly in a structure that is easy to work with.

Both approaches—UDFs and native BQML integration—offer mechanisms for integrating Vertex AI predictions into BigQuery, though they are suitable for slightly different use cases. UDFs provide high customizability and the ability to include pre- and post-processing logic, while BQML integration offers a streamlined, direct approach ideal for standard prediction requests where the column mapping is simple. The choice between them should be based on specific requirements for data transformation, the complexity of prediction logic, latency requirements, and the desired level of management overhead.

For additional guidance, I would recommend reviewing Google Cloud documentation on BigQuery UDFs and BQML, as well as exploring the official Vertex AI documentation, including tutorials and examples. Specifically, look for material focusing on connecting to external services from within BigQuery. Understanding the IAM permissions required for both the UDF and BQML integration approaches is also crucial, to ensure both are properly configured and have access to Vertex AI resources. Additionally, explore community forums and examples related to Google Cloud Platform integration for deeper insight into production challenges and best practices.
