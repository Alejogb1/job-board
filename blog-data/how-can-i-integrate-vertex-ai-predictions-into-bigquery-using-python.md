---
title: "How can I integrate Vertex AI predictions into BigQuery using Python?"
date: "2024-12-23"
id: "how-can-i-integrate-vertex-ai-predictions-into-bigquery-using-python"
---

, let's tackle integrating Vertex AI predictions directly into BigQuery using Python – a challenge I've definitely navigated more than a few times in my past projects. This isn't just about running some API calls; it's about building a resilient, scalable data pipeline. My experience involves building real-time fraud detection systems and large-scale predictive maintenance platforms, both of which heavily leveraged this very integration. So, let's break down how it's done, step-by-step, focusing on practical considerations rather than abstract theory.

The key here lies in orchestrating a few essential components: the Vertex AI prediction endpoint, the BigQuery client, and some robust Python code to handle data transformation and invocation. The fundamental approach involves preparing your data in BigQuery, fetching it, formatting it to match what your Vertex AI model expects, sending the prediction requests, and finally, inserting the results back into BigQuery. We'll avoid convoluted abstractions and focus on clarity.

First off, we need to establish the connection. This will generally involve setting up the google cloud client library authentication. I assume you already have a properly configured service account with the necessary permissions for both Vertex AI and BigQuery. If not, that's step zero. Make sure it has at least the 'vertex ai user' role and the 'bigquery data editor' role, or equivalent custom roles. Now, let’s look at some Python code.

**Code Snippet 1: Basic Data Fetch and Preparation**

```python
from google.cloud import bigquery
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import json

def fetch_data_from_bq(project_id, dataset_id, table_id, query_filter=None):
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    if query_filter:
        query += f" WHERE {query_filter}"
    query_job = client.query(query)
    results = query_job.result()
    
    data = []
    for row in results:
        row_dict = dict(row)
        data.append(row_dict)  # each row as a dictionary
    
    return data

def format_for_vertex(data):
  formatted_instances = []
  for row in data:
    # Convert each value to a google.protobuf.struct_pb2.Value object
    instance = {k: Value(string_value=str(v) if v is not None else "") for k,v in row.items()}
    formatted_instances.append(instance)
  return formatted_instances


if __name__ == '__main__':
    project_id = "your-gcp-project-id"  # Replace with your project ID
    dataset_id = "your_dataset"      # Replace with your dataset ID
    table_id = "your_table"         # Replace with your table ID
    query_filter = "some_column > 10" #Optional
    
    data = fetch_data_from_bq(project_id, dataset_id, table_id, query_filter)
    formatted_data = format_for_vertex(data)
    print(f"Prepared {len(formatted_data)} instances for Vertex AI")
    # this point you'd typically call the prediction endpoint.
    # example of what data looks like before Vertex.
    print(json.dumps(formatted_data[0], indent=2))

```

This first snippet demonstrates how I'd typically grab the data from BigQuery and prepare it. Note that this fetches everything from the table, with an optional filter. In real-world settings, I often utilize more complex queries and windowing functions to retrieve the exact data slice needed for the batch prediction. The `format_for_vertex` function specifically prepares the data into a dictionary containing google protobuf Value objects, which matches what Vertex AI’s `predict()` method expects for structured data. This conversion avoids type mismatches during prediction.

Now, lets look at interacting with Vertex AI. This requires instantiating a `PredictionServiceClient`.

**Code Snippet 2: Invoking Vertex AI Endpoint**

```python
from google.cloud import aiplatform
from google.protobuf.json_format import MessageToDict

def invoke_vertex_ai_prediction(project_id, location, endpoint_id, instances):
    aiplatform.init(project=project_id, location=location)
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)

    # the raw prediction data
    response = endpoint.predict(instances=instances)
    predictions = [MessageToDict(prediction) for prediction in response.predictions]
    return predictions

if __name__ == '__main__':
    project_id = "your-gcp-project-id" # Replace with your project ID
    location = "us-central1"         # Replace with your location
    endpoint_id = "your-endpoint-id"  # Replace with your endpoint ID
    
    # Assuming formatted_data was created in the prior example.
    # For brevity, lets imagine some sample formatted_data:
    formatted_data = [
        {'feature1': Value(string_value='10'), 'feature2': Value(string_value='20')},
        {'feature1': Value(string_value='30'), 'feature2': Value(string_value='40')}
    ]

    predictions = invoke_vertex_ai_prediction(project_id, location, endpoint_id, formatted_data)
    print(f"Retrieved {len(predictions)} predictions from Vertex AI")
    print(json.dumps(predictions, indent=2))
```

This second snippet demonstrates how I'd interact with the Vertex AI prediction endpoint after formatting the data from the previous step.  The endpoint is configured via its ID. The `MessageToDict` function converts the protobuf object to a native dictionary which can be more conveniently used in python downstream.

Finally, the last piece is pushing the results back into BigQuery alongside the original input data. I will emphasize here that you *must* ensure your schema on your destination table matches what the prediction endpoint returns. That includes nesting of objects if needed.

**Code Snippet 3: Inserting Prediction Results into BigQuery**

```python
from google.cloud import bigquery
import json

def insert_predictions_to_bq(project_id, dataset_id, table_id, original_data, predictions):
    client = bigquery.Client(project=project_id)
    
    # Build data with original input and prediction output
    rows_to_insert = []
    for original_row, prediction in zip(original_data, predictions):
        # Construct a dictionary containing both original and prediction output.
        combined_row = dict(original_row) # copy the original
        combined_row['prediction'] = prediction  # add the prediction 
        rows_to_insert.append(combined_row)
   
    table_ref = client.dataset(dataset_id).table(table_id)
    errors = client.insert_rows_json(table_ref, rows_to_insert)
    if errors:
        print(f"Encountered errors while inserting rows: {errors}")
    else:
        print(f"Successfully inserted {len(rows_to_insert)} rows into BigQuery")

if __name__ == '__main__':
    project_id = "your-gcp-project-id"     # Replace with your project ID
    dataset_id = "your_dataset"         # Replace with your dataset ID
    table_id = "your_prediction_table"     # Replace with your destination table ID
    
    # Assuming original data and predictions are created in previous steps
    # example data.
    original_data = [
        {'feature1': '10', 'feature2': '20'},
        {'feature1': '30', 'feature2': '40'}
    ]

    predictions = [
      {'prediction_output_value': 0.5},
      {'prediction_output_value': 0.9}
    ]
    insert_predictions_to_bq(project_id, dataset_id, table_id, original_data, predictions)
```

This final snippet shows how I'd typically write the prediction results back into BigQuery. This is where careful schema management comes in. In my experience, it's frequently useful to include the original input data alongside the prediction output to facilitate debugging and analysis. Ensure you create a new table or ensure the schema of the destination table matches with the data being pushed via the `insert_rows_json` method.

For further learning and a deeper dive into related topics, I highly recommend reviewing the official Google Cloud documentation on Vertex AI and BigQuery. Additionally, "Designing Data-Intensive Applications" by Martin Kleppmann offers incredible insight into the underlying concepts of data processing at scale, which is critical for making these integrations efficient. You might also find "Python for Data Analysis" by Wes McKinney helpful for some Python-specific tips.  Finally, the official google cloud python libraries themselves provide excellent examples and should be consulted regularly for the most up to date information.

These examples are a solid foundation, but each system will have its unique quirks and challenges. This process requires careful monitoring, error handling (that’s beyond the scope here but is critical in production), and performance considerations, especially as your data grows. Remember, this is about building robust and reliable data pipelines, not just running a single API call. Happy coding.
