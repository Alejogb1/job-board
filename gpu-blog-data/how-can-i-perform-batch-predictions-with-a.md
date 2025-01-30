---
title: "How can I perform batch predictions with a Vertex AI model using files from Cloud Storage?"
date: "2025-01-30"
id: "how-can-i-perform-batch-predictions-with-a"
---
Batch prediction on Vertex AI using Cloud Storage files requires careful orchestration of input data location, output destination, and the prediction request structure itself. I've handled numerous deployment scenarios where large datasets stored in Cloud Storage needed to be processed through machine learning models. The core issue isn't about the model's mechanics, but rather about efficiently connecting your Cloud Storage data to Vertex AI's prediction infrastructure. Let me explain the process and illustrate it with practical examples.

At its foundation, Vertex AI batch prediction leverages a distributed processing framework that reads files from a specified Cloud Storage path, feeds them to your deployed model (either custom-trained or a Google-provided API), and writes the resulting predictions to another Cloud Storage location. You don't directly manage individual prediction calls; instead, you initiate a batch prediction job that handles the underlying parallelization. The key to a successful batch prediction lies in correctly configuring the input and output parameters of this job. This includes specifying the input format (e.g., JSON Lines, CSV) and ensuring that the model is configured to receive inputs compatible with that format.

Let's delve into three common scenarios to clarify the practical application:

**Example 1: Processing JSON Lines Files**

Assume you have a model deployed on Vertex AI that takes a single JSON object as input per prediction. Your data is stored in Cloud Storage as a sequence of JSON objects, one per line within a `.jsonl` file. The following Python code snippet using the Vertex AI SDK illustrates how to configure and initiate a batch prediction job in this case:

```python
from google.cloud import aiplatform

def perform_batch_prediction_jsonl(
    project_id: str,
    location: str,
    model_id: str,
    input_uri: str,
    output_uri: str
):
    """Performs batch prediction using JSON lines input."""

    aiplatform.init(project=project_id, location=location)

    model = aiplatform.Model(model_id)

    batch_prediction_job = model.batch_predict(
        job_display_name="batch_prediction_jsonl_job",
        gcs_source=input_uri,
        gcs_destination_prefix=output_uri,
        instances_format="jsonl",
        predictions_format="jsonl"
    )

    batch_prediction_job.wait()
    print(f"Batch prediction job finished. Results at: {output_uri}")


if __name__ == "__main__":
    PROJECT_ID = "your-gcp-project-id" # Replace with your GCP Project ID
    LOCATION = "us-central1"          # Replace with your desired location
    MODEL_ID = "your-model-id"        # Replace with your model's resource ID
    INPUT_URI = "gs://your-input-bucket/input_data.jsonl" # Replace with your input GCS path
    OUTPUT_URI = "gs://your-output-bucket/output_predictions" # Replace with your desired output GCS path

    perform_batch_prediction_jsonl(
       project_id = PROJECT_ID,
       location = LOCATION,
       model_id = MODEL_ID,
       input_uri = INPUT_URI,
       output_uri = OUTPUT_URI
    )
```

**Explanation:**

1.  **Initialization:**  The code begins by initializing the Vertex AI SDK with your project ID and the location where your model is deployed.
2.  **Model Retrieval:** It then fetches the model resource using its unique identifier.
3.  **Batch Prediction Configuration:** The `model.batch_predict()` method is crucial. Here, `gcs_source` specifies the Cloud Storage location of your input JSONL file, and `gcs_destination_prefix` indicates the desired Cloud Storage path for output.
4.  **Format Declaration:** Critically, the `instances_format="jsonl"` parameter tells Vertex AI that the input data is in JSONL format. Similarly, `predictions_format="jsonl"` states the format for the output.
5.  **Job Execution:** The `batch_prediction_job.wait()` method initiates and monitors the prediction process, returning once the job completes.
6.  **Output location:** The script outputs the Cloud Storage location of your predictions.

**Example 2: Processing CSV Files**

Consider a scenario where your model expects a CSV file as input. Each row in the CSV corresponds to a single prediction input, and you have headers in the file. The Python code below demonstrates how to run a batch prediction job using such files:

```python
from google.cloud import aiplatform

def perform_batch_prediction_csv(
    project_id: str,
    location: str,
    model_id: str,
    input_uri: str,
    output_uri: str
):
    """Performs batch prediction using CSV input."""

    aiplatform.init(project=project_id, location=location)

    model = aiplatform.Model(model_id)

    batch_prediction_job = model.batch_predict(
        job_display_name="batch_prediction_csv_job",
        gcs_source=input_uri,
        gcs_destination_prefix=output_uri,
        instances_format="csv",
        predictions_format="jsonl" # output in JSON lines format
    )

    batch_prediction_job.wait()
    print(f"Batch prediction job finished. Results at: {output_uri}")


if __name__ == "__main__":
    PROJECT_ID = "your-gcp-project-id" # Replace with your GCP Project ID
    LOCATION = "us-central1"          # Replace with your desired location
    MODEL_ID = "your-model-id"        # Replace with your model's resource ID
    INPUT_URI = "gs://your-input-bucket/input_data.csv" # Replace with your input GCS path
    OUTPUT_URI = "gs://your-output-bucket/output_predictions" # Replace with your desired output GCS path

    perform_batch_prediction_csv(
       project_id = PROJECT_ID,
       location = LOCATION,
       model_id = MODEL_ID,
       input_uri = INPUT_URI,
       output_uri = OUTPUT_URI
    )
```

**Explanation:**

1.  **Initialization and Model Retrieval:** Same as the previous example.
2.  **Batch Prediction Configuration:** Here, we specify `instances_format="csv"` to indicate that the input is a CSV file.
3.  **Output Format:** Notice that in this case the output is still set to `predictions_format = "jsonl"`. This demonstrates that you don't need to keep input and output formats the same.
4.  **Job Execution and Output location:** Same as the previous example.

**Example 3: Specifying Instance Schema (for structured data)**

When working with models that expect structured inputs, it can be beneficial to specify an *instance schema*.  This provides type information for the model and ensures data compatibility. I've seen this greatly improve the reliability of batch prediction jobs when the data schema is complex. The following example illustrates this with a hypothetical instance schema for data expected in a JSON lines file:

```python
from google.cloud import aiplatform

def perform_batch_prediction_schema(
    project_id: str,
    location: str,
    model_id: str,
    input_uri: str,
    output_uri: str
):
    """Performs batch prediction with an instance schema."""

    aiplatform.init(project=project_id, location=location)

    model = aiplatform.Model(model_id)

    instance_schema = {
        "type": "object",
        "properties": {
            "feature_1": {"type": "number"},
            "feature_2": {"type": "string"},
            "feature_3": {"type": "array", "items":{"type":"integer"}}
        },
        "required": ["feature_1", "feature_2","feature_3"]
    }

    batch_prediction_job = model.batch_predict(
        job_display_name="batch_prediction_schema_job",
        gcs_source=input_uri,
        gcs_destination_prefix=output_uri,
        instances_format="jsonl",
        predictions_format="jsonl",
        instance_schema=instance_schema
    )

    batch_prediction_job.wait()
    print(f"Batch prediction job finished. Results at: {output_uri}")


if __name__ == "__main__":
    PROJECT_ID = "your-gcp-project-id" # Replace with your GCP Project ID
    LOCATION = "us-central1"          # Replace with your desired location
    MODEL_ID = "your-model-id"        # Replace with your model's resource ID
    INPUT_URI = "gs://your-input-bucket/input_data.jsonl" # Replace with your input GCS path
    OUTPUT_URI = "gs://your-output-bucket/output_predictions" # Replace with your desired output GCS path

    perform_batch_prediction_schema(
       project_id = PROJECT_ID,
       location = LOCATION,
       model_id = MODEL_ID,
       input_uri = INPUT_URI,
       output_uri = OUTPUT_URI
    )
```

**Explanation:**

1.  **Initialization and Model Retrieval:** As before.
2.  **Instance Schema Definition:** The `instance_schema` dictionary defines the structure and data types of your input. In this example it demonstrates an object composed of various primitive and complex types. You'll need to define this schema to align with the input requirements of your model.
3.  **Batch Prediction Configuration:** We provide `instance_schema` as an argument to `batch_predict()`.  The rest of the configuration is similar to the first example, specifying `instances_format` and `predictions_format`.
4.  **Job Execution and Output Location:** The rest of the code handles job execution and outputs the results path.

**Resource Recommendations:**

For further exploration of Vertex AI's capabilities, I recommend referring to the official Google Cloud documentation on Vertex AI batch prediction. Specifically, examine sections detailing input formats, instance schema specifications, and the different batch prediction job configurations. The Python SDK reference guide will provide a detailed overview of the available parameters for methods like `batch_predict()`. Additionally, the concepts of job monitoring and output inspection are crucial to understanding and verifying your results.
