---
title: "Can I deploy ONNX models from a Synapse Notebook to a dedicated Synapse SQL pool?"
date: "2024-12-16"
id: "can-i-deploy-onnx-models-from-a-synapse-notebook-to-a-dedicated-synapse-sql-pool"
---

, let’s dive into this. It's a question I encountered some years back when we were transitioning from purely analytical workflows to integrating more machine learning output within our data warehouse. The goal, as with your case, was to leverage onnx models directly within a Synapse SQL pool, avoiding the extra hops between compute environments. The short answer is: it’s not a straightforward “deploy” in the traditional sense, like you might deploy a web application. Instead, it involves using a combination of mechanisms to achieve a similar effect, primarily by making the model’s predictions accessible to the SQL pool. Let me explain the practical approach we took, and I'll throw in some code snippets to clarify.

The key to accessing the predictions from an onnx model within a Synapse SQL pool isn’t a direct model deployment; rather, it relies on a pattern often called “scoring” or “inferencing.” You won’t install the onnx runtime into the SQL pool itself. Instead, we typically employ an external compute resource to perform the inference—in our case, that was often an Azure Machine Learning compute instance, though other resources like an Azure Function or even an on-premise compute server can also be used. What matters is that this resource can execute your onnx model. Then, the result of this execution needs to be ingested back into the SQL Pool.

Our setup looked like this: A Synapse Notebook was responsible for preparing the input data, initiating the inference process on the remote compute, and then bringing the predicted outputs back into the SQL pool. We also ensured the model itself was accessible from the external compute, typically through Azure Blob Storage. I'll explain why this indirect approach is important. Synapse SQL pools are designed for handling large-scale structured data queries and analytical workloads, not general purpose computation or running machine learning models. That's why we leverage an external execution resource that *is* suited for running our onnx model.

Here's how it breaks down with some practical examples:

**Example 1: Setting up the Inference with Python**

First, let's say you've trained a model and saved it as `my_model.onnx`. In your Synapse Notebook, you’ll need code similar to this. This part focuses on preparing data and initiating the remote inference:

```python
import pandas as pd
import requests
import json

# 1. Load the data to be scored (example data from a SQL query result)
sql_query_result = spark.sql("SELECT feature1, feature2 FROM my_table WHERE ...").toPandas()

# 2. Prepare the data in a JSON format for the inference endpoint
data_to_score = sql_query_result.to_dict(orient='records')

# 3.  Define the endpoint for your remote scoring service
# (Replace with your actual endpoint, potentially using Azure Function or AML endpoint)
inference_endpoint = "https://my-inference-service.azurewebsites.net/score"

# 4. Make the API call to the scoring endpoint
headers = {'Content-Type': 'application/json'}
response = requests.post(inference_endpoint, json=data_to_score, headers=headers)

# 5. Handle the response (assuming the response is also JSON)
if response.status_code == 200:
    predictions = response.json()
    predictions_df = pd.DataFrame(predictions)

    # 6. Store the results back into a new table, or update existing table in your SQL pool
    # (this part needs to be adapted to match your existing SQL infrastructure)
    spark.createDataFrame(predictions_df).write.mode('append').saveAsTable("predictions_table")

    print("Predictions successfully imported")
else:
    print(f"Error: Inference failed with status code {response.status_code}. Reason: {response.text}")

```

In this snippet, we’re assuming an external REST endpoint for the scoring service, often hosted on an Azure function or a similar serverless service. Note the core elements: Data extraction from the SQL pool, data formatting for the API, the remote API call, and finally, importing the results back into a Synapse SQL Pool table.

**Example 2: Scoring using Azure Machine Learning (AML)**

If you’re utilizing Azure Machine Learning, the setup will slightly differ. The AML endpoint allows for a somewhat more streamlined approach:

```python
from azureml.core import Workspace, Environment
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model
import pandas as pd
from azureml.core.conda_dependencies import CondaDependencies
import json

# 1. Initialize Workspace (ensure you have appropriate config)
ws = Workspace.from_config()

# 2. Define your model (replace 'my_model_name' and 'model_version')
model = Model(ws, "my_model_name", version=1)

# 3. Configure inference environment
env = Environment(name="my_onnx_env")
conda_dep = CondaDependencies()
conda_dep.add_conda_package("onnxruntime") # ensure onnx runtime is available
conda_dep.add_conda_package("pandas")
env.python.conda_dependencies = conda_dep

# 4. Get your ACI service for inferencing, assuming this has already been set up in AML
service_name = 'my-aml-service' # replace with your actual service name
service = AciWebservice(ws, service_name)

# 5. Load and prepare the data from SQL Pool
sql_query_result = spark.sql("SELECT feature1, feature2 FROM my_table WHERE ...").toPandas()
data_to_score = sql_query_result.to_dict(orient='records')

# 6. Make the AML API call using the provided method
input_data = json.dumps({'data':data_to_score})
output = service.run(input_data = input_data)
output_df = pd.DataFrame(output.result)

# 7. Store the results in your Synapse SQL pool.
spark.createDataFrame(output_df).write.mode('append').saveAsTable("predictions_table")

print("Predictions from AML service successfully imported")
```

This snippet leverages the Azure Machine Learning SDK to interact with an already deployed web service on AML. The service is setup to take the JSON data, run the onnx model and return results. We can then ingest these results back in the Synapse SQL pool for reporting or further analysis.

**Example 3: Batch scoring on a serverless function**

A third popular approach involves leveraging Azure Functions for batch processing. It’s similar to example 1, but more tailored for processing multiple records in one go:

```python
import onnxruntime
import pandas as pd
import json
import azure.functions as func
import logging
import numpy as np

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # 1. Try to load the model from shared location
    try:
         sess = onnxruntime.InferenceSession("/path/to/my_model.onnx")  # Assuming model is local to function
    except Exception as e:
         logging.error(f"Error loading onnx model: {e}")
         return func.HttpResponse("Error loading model", status_code=500)

    # 2. Load data from the request
    try:
         req_body = req.get_json()
         data = req_body['data']
         input_df = pd.DataFrame(data)
    except Exception as e:
         logging.error(f"Error loading request data: {e}")
         return func.HttpResponse("Error loading data", status_code=400)


    # 3. Do the inference
    try:
         input_name = sess.get_inputs()[0].name # get input name from model metadata
         input_data = input_df.to_numpy().astype(np.float32)
         output = sess.run(None, {input_name: input_data})[0]
         predictions = output.tolist()
    except Exception as e:
         logging.error(f"Error running inference: {e}")
         return func.HttpResponse("Error during inference", status_code=500)

    # 4.  Prepare results for response
    result =  {"predictions": predictions}


    return func.HttpResponse(json.dumps(result),mimetype="application/json")
```

This snippet shows what a typical python based Azure Function might do to process scoring requests using an onnx model. Your Synapse Notebook would then need to call this function for every batch of input data. The important point here is that the Synapse Notebook itself doesn’t execute the model, but calls an external resource that does.

**Key Considerations and Recommendations**

The critical piece here is that Synapse SQL pools are not designed to directly run machine learning models. The methods we’ve discussed are workarounds using their architecture, not the architecture doing ML. For deeper understanding, I recommend delving into these resources:

*   **"Programming Microsoft Azure: Implementing Cloud and Serverless Solutions"** by Haishi Bai, provides insight into integrating different Azure services (like Function apps, Azure ML, and Synapse) in a consistent and scalable manner.
*   **“Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow”** by Aurélien Géron – While not directly focused on Azure, it provides solid background on model deployment strategies and understanding the mechanics of model scoring which directly impacts the approaches shown.
*   **Azure documentation on Azure Machine Learning and Azure Synapse Analytics:** The official docs provide detailed explanations and best practices on both services and their integration.
*   **Relevant ONNX documentation:** The ONNX project documentation itself explains onnx runtime and its usage. Understanding ONNX's core principles is fundamental to ensuring the model’s interoperability between different environments.

Ultimately, your chosen approach depends on your infrastructure and specific needs, such as how much data you intend to process, the latency requirements, and the level of integration complexity you are willing to manage. But in all cases, you'll be indirectly using a compute resource capable of running the onnx model outside of the Synapse SQL pool.
