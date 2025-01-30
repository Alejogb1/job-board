---
title: "Do Clarifai general models use static concept IDs?"
date: "2025-01-30"
id: "do-clarifai-general-models-use-static-concept-ids"
---
The fundamental principle underpinning Clarifai's general models is that concept IDs are not static, but rather dynamic and versioned. This is crucial for model evolution and consistency across application deployments. Specifically, while seemingly persistent from a user perspective, these IDs reference specific, timestamped versions within Clarifai's internal system, not a fixed mapping of text to an immutable identifier.

Clarifai's general models, such as those for object detection or image classification, are continuously improved and retrained. These updates inevitably lead to adjustments in the model's understanding of concepts, refinements to concept hierarchies, and even entirely new concept additions or deprecations. Assigning static concept IDs would create significant versioning and consistency challenges. Imagine, for instance, a new model better at identifying car types, leading to a more granular set of concepts; with static IDs, existing applications using the older model and its associated IDs would immediately become inconsistent, potentially even breaking down.

Instead, Clarifai employs a system where each model version maintains its own mapping between text descriptions (e.g., "cat," "dog," "car") and internal numerical representations, accessed through the API. These internal representations, not directly exposed as 'concept IDs' to the end-user, are then dynamically resolved to the user-facing concept ID via the API's model version lookup. The user-facing concept ID remains relatively stable for any given model version but will resolve to different internal representations if a different model version is selected. This separation of presentation (user concept ID) from the underlying technical identifiers used for processing allows for a seamless upgrade experience. The user can update their app to use a more recent model without needing to change their concept IDs or manage complex mappings. The core of the system relies on this versioning system.

My own experience migrating from Clarifai’s older v1 to v2 model APIs provided stark evidence of this architecture. In v1, although concept names existed, the system relied much more on implicit understanding from training and data pipelines. In v2, the concept IDs became the primary means of access but are tied to explicit versions of models. During that migration, it became apparent that using the exact same code, but merely changing the model version I was querying, would result in identical concept ID access, but different internal representations.

The way to understand the process, I found, involves thinking about concept IDs as stable aliases that point to a specific mapping within a particular model's version. These aliases are managed within Clarifai's platform, simplifying client-side application code. This means that while the concept name "cat" might always be linked to the same user-facing concept ID in a particular application, internally, the system resolves it to a different internal identifier based on the model version and underlying model training data.

Here are some code examples that highlight the dynamic mapping in practice:

**Example 1: Using the Python SDK to Access Different Model Versions**

```python
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc import service_pb2_grpc, service_pb2
from clarifai_grpc.grpc.api import resources_pb2

PAT = "YOUR_PERSONAL_ACCESS_TOKEN"
USER_ID = "YOUR_USER_ID"
APP_ID = "YOUR_APP_ID"
MODEL_ID = "general-image-recognition"

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (("authorization", "Key " + PAT),)

def get_concept_id_for_version(model_version_id):
  request = service_pb2.GetModelRequest(
      model_id=MODEL_ID,
      user_id=USER_ID,
      app_id=APP_ID,
      version_id=model_version_id,
    )

  response = stub.GetModel(request, metadata=metadata)

  for concept in response.model.model_version.concepts:
    if concept.name == "cat":
      return concept.id

  return None # Cat not found for this version

model_version_id_v1 = 'MODEL_VERSION_ID_V1' # Replace with real ID
model_version_id_v2 = 'MODEL_VERSION_ID_V2' # Replace with real ID

concept_id_v1 = get_concept_id_for_version(model_version_id_v1)
concept_id_v2 = get_concept_id_for_version(model_version_id_v2)

print(f"Concept ID for 'cat' in version {model_version_id_v1}: {concept_id_v1}")
print(f"Concept ID for 'cat' in version {model_version_id_v2}: {concept_id_v2}")


```

*   This code retrieves the concept ID associated with the concept "cat" from two different versions of the "general-image-recognition" model.
*   Note that while the user-facing concept ID might be the same across versions (this is the assumption) the underlying internal identifiers will differ. This demonstrates how using different model versions returns the same user-facing concept ID but represents different internal mapping.
*  Replace the placeholders `YOUR_PERSONAL_ACCESS_TOKEN`, `YOUR_USER_ID`, `YOUR_APP_ID`, `MODEL_VERSION_ID_V1`, and `MODEL_VERSION_ID_V2` with valid values from your Clarifai account. The `MODEL_VERSION_ID_V1` and `MODEL_VERSION_ID_V2` values would correspond to specific version identifiers for a given model.

**Example 2: Making a Prediction Request with Different Model Versions**

```python
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc import service_pb2_grpc, service_pb2
from clarifai_grpc.grpc.api import resources_pb2

PAT = "YOUR_PERSONAL_ACCESS_TOKEN"
USER_ID = "YOUR_USER_ID"
APP_ID = "YOUR_APP_ID"
MODEL_ID = "general-image-recognition"
IMAGE_URL = "URL_TO_AN_IMAGE"

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (("authorization", "Key " + PAT),)

def make_prediction(model_version_id):
    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID),
        model_id=MODEL_ID,
        version_id=model_version_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(url=IMAGE_URL)
                )
            )
        ]
    )

    response = stub.PostModelOutputs(request, metadata=metadata)

    for concept in response.outputs[0].data.concepts:
        print(f"Concept: {concept.name}, Value: {concept.value}")


model_version_id_v1 = 'MODEL_VERSION_ID_V1' # Replace with real ID
model_version_id_v2 = 'MODEL_VERSION_ID_V2' # Replace with real ID

print(f"Predictions using model version: {model_version_id_v1}")
make_prediction(model_version_id_v1)

print(f"Predictions using model version: {model_version_id_v2}")
make_prediction(model_version_id_v2)

```

* This code demonstrates the use of different model versions in making predictions. The output will show the results from each model version.
*   While the concept names might be consistent across different version outputs, the internal concept identifiers used in model processing will be different. This will result in different predictions or scores for the same concept.
*   The code will demonstrate how model outputs are tied directly to their associated versions.
* Replace the placeholders `YOUR_PERSONAL_ACCESS_TOKEN`, `YOUR_USER_ID`, `YOUR_APP_ID`, `MODEL_VERSION_ID_V1`, `MODEL_VERSION_ID_V2` and `URL_TO_AN_IMAGE` with valid values. The `MODEL_VERSION_ID_V1` and `MODEL_VERSION_ID_V2` should be different valid identifiers for the same model.

**Example 3: Using the API directly to fetch models and concept IDs**

```python
import requests
import json

PAT = "YOUR_PERSONAL_ACCESS_TOKEN"
USER_ID = "YOUR_USER_ID"
APP_ID = "YOUR_APP_ID"
MODEL_ID = "general-image-recognition"
MODEL_VERSION_ID_V1 = 'MODEL_VERSION_ID_V1'  # Replace with a real version ID
MODEL_VERSION_ID_V2 = 'MODEL_VERSION_ID_V2' # Replace with a real version ID

BASE_URL = f"https://api.clarifai.com/v2/users/{USER_ID}/apps/{APP_ID}"

headers = {
    "Authorization": f"Key {PAT}",
    "Content-Type": "application/json"
}

def fetch_model(model_version_id):
    url = f"{BASE_URL}/models/{MODEL_ID}/versions/{model_version_id}"
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    return response.json()


model_v1 = fetch_model(MODEL_VERSION_ID_V1)
model_v2 = fetch_model(MODEL_VERSION_ID_V2)

def get_concept_id_from_model(model_data, concept_name):
   for concept in model_data["model"]["model_version"]["concepts"]:
        if concept["name"] == concept_name:
            return concept["id"]
   return None

concept_id_v1 = get_concept_id_from_model(model_v1, "cat")
concept_id_v2 = get_concept_id_from_model(model_v2, "cat")



print(f"Concept ID for 'cat' in version {MODEL_VERSION_ID_V1}: {concept_id_v1}")
print(f"Concept ID for 'cat' in version {MODEL_VERSION_ID_V2}: {concept_id_v2}")
```

* This example interacts with the Clarifai API directly using HTTP requests to get model information.
* The code retrieves two different model version JSON payload.
* The `get_concept_id_from_model` function extracts the concept id for a given model version response.
* This illustrates that accessing the concept ID through a REST call is also dependent on the model version.
* Replace the placeholders `YOUR_PERSONAL_ACCESS_TOKEN`, `YOUR_USER_ID`, `YOUR_APP_ID`, `MODEL_VERSION_ID_V1`, and `MODEL_VERSION_ID_V2` with valid values. The `MODEL_VERSION_ID_V1` and `MODEL_VERSION_ID_V2` should be different valid identifiers for the same model.

For further exploration of Clarifai’s model architecture, I would recommend studying the official API documentation which is well organized and clarifies the model versioning aspects. Additionally, the tutorials available on the Clarifai website offer practical guidance and real-world examples. Examining the example notebooks often provided will also reveal how different versions interact with concept IDs in the context of specific applications. The community forums can also be an excellent resource for finding specific insights and answering questions about nuanced behaviors of models and their versioning mechanisms.
