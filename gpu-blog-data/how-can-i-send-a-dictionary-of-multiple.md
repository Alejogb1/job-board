---
title: "How can I send a dictionary of multiple inputs to a TensorFlow Serving model via the REST API?"
date: "2025-01-30"
id: "how-can-i-send-a-dictionary-of-multiple"
---
A critical aspect of deploying machine learning models with TensorFlow Serving is accurately formatting input data for the REST API, particularly when dealing with models that expect a dictionary of tensors as input.  I encountered this challenge firsthand while developing a multi-modal recommendation system, where the model required user features, item features, and context data to make predictions. The standard approach of sending a single JSON object often falls short when the input is structured as a dictionary, leading to errors and incorrect model inference.

The core issue lies in the way TensorFlow Serving interprets incoming JSON payloads. When a model's signature specifies multiple inputs as a dictionary, the REST endpoint expects a specific structure within the JSON, mirroring this dictionary. Simply sending a JSON object with keys that match the input names of the model's signature isn’t sufficient. Instead, the payload must explicitly represent each input as a distinct tensor, even if the input is not a single numerical array. TensorFlow Serving expects each input value to be nested within a “instances” key that holds an array of prediction instances. For dictionary inputs, each instance, instead of a single entry, is expected to be a dictionary itself, whose keys match the names of model input keys. This nested structure ensures that the correct type information is associated with the input data during deserialization at the serving side.

To be more precise, the JSON payload for a model accepting a dictionary of inputs, such as `{"user_features": tensor_a, "item_features": tensor_b, "context_data": tensor_c}`, must be structured as follows:
```json
{
  "instances": [
    {
      "user_features": { "b64": "base64_encoded_tensor_a_data" , "dtype": "DATA_TYPE"},
      "item_features": { "b64": "base64_encoded_tensor_b_data", "dtype": "DATA_TYPE"},
      "context_data": { "b64": "base64_encoded_tensor_c_data", "dtype": "DATA_TYPE"}
    }
   ]
}
```
Here, each key in the inner dictionary corresponds to the input name, and the associated value is *another* dictionary containing the tensor's base64 encoded binary data (`b64`) and data type (`dtype`). It's crucial to understand that the `dtype` string represents one of TensorFlow’s defined datatypes.

Let's examine three specific code examples demonstrating how to prepare such a payload, with varying tensor complexities:

**Example 1: Simple Numerical Data**

Suppose a model takes two scalar inputs, “age” and “income”, as a dictionary. The corresponding input tensors would have shape `[]`.

```python
import json
import base64
import numpy as np

def prepare_simple_input(age, income):
    input_data = {
        "instances": [
            {
              "age": {
                  "b64": base64.b64encode(np.array(age, dtype=np.float32).tobytes()).decode(),
                  "dtype": "DT_FLOAT"
              },
              "income": {
                  "b64": base64.b64encode(np.array(income, dtype=np.float32).tobytes()).decode(),
                  "dtype": "DT_FLOAT"
              }
            }
        ]
    }
    return json.dumps(input_data)

age = 35.0
income = 75000.0

payload = prepare_simple_input(age, income)
print(payload)
```

This example demonstrates the fundamental process.  First, we convert numerical data into NumPy arrays with the correct datatype (in this case `float32`). Then, we serialize the NumPy array into bytes and encode the bytes into base64. Finally, we construct the dictionary with the “b64” and `dtype` keys and build up to the “instances” JSON structure.  The output would be valid json, suitable for sending via HTTP post request to the `/v1/models/{model_name}:predict` endpoint. Note that if multiple predictions are required, each needs to be added to the instances array.

**Example 2: One Dimensional Array Data**

Now consider a scenario where the input is a dictionary, with one key being numerical, and the other key is a one dimensional sequence of numbers. For example, the input signature might be `{"user_id": id_number, "last_5_viewed_items": item_ids_array}`.

```python
import json
import base64
import numpy as np

def prepare_sequence_input(user_id, last_5_viewed_items):
    input_data = {
        "instances": [
            {
                "user_id": {
                    "b64": base64.b64encode(np.array(user_id, dtype=np.int64).tobytes()).decode(),
                    "dtype": "DT_INT64"
                  },
                "last_5_viewed_items": {
                   "b64": base64.b64encode(np.array(last_5_viewed_items, dtype=np.int32).tobytes()).decode(),
                   "dtype": "DT_INT32"
                }
            }
        ]
    }
    return json.dumps(input_data)

user_id = 12345
last_5_viewed_items = [100, 105, 203, 305, 102]

payload = prepare_sequence_input(user_id, last_5_viewed_items)
print(payload)
```

This example shows how to correctly encode a one dimensional NumPy array, `last_5_viewed_items`, alongside a singular id. Again, we are using appropriate datatype specifiers (`DT_INT32`, `DT_INT64`). The base64 encoded form of the array is crucial for sending non-textual data over HTTP. Again, note that this structure will also be compatible if multiple instances of the input data are required.

**Example 3: String Array Data**

Finally, let's consider handling string data, especially when dealing with natural language processing models. Here, we’ll assume the input signature is `{"query": query_string, "user_location": location_string}`, where strings need to be represented as bytes and encoded into base64.

```python
import json
import base64
import numpy as np

def prepare_string_input(query, location):
    input_data = {
        "instances": [
            {
                "query": {
                    "b64": base64.b64encode(query.encode('utf-8')).decode(),
                    "dtype": "DT_STRING"
                  },
                  "user_location": {
                    "b64": base64.b64encode(location.encode('utf-8')).decode(),
                    "dtype": "DT_STRING"
                }
            }
        ]
    }
    return json.dumps(input_data)


query = "What is the capital of France?"
location = "Paris"

payload = prepare_string_input(query, location)
print(payload)
```

In this case, we must first convert our unicode strings into a bytes using `encode('utf-8')`. There's no conversion to NumPy arrays needed, but the byte sequence needs to be converted to base64 for transport over HTTP. The data type is specified as `DT_STRING`. The TensorFlow Serving model will be able to decode these input strings and continue its inference. Again, note that if multiple string inputs are required they can all be added to the `instances` array.

These examples highlight the need for specific encoding and structure. Incorrectly formatted data will result in errors from the TensorFlow Serving server. The keys within the innermost dictionary have to match the input keys in the models signature.  Similarly, the data type of the tensor has to match the data type in the models signature.

For further exploration of this topic, I would recommend consulting the official TensorFlow Serving documentation, which provides detailed descriptions of the REST API, input formatting requirements, and supported data types. The TensorFlow Python API documentation also covers tensor serialization. Resources covering JSON format and base64 encoding are also beneficial for a deeper understanding of the encoding process. Specifically, I found the sections on model signature specification, REST API structure, and tensor input encoding most pertinent to resolving issues related to input dictionaries. Finally, experimentation with dummy models and careful validation against the expected model input signatures is crucial for robust integration with TF Serving.
