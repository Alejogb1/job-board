---
title: "What is the key-value format for TensorFlow GCP online prediction endpoints?"
date: "2025-01-30"
id: "what-is-the-key-value-format-for-tensorflow-gcp"
---
The key-value format for TensorFlow Serving online prediction endpoints on Google Cloud Platform (GCP) isn't rigidly defined by a single, universal standard.  Instead, the expected input format is determined entirely by the signature definition specified during the model's export process.  This is a crucial point often overlooked, leading to prediction failures.  My experience building and deploying numerous TensorFlow models on GCP has repeatedly highlighted the significance of this signature-driven approach.  Understanding the signature definition is paramount to correctly formatting the prediction request.

**1. Clear Explanation:**

The TensorFlow Serving REST API expects a JSON payload as input for online prediction requests.  The structure of this JSON mirrors the signature definition of your exported TensorFlow model.  The `signatures` field within the SavedModel's metagraph specifies input tensors (keys) and their corresponding expected data types and shapes (values).  Therefore, your request JSON must meticulously map to these specifications.  Failure to align the request with the model's signature will result in a server error, typically a `INVALID_ARGUMENT` error message.

Crucially, the "key" in the key-value pair corresponds to the name of the input tensor as defined in the model's signature.  The "value" is the data representing the input for that tensor, formatted according to its specified data type and shape.  For instance, if your signature defines an input tensor named "input_image" of type `INT32` and shape `[28, 28, 1]`, your JSON request must include a key "input_image" whose value is a two-dimensional array of integers matching the specified shape.  This rigorous adherence is non-negotiable for successful prediction.

In addition to input tensors, the signature might specify output tensors. While the server automatically returns the values for the output tensors based on the model's inference, correctly specifying the inputs is your responsibility as the requestor.  Incorrect input shapes, types, or missing keys will lead to failures.  The model's metagraph, accessible through tools like `saved_model_cli`, is your primary resource for deciphering this signature information.


**2. Code Examples with Commentary:**

**Example 1: Simple Classification Model**

Let's assume a simple image classification model with a single input tensor named "image" (type `float32`, shape `[28, 28, 1]`) and a single output tensor named "prediction" (type `int64`, shape `[1]`). The request would be:

```json
{
  "instances": [
    {
      "image": [
        [
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          ...
        ],
        ...
      ]
    }
  ]
}
```

**Commentary:** This example shows a single instance.  The `instances` field is a list;  you can send multiple instances in a single request for batch processing. The nested array structure mirrors the `[28, 28, 1]` shape of the "image" tensor.  Note the data type is implicitly `float32` because of the decimal values.  The actual values would depend on your pre-processing pipeline.

**Example 2: Model with Multiple Inputs**

Consider a model with two input tensors: "text" (type `string`, shape `[1]`) and "image" (type `float32`, shape `[64, 64, 3]`).  The JSON request would be:

```json
{
  "instances": [
    {
      "text": ["This is sample text."],
      "image": [
        [
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          ...
        ],
        ...
      ]
    }
  ]
}
```

**Commentary:** This demonstrates handling multiple input tensors.  The keys ("text", "image") directly correspond to the input tensor names in the model's signature.  The value for "text" is a string array, while "image" remains a multi-dimensional array representing image data. Again, the exact numerical values within are placeholders and depend on preprocessing steps.


**Example 3: Handling Variable-length Sequences (using Ragged Tensors)**

For models that handle variable-length sequences,  you might encounter ragged tensors.  Let's assume an input tensor "sentences" representing variable-length sentences, each represented as a list of word embeddings.

```json
{
  "instances": [
    {
      "sentences": [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
      ],
      "sentence_lengths": [3]
    }
  ]
}

```

**Commentary:**  This example introduces an additional input tensor, "sentence_lengths", to handle the ragged nature of the data.  This tensor tracks the length of each sentence, which allows the model to process variable-length sequences correctly. This is crucial because ragged tensors need a specific structure within the input JSON. This approach directly reflects how a model using ragged tensors would be set up for input.  Improper handling will lead to errors.



**3. Resource Recommendations:**

The TensorFlow Serving documentation, the `saved_model_cli` tool for inspecting SavedModels, and the GCP documentation on deploying TensorFlow models are indispensable resources.   Thoroughly understanding the specifics of your model's signature, accessible via the `saved_model_cli`, is the most crucial step in crafting correctly formatted prediction requests. Pay close attention to the data types and shapes declared in the signature, as even small discrepancies will prevent successful prediction.  Reviewing official TensorFlow tutorials and examples focused on deploying models to GCP will also be immensely beneficial.  Finally, diligent testing and logging of your prediction requests and responses will help to identify and resolve any format-related issues efficiently.
