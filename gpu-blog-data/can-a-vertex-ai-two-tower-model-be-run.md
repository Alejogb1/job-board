---
title: "Can a Vertex AI two-tower model be run locally?"
date: "2025-01-30"
id: "can-a-vertex-ai-two-tower-model-be-run"
---
A two-tower model, specifically those developed and trained within Google Cloud's Vertex AI framework, presents unique challenges when considering local deployment compared to more conventional single-tower models. The core issue lies in Vertex AI's infrastructure-centric design, which deeply integrates with its managed services, making direct local execution non-trivial but not impossible with appropriate adaptation.

Vertex AI, while offering robust training and deployment capabilities, primarily operates as a Platform-as-a-Service (PaaS) solution. This means that many model operations are optimized for a cloud-based environment. The two-tower model, typically involving separate encoders for query and context, leverages this cloud infrastructure for tasks like data pre-processing, distributed training, and online serving via its custom-trained prediction endpoints. These features, although beneficial in production, introduce complexities when porting the model to a local machine. Furthermore, the exact architecture and implementation details of the two-tower model within Vertex AI may not always be completely transparent, hindering direct replication outside of the platform.

The primary barrier to direct local execution stems from the dependency on Vertex AI's pre-processing pipeline, which often involves specific data transformation steps that are executed within the cloud. Specifically, model input tensors are constructed through a combination of transformations, tokenization, and potentially feature engineering functions designed for cloud-based execution and scale. Moreover, model exports from Vertex AI often utilize a SavedModel format which, while portable in principle, assumes a TensorFlow serving environment optimized for its cloud ecosystem. The two-tower architecture itself doesn’t preclude local inference, but the specifics of Vertex AI’s implementation require a degree of manual re-implementation when shifting to a non-cloud environment. Finally, Vertex AI model deployment often leverages cloud resources like GPUs for accelerated inference, creating another layer of difficulty when deploying locally on CPU-only environments.

Here's how one would typically approach local execution: We begin by exporting the model from Vertex AI. The typical output format for our purposes is the TensorFlow SavedModel. This output includes the graph definition, weights, and some metadata. The most immediate challenge is replicating the pre-processing steps. Since the input tensors rely on Vertex AI’s pre-processing functions that are executed prior to model loading, it's necessary to re-implement these locally, ensuring consistency between training and inference data transformations. We then use TensorFlow to load the SavedModel, but often require a custom serving function for this architecture. This custom function needs to load the two towers separately and then combine their outputs in a compatible fashion, as the underlying logic for combining the tower output might be embedded inside the Vertex AI endpoint logic.

Here are three illustrative examples with commentary:

**Example 1: Loading the SavedModel and defining basic input tensors**

```python
import tensorflow as tf
import numpy as np

# Load the SavedModel (assuming it's exported from Vertex AI)
model_path = 'path/to/exported/savedmodel'
loaded_model = tf.saved_model.load(model_path)

# Define input tensors mimicking the expected format
# These would need to be determined based on the model's input signature.
# In this example, we assume text inputs for both towers.

query_input = tf.constant([['This is a sample query']], dtype=tf.string)
context_input = tf.constant([['This is a sample context']], dtype=tf.string)

# These tensors will need further pre-processing to match the model's expectation
# This example is for illustration only
```
This initial step loads the exported SavedModel. It defines placeholder input tensors. The key point here is that these input tensors must be in the exact format and type that the model expects post-pre-processing within Vertex AI. Recreating that processing locally is often done by inspecting the SavedModel signatures and data schemas used during Vertex AI training. The example assumes text inputs for illustration purposes; the actual types and shapes might vary significantly.

**Example 2: Implementing Custom pre-processing logic**

```python
import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np

# Example tokenizer, replace with actual tokenizer used in Vertex AI
# This often involves using a Vocabulary file which would be stored elsewhere.
tokenizer = tf_text.WhitespaceTokenizer()

# Example pre-processing
def preprocess_input(text_tensor):
  tokens = tokenizer.tokenize(text_tensor)
  # Padding and numericalization steps must match what was used in Vertex AI.
  tokens = tokens.to_tensor(default_value='<pad>')
  return tokens

# Example input tensors
query_input = tf.constant([['This is a sample query']], dtype=tf.string)
context_input = tf.constant([['This is a sample context']], dtype=tf.string)

# Apply pre-processing (needs to match Vertex AI)
processed_query = preprocess_input(query_input)
processed_context = preprocess_input(context_input)

# This is a simplified example, the process may be significantly more complex.
```

This example introduces a simplified version of a pre-processing function. Critically, the tokenizer and transformation logic must precisely match that used by Vertex AI during the model’s training phase. We cannot assume the input format of the raw strings. If, for example, the Vertex AI pipeline involved a BERT tokenizer, the implementation here would also need to replicate that BERT tokenizer using TensorFlow Text library or a similar tool, and then generate appropriately padded sequences. Missing this step completely invalidates the local inference. The padding and numericalization must also match the padding lengths specified during training.

**Example 3: Defining custom inference with multiple tower outputs**
```python
import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np

# Assuming pre-processing functions from example 2
# Assume that model expects the output to be a similarity score.
# In this simplified case, we use the dot product of the representations.

model_path = 'path/to/exported/savedmodel'
loaded_model = tf.saved_model.load(model_path)

# Example input tensors (processed from Example 2)
query_input = tf.constant([['This is a sample query']], dtype=tf.string)
context_input = tf.constant([['This is a sample context']], dtype=tf.string)

def preprocess_input(text_tensor):
  tokenizer = tf_text.WhitespaceTokenizer() # Replace with actual tokenizer
  tokens = tokenizer.tokenize(text_tensor)
  tokens = tokens.to_tensor(default_value='<pad>')
  return tokens

processed_query = preprocess_input(query_input)
processed_context = preprocess_input(context_input)


def predict(query_tensor, context_tensor):

    # Fetch the specific signature by name for each tower.
    # This signature name can be found by inspecting the loaded_model
    query_tower_signature = loaded_model.signatures['serving_default']
    context_tower_signature = loaded_model.signatures['serving_default']

    # The inputs to each towers must be known.
    query_embedding  = query_tower_signature(query=query_tensor)['output_0']
    context_embedding = context_tower_signature(query=context_tensor)['output_0']

    # Compute the similarity score
    similarity = tf.reduce_sum(tf.multiply(query_embedding, context_embedding), axis=1)
    return similarity

# Run inference
similarity_score = predict(processed_query, processed_context)
print(f"Similarity Score: {similarity_score.numpy()}")
```
In this example, the `predict` function explicitly retrieves each of the tower outputs by name, and then computes the similarity by taking the dot product.  We are assuming here that the specific signature for both towers are the same and use a key named `output_0`. These keys are specific to the exported savedmodel, and must be inferred by inspecting the model's signatures. This is crucial, as the tower outputs must be extracted and then combined appropriately using a custom function for inference. The method used for the computation of the similarity score would depend on how the model was trained, and is not always dot product. The key point is that it must be replicated to the letter.

The successful local execution of a Vertex AI two-tower model demands a thorough understanding of the underlying model architecture, the Vertex AI pre-processing pipeline, and a careful reproduction of the same process outside the platform. Debugging challenges are likely to arise due to the lack of complete visibility into Vertex AI’s internal operations, and iterative adjustments to the custom functions and input transformations are typically necessary to achieve results mirroring those obtained within the Vertex AI environment.

To further investigate and develop the described solution, one should consult the official TensorFlow documentation for `tf.saved_model.load` and other relevant TensorFlow functions. Exploring guides on custom training and inference workflows can be extremely beneficial for this type of problem. Reviewing any documentation on data pre-processing using `tf.data` might also prove helpful. Finally, gaining experience by exploring some of the public code repositories that demonstrate similar use cases often proves useful.
