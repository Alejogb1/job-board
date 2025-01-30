---
title: "How can Vertex AI batch predictions be customized to return confidence scores?"
date: "2025-01-30"
id: "how-can-vertex-ai-batch-predictions-be-customized"
---
The default output of Vertex AI batch prediction often lacks the granular confidence scores necessary for nuanced analysis. I've frequently encountered scenarios, particularly in image classification and object detection tasks, where understanding the certainty of a model's prediction is just as crucial as the prediction itself. Accessing these confidence scores directly within the batch prediction output requires specific configuration and understanding of the underlying model's capabilities.

A primary approach involves modifying the request format sent to the Vertex AI Prediction service. Instead of accepting the default response, the prediction endpoint can be configured to return detailed output, often embedded within a dedicated field in the JSON response. This relies on the model providing these confidence scores, something not universally supported by all model types and deployment configurations. In the Vertex AI framework, this process typically centers around how the instance data is formatted within the batch request and the interpretation logic employed by the deployed model itself. In essence, one is not altering the model’s fundamental output directly, but influencing how that output is structured and presented within the batch prediction results.

The model's implementation plays a critical role. Specifically, the trained model's prediction function needs to expose confidence values. For pre-built models offered by Vertex AI, the configuration is often handled automatically by passing the correct request structure. However, when dealing with a custom-trained model, such as one built with TensorFlow or PyTorch, the prediction function must be explicitly crafted to include the desired confidence scores within its output. Without this, Vertex AI will simply not expose this information, regardless of request configurations.

Let's examine this with practical examples. I've structured these around different common scenarios.

**Example 1: A TensorFlow-based Image Classification Model**

Assume I've trained a TensorFlow model for image classification. The prediction function, when receiving an image, outputs both the predicted class and the associated confidence score (probabilities from the softmax layer). In this case, the model serving logic, not the batch prediction configuration *per se*, provides the required information.

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3))
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
      x = self.base_model(x)
      x = self.pooling(x)
      return self.dense(x)


    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)])
    def serving_default(self, input_tensor):
        output_tensor = self.call(input_tensor)
        return {"predictions": output_tensor.numpy()}
```

```python
# Prediction function, designed to be served by Vertex AI custom prediction routine.
import numpy as np

def prediction_function(instances, model):
    images = np.array([tf.image.decode_jpeg(instance['image'], channels=3).numpy() for instance in instances])
    images = tf.image.resize(images, [224, 224])
    images = images / 255.0 # Normalize
    predictions = model(images)
    # Include class and confidence
    return  [{'class': np.argmax(prediction), 'confidence': np.max(prediction)}  for prediction in predictions]
```

In this example, `prediction_function` processes the batch of instances and returns a list of dictionaries. Each dictionary includes the predicted `class` index and the associated `confidence` (maximum probability across classes). This structure, when deployed on Vertex AI, will be the returned format for batch predictions. The key here is formatting the output within the prediction function to include the relevant confidence information. Within the deployed model, this data becomes accessible without any further Vertex AI specific configurations. The batch request format follows the standard Vertex AI specification for image data.

**Example 2: A Pre-trained Object Detection Model (Vertex AI Managed API)**

For pre-trained Vertex AI models, the configuration relies on setting the correct output format within the batch prediction request. For instance, consider a pre-trained object detection model. The standard response might contain bounding boxes and class labels. But, the model also internally computes confidence scores for each detected object.

The Vertex AI batch prediction API lets you request these scores via the appropriate `parameters`. The specifics will vary between models, but the approach is similar. Assume that the pre-trained model in Vertex AI expects a particular parameter such as `confidence_threshold`, as provided below.

```json
{
  "instances": [
    { "image_uri": "gs://path/to/image1.jpg" },
    { "image_uri": "gs://path/to/image2.jpg" }
  ],
   "parameters": {
     "confidence_threshold": 0.3
    }
}
```

Here, the `parameters` section signals to the Vertex AI service, and thus the underlying pre-trained model, to filter detections based on a confidence threshold. The returned output, in this case, will also contain the confidence scores directly. The response format from this API will include a “detection_scores” field, with the corresponding values. I learned this from reviewing the specific API documentation for the deployed object detection model I was using.

**Example 3: A Custom PyTorch Text Classification Model**

For a custom PyTorch model deployed on Vertex AI, the approach is similar to the TensorFlow example, where the prediction function is the critical part. The prediction function will need to process input text and return output with probability scores.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1,:,:]) # take the last hidden
        return output
```

```python
#prediction function for Vertex AI custom container
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import softmax
import json

def prediction_function(instances, model, tokenizer):
    texts = [instance['text'] for instance in instances]
    tokens = [torch.tensor(tokenizer.encode(text)) for text in texts]
    # Pad sequences
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    with torch.no_grad():
        outputs = model(tokens)
        probabilities = softmax(outputs, dim=1)
        results = [{'class': int(torch.argmax(prob)), 'confidence': float(torch.max(prob))} for prob in probabilities]

    return results
```

The Python code shows the necessary logic for batch input in a PyTorch setting. The key within the `prediction_function` is to explicitly calculate and return the confidence scores (probabilities via softmax), in addition to the class label. In this scenario, the tokenizer logic is assumed to be available. The batch input is processed, the probability is calculated, and the most probable class together with its associated probability are returned as a dictionary to Vertex AI.

**Resource Recommendations:**

For further study, I recommend reviewing the Vertex AI documentation specifically concerning the following areas:

*   **Custom Model Training:** Focus on the requirements for custom prediction routines. Pay particular attention to the formatting of output data within these functions.
*   **Batch Prediction API:** Examine the request and response structures for your chosen model type. Look for specific parameters that govern how the model outputs predictions, particularly in relation to confidence information.
*  **Pre-trained Models:** Peruse the detailed documentation for the specific pre-trained models that you will be using. Each often provides slightly varying configurations.

By understanding these core concepts – the serving function's importance in custom model scenarios and the configuration options offered by the Vertex AI batch prediction API – you should be equipped to return confidence scores along with your batch predictions, enabling much more informed decision-making.
