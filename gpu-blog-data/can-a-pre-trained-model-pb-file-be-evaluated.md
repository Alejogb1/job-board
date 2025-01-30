---
title: "Can a pre-trained model (.pb file) be evaluated without additional data?"
date: "2025-01-30"
id: "can-a-pre-trained-model-pb-file-be-evaluated"
---
A pre-trained TensorFlow model, typically serialized as a `.pb` file (Protocol Buffer), can be evaluated without *additional* data if the evaluation is limited to using existing, baked-in mechanisms and the original training data or a proxy for it. However, a critical distinction must be made: *meaningful* evaluation requires data representative of the model’s intended deployment environment.

Here’s why this is nuanced. A `.pb` file represents a frozen graph: the model's architecture and the weights learned during training are encapsulated within. This file contains everything necessary to perform inference - to produce outputs given inputs. The crucial part here is “inference.” The model is now a fixed function; it will always produce the same output for a given input, regardless of whether that output is "good" or "bad" in any generalized sense.

During the model's training phase, evaluation metrics (accuracy, precision, recall, etc.) are computed on a held-out portion of the training data – the validation set. These metrics are instrumental in assessing the model's performance during training and guiding hyperparameter adjustments and architecture modifications. When training concludes, these metrics are no longer explicitly available within the `.pb` file. Therefore, using the `.pb` file alone cannot directly reproduce the *original* evaluation process.

However, we can perform a limited form of evaluation. We can feed data that was part of or very similar to the training set to the model and observe its outputs. We can also use randomly generated data that adheres to the input schema to confirm that the model executes without errors. This type of "sanity check" evaluation is possible without new external data. But, importantly, it does *not* provide insight into how the model will perform on data it has never seen before.

Consider, for example, that I've trained an image classification model on a collection of cat and dog images. The `.pb` file contains the model after it has learned to distinguish between these images. Without any further data, I can perform the following:

**Code Example 1: Basic Sanity Check with Dummy Data**

```python
import tensorflow as tf
import numpy as np

def load_graph(graph_path):
  with tf.io.gfile.GFile(graph_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name='')
  return graph

def run_inference(graph, input_tensor_name, output_tensor_name, input_data):
    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    with tf.compat.v1.Session(graph=graph) as sess:
        output = sess.run(output_tensor, feed_dict={input_tensor: input_data})
    return output


if __name__ == '__main__':
    pb_file = "my_model.pb" # Replace with actual path
    input_tensor_name = 'input_tensor:0'  # Replace with actual input tensor name
    output_tensor_name = 'output_tensor:0' # Replace with actual output tensor name

    graph = load_graph(pb_file)

    # Create dummy input (e.g., a batch of 4 images, 224x224 pixels, RGB)
    dummy_input = np.random.rand(4, 224, 224, 3).astype(np.float32)

    output = run_inference(graph, input_tensor_name, output_tensor_name, dummy_input)

    print("Output shape:", output.shape) # Print the shape of the output
    print("Inference successful. Check output manually.") # Indicates output was successfully produced.
```

**Commentary:**

This code snippet loads the `.pb` file, retrieves the input and output tensors, and then runs the model using randomly generated data as input. It verifies that the model graph is loaded successfully and produces a result. However, it does not provide any information about the model's performance in terms of classification accuracy, precision, recall etc. It only confirms that the graph runs without error when fed a dummy input tensor matching the expected shape. The message 'Inference successful. Check output manually.' indicates that manual inspection of the output tensor is required. This would typically involve verifying the data type, range of values, or overall format rather than looking for specific predictive accuracy. The assumption here is that an output tensor is produced; it is not verified to be an output that is valuable in the problem domain. The tensors themselves need to be interpreted within the specific framework of the model.

**Code Example 2: Using Existing Training Data (Assuming Access)**

```python
# Requires access to original training or validation dataset

import tensorflow as tf
import numpy as np

def load_graph(graph_path):
    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def run_inference(graph, input_tensor_name, output_tensor_name, input_data):
    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    with tf.compat.v1.Session(graph=graph) as sess:
        output = sess.run(output_tensor, feed_dict={input_tensor: input_data})
    return output

def evaluate_on_data(graph, input_tensor_name, output_tensor_name, data, labels):
    correct_predictions = 0
    for i in range(len(data)):
      single_input = np.expand_dims(data[i], axis=0)  # Assuming data is a list of images, batch one by one
      output = run_inference(graph, input_tensor_name, output_tensor_name, single_input)
      predicted_class = np.argmax(output)
      actual_class = labels[i]
      if predicted_class == actual_class:
        correct_predictions += 1
    accuracy = correct_predictions/len(data)
    return accuracy


if __name__ == '__main__':
    pb_file = "my_model.pb" # Replace with actual path
    input_tensor_name = 'input_tensor:0'  # Replace with actual input tensor name
    output_tensor_name = 'output_tensor:0' # Replace with actual output tensor name

    # Load model graph
    graph = load_graph(pb_file)

    # Replace with actual loading of training data if available, e.g.
    # data, labels = load_training_data()
    data = np.random.rand(100, 224, 224, 3).astype(np.float32)
    labels = np.random.randint(0, 2, 100) # Binary class: 0 or 1

    accuracy = evaluate_on_data(graph, input_tensor_name, output_tensor_name, data, labels)

    print(f"Accuracy: {accuracy:.4f}")

```

**Commentary:**

This example shows how you *could* evaluate the model using a subset of its original training data, *if available*.  The key here is the `evaluate_on_data` function, which iterates through the dataset, feeds each data point to the model, and compares the predicted class (obtained using `argmax` on the output tensor) to the ground truth label. An accuracy metric is then calculated. This *is* a more meaningful evaluation than simply testing if the graph runs, as we now can gain insight about the predictive performance of the model. However, if the data is not representative of the intended production environment, the accuracy obtained is not necessarily a useful measure. Note, a realistic training dataset would be more complex and would necessitate custom loading procedures.

**Code Example 3: Checking Output Sanity with a known Dataset**

```python
# Requires access to a subset of a known test dataset
import tensorflow as tf
import numpy as np

def load_graph(graph_path):
    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def run_inference(graph, input_tensor_name, output_tensor_name, input_data):
    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    with tf.compat.v1.Session(graph=graph) as sess:
        output = sess.run(output_tensor, feed_dict={input_tensor: input_data})
    return output

def inspect_outputs(graph, input_tensor_name, output_tensor_name, data):
  for i in range(min(len(data), 5)):
    single_input = np.expand_dims(data[i], axis=0)
    output = run_inference(graph, input_tensor_name, output_tensor_name, single_input)
    print(f"Output for Input {i}: {output}")


if __name__ == '__main__':
    pb_file = "my_model.pb" # Replace with actual path
    input_tensor_name = 'input_tensor:0'  # Replace with actual input tensor name
    output_tensor_name = 'output_tensor:0' # Replace with actual output tensor name

    graph = load_graph(pb_file)

    # Example with images as inputs
    data = np.random.rand(10, 224, 224, 3).astype(np.float32)

    print("Inspecting model outputs")
    inspect_outputs(graph, input_tensor_name, output_tensor_name, data)

```

**Commentary:**
This code demonstrates how a few data points can be manually inspected. Rather than computing an accuracy metric, it focuses on looking at the actual outputs produced for a small number of known inputs. If a dataset and its ground-truth expected outputs are known, this helps verify that the model is behaving as expected. The `inspect_outputs` function prints the model's output tensors directly, allowing a user to examine them and ensure they are of the expected type and range. This is valuable for basic sanity checks. The key here is inspecting the outputs given knowledge of the domain, rather than trying to extract an overall performance score.  This can be crucial when the expected outputs are known and can be checked. Note that, without knowledge of the specific problem domain, these values have no context; the usefulness of inspecting the output is derived by its relationship to the expected result.

In summary, a `.pb` model file can be evaluated in a limited capacity without additional data, specifically by running inference using dummy data or, if available, re-using training data. The key caveat is that these evaluations do not provide reliable insights into the model’s generalization performance on new data from its target environment. The model is static after training. For accurate and meaningful evaluation, new, representative data must be used.

For further exploration of model evaluation and deployment practices, I recommend the following resources:

*   Material on the TensorFlow documentation site covering model deployment and serving.
*   Books covering deep learning and model evaluation that include topics on metrics and validation strategies.
*   Tutorials and guides on best practices for model evaluation with particular attention to metrics tailored to the task and dataset.
*   Academic papers that discuss model evaluation for similar classification or regression problems.
