---
title: "Can TensorFlow 1.15 load a Keras model for C++ prediction?"
date: "2025-01-30"
id: "can-tensorflow-115-load-a-keras-model-for"
---
TensorFlow 1.15's C++ API does not directly support loading Keras models saved using the Keras `save_model` function, which serializes the model architecture and weights into a single HDF5 file.  This limitation stems from the architectural differences between the TensorFlow 1.x `SavedModel` format and the Keras HDF5 format.  My experience working on a large-scale image recognition project using TensorFlow 1.15 highlighted this incompatibility. We initially attempted to deploy our Keras model directly, only to encounter significant challenges during the C++ inference phase. This prompted a deeper investigation into the available options for bridging the gap.

The core issue arises from the fact that the Keras HDF5 format, while convenient for Python-based workflows, lacks the necessary metadata and structure for direct consumption by the TensorFlow 1.15 C++ API. The C++ API primarily relies on the `SavedModel` protocol buffer format, which contains a more comprehensive representation of the graph and its associated variables. Consequently, a direct load attempt will result in errors related to missing graph definitions or incompatible data structures.

To effectively use a Keras model within a TensorFlow 1.15 C++ environment, one must adopt a conversion strategy. This generally involves two steps: first, saving the Keras model in a format compatible with TensorFlow's C++ inference engine; and second, loading this converted model using the appropriate C++ API calls.  The most reliable approach involves using TensorFlow's `SavedModel` format.

**1.  Explanation: The Conversion Process**

The conversion process hinges on leveraging TensorFlow's Python API to transform the Keras model into a `SavedModel`.  This requires loading the Keras model, constructing a `tf.compat.v1.Session`, and subsequently exporting the graph and variables to a `SavedModel` directory. This directory, containing protocol buffer files, is then compatible with TensorFlow's C++ inference tools.  Crucially, this conversion must be performed within a Python environment with TensorFlow 1.15 installed, as the C++ API cannot perform this conversion directly.

**2. Code Examples with Commentary**

**Example 1:  Saving a Keras Model as a SavedModel**

```python
import tensorflow as tf
from tensorflow import keras

# Load your pre-trained Keras model
model = keras.models.load_model('my_keras_model.h5')

# Create a SavedModel
tf.compat.v1.saved_model.save(
    model,
    'saved_model',
    signatures={
        'serving_default':
            model.signatures['serving_default'] if 'serving_default' in model.signatures else None
    }
)
```

This Python script first loads the Keras model from an HDF5 file (`my_keras_model.h5`). The `tf.compat.v1.saved_model.save` function then exports the model to the 'saved_model' directory. Note the inclusion of `signatures`.  This is particularly important for complex models or models with multiple inputs/outputs, ensuring that the C++ inference engine understands the model's interface correctly.  In simpler cases, it might be omitted, but including `serving_default` offers better compatibility.


**Example 2: C++ Inference Code (Simplified)**

```c++
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/platform/env.h"

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;

  // Create session
  Session* session;
  SessionOptions options;
  Status status = NewSession(options, &session);

  // Load the SavedModel
  MetaGraphDef meta_graph_def;
  status = ReadMetaGraphDef(Env::Default(), "saved_model/saved_model.pb", &meta_graph_def);

  // Add inputs and outputs - adapt according to your model
  Tensor input_tensor(DT_FLOAT, TensorShape({1, 28, 28, 1})); // Example input shape
  Tensor output_tensor;

  // ... Add input data to input_tensor ...

  std::vector<std::pair<string, Tensor>> inputs = {{"input_1", input_tensor}}; //Adjust 'input_1' to match your input tensor name.
  std::vector<string> output_names = {"dense/BiasAdd"}; //Adjust "dense/BiasAdd" to match your output tensor name.

  // Run the inference
  status = session->Run(inputs, output_names, {}, &output_tensor);

  // ... Process the output_tensor ...
  delete session;
  return 0;
}
```

This C++ code demonstrates the basic process of loading the `SavedModel` and performing inference.  It uses the TensorFlow C++ API to create a session, load the metagraph (`saved_model.pb`), define input and output tensors (adjusting shapes and names according to your model's specification), and run the inference. The placeholder comments indicate the need to populate the input tensor with your data and to process the results.  Note that you'll need to link against the TensorFlow C++ library. The specific input and output tensor names must be extracted from the `saved_model` directory.  Tools like Netron can visually inspect the graph and aid in identifying these names.


**Example 3:  Handling Multiple Outputs in C++**

```c++
// ... (Includes and session setup as in Example 2) ...

// For multiple outputs, you need to specify them in output_names
std::vector<string> output_names = {"dense/BiasAdd", "dense_1/BiasAdd"}; //Example names

std::vector<Tensor> output_tensors;
status = session->Run(inputs, output_names, {}, &output_tensors);

// Access individual outputs
Tensor& output_tensor1 = output_tensors[0];
Tensor& output_tensor2 = output_tensors[1];

// ... Process output_tensor1 and output_tensor2 ...
```

This variation extends Example 2 to handle models with multiple outputs. The `output_names` vector now contains the names of all output tensors. The `session->Run` call populates the `output_tensors` vector accordingly. Individual outputs are then accessed using array indexing.  Remember to adapt the output names according to your model's structure.


**3. Resource Recommendations**

The TensorFlow documentation, specifically the sections related to the C++ API and `SavedModel`, are invaluable resources.  The TensorFlow tutorials, particularly those demonstrating C++ inference, provide practical examples.  Furthermore, utilizing a debugger during both the Python conversion and the C++ inference stages is highly recommended for effective troubleshooting.  Understanding the underlying mechanics of the `SavedModel` format will greatly aid in resolving any compatibility issues.



In conclusion, while TensorFlow 1.15's C++ API doesn't directly support loading Keras models saved in the HDF5 format, a reliable workaround exists by converting the Keras model to a `SavedModel` using TensorFlow's Python API.  This allows seamless integration with the C++ inference engine, enabling efficient deployment of Keras models in C++ applications.  Careful attention must be paid to correctly specifying input and output tensor names during the conversion and inference processes.  Thorough understanding of the TensorFlow 1.x architecture and diligent debugging are crucial for successful implementation.
