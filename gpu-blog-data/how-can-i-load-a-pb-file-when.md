---
title: "How can I load a .pb file when converting a PyTorch model to tf.keras?"
date: "2025-01-30"
id: "how-can-i-load-a-pb-file-when"
---
TensorFlow's `tf.compat.v1.GraphDef` format, often found in `.pb` files, represents a frozen computation graph. Direct loading of a `.pb` file into a standard `tf.keras.Model` object is not inherently supported, necessitating intermediate steps when converting from a PyTorch model. My experience suggests this conversion involves first converting the PyTorch model to an ONNX format, subsequently converting ONNX to TensorFlow's SavedModel format, and finally, potentially loading the graph definition into a custom model construct if fine-grained access to the graph’s operations is required. A direct import and utilization of a `.pb` file into Keras is atypical and requires some manipulation of TensorFlow primitives.

The typical workflow involves these stages: 1) export PyTorch to ONNX, 2) convert ONNX to a SavedModel using TensorFlow, and 3) if required, extract the `GraphDef` from the SavedModel and instantiate a model from it. The direct loading of `.pb` files arises in older TensorFlow workflows or when working with models originating from other frameworks that were exported to TensorFlow as a frozen graph. Since these are typically static graphs, there is no training associated with these.

I’ve encountered this scenario when dealing with legacy computer vision models, where the original training and deployment were done outside of Keras' more structured model definitions. In my case, a PyTorch-based image classifier had been exported to ONNX format for inference. My task involved integrating that model into a TensorFlow application. We'll approach this with three code examples.

**Example 1: Exporting a PyTorch model to ONNX**

The initial stage requires converting the PyTorch model to an ONNX model. This intermediary step ensures a standardized representation transferable between frameworks. I use `torch.onnx.export` for this. Assume `MyPyTorchModel` is your PyTorch model class. This needs to be a subclass of `torch.nn.Module`, and you will need some dummy input data.

```python
import torch
import torch.onnx

class MyPyTorchModel(torch.nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super(MyPyTorchModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def export_pytorch_to_onnx(model, dummy_input, onnx_path):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

# Example Usage
pytorch_model = MyPyTorchModel()
dummy_input = torch.randn(1, 10)
onnx_file_path = "my_model.onnx"
export_pytorch_to_onnx(pytorch_model, dummy_input, onnx_file_path)
print(f"ONNX model saved to {onnx_file_path}")
```

In this code, I defined a simple `MyPyTorchModel` class. `export_pytorch_to_onnx` uses `torch.onnx.export` to convert the PyTorch model to an ONNX file, specifying input and output names for later use and dynamic axes to handle batch sizes. The crucial part is the `opset_version`, which is sometimes relevant depending on TensorFlow's ONNX parser capabilities. The generated `.onnx` file will be used for the next step.

**Example 2: Converting ONNX to TensorFlow SavedModel and extracting GraphDef**

The `.onnx` model must be converted to a TensorFlow format. I utilize TensorFlow's ONNX importer for this conversion and extract the `GraphDef`. This conversion generates a SavedModel, a preferred structure in modern TensorFlow applications. The core functionality lies in `tf.compat.v1.import_graph_def`.

```python
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import os

def convert_onnx_to_savedmodel(onnx_path, savedmodel_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(savedmodel_path)

def load_graphdef_from_savedmodel(savedmodel_path):
  """Loads GraphDef from a SavedModel."""
  graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v1.Session() as sess:
    meta_graph_def = tf.compat.v1.saved_model.loader.load(
      sess,
      [tf.saved_model.tag_constants.SERVING],
      savedmodel_path
    )
    graph_def.CopyFrom(meta_graph_def.graph_def)
  return graph_def

# Example usage
onnx_file_path = "my_model.onnx"
saved_model_dir = "my_savedmodel"
convert_onnx_to_savedmodel(onnx_file_path, saved_model_dir)
print(f"SavedModel saved to {saved_model_dir}")

graph_def = load_graphdef_from_savedmodel(saved_model_dir)
print(f"GraphDef loaded. First node: {graph_def.node[0].name if graph_def.node else 'No nodes'}")


```

In the provided code, I first convert the `.onnx` file to a TensorFlow SavedModel using `onnx_tf.backend.prepare`. Crucially, I then implement a function, `load_graphdef_from_savedmodel`, to extract the `GraphDef` using `tf.compat.v1.saved_model.loader.load` with a session, which is necessary when loading SavedModels. The `graph_def` object now contains the frozen computational graph. The SavedModel format, exported in a specific directory, provides a more complete solution for deployment and versioning.

**Example 3: Building a Keras Model from GraphDef (advanced)**

Accessing the graph's operations requires directly working with TensorFlow's graph API, which falls outside the standard Keras model construction. This is only needed when very specific fine-tuning or access to graph operations is needed and is less common for most users who just require a keras compatible model. The following is an advanced example of wrapping a graphdef into a model like API.

```python
import tensorflow as tf
import numpy as np

class GraphDefModel(tf.keras.Model):
  def __init__(self, graph_def, input_name='input:0', output_name='output:0'):
    super(GraphDefModel, self).__init__()
    self.graph_def = graph_def
    self.input_name = input_name
    self.output_name = output_name
    self.input_tensor = None
    self.output_tensor = None
    self._load_graph()

  def _load_graph(self):
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.import_graph_def(self.graph_def, name='')
        self.input_tensor = graph.get_tensor_by_name(self.input_name)
        self.output_tensor = graph.get_tensor_by_name(self.output_name)
    self.graph = graph

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32)]) # Adapt this
  def call(self, inputs):
    with self.graph.as_default():
      return tf.compat.v1.Session().run(self.output_tensor, feed_dict={self.input_tensor:inputs})

# Example usage:
graphdef_model = GraphDefModel(graph_def)
dummy_input = np.random.rand(1, 10).astype(np.float32)
output = graphdef_model(dummy_input)
print(f"Output shape: {output.shape}")
```

Here, I define `GraphDefModel`, a custom Keras `Model` class, taking a `GraphDef` object during initialization. The `_load_graph` method imports the graph and fetches the input and output tensors by their given names.  The `call` method then uses a session to execute the graph, passing the inputs through. Importantly I explicitly specify `tf.function` along with a type and shape signature to ensure compatibility with a Keras workflow. This is crucial if you want to incorporate it into a training or inference pipelines with the Keras API. Note the `feed_dict` usage and tensor access, which are core to this approach to utilizing a raw TensorFlow `GraphDef`. This implementation is significantly more low-level than the typical Keras `Model`.

For resources beyond this, I recommend exploring the official TensorFlow documentation for `tf.compat.v1.GraphDef` and SavedModel, which provide detailed explanations of the underlying concepts. The ONNX documentation also is beneficial to understanding how different frameworks represent models and the conversion process. Additionally, TensorFlow’s tutorials on using SavedModels and the advanced topics section in Keras provide helpful information for integration. Specifically, delving into how TensorFlow handles computation graphs will significantly enhance understanding of these concepts. Lastly, the 'onnx-tensorflow' package documentation and its Github repository can be helpful to understand the steps between the `.onnx` and TensorFlow.
