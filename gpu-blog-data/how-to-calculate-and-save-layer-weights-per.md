---
title: "How to calculate and save layer weights per epoch?"
date: "2025-01-30"
id: "how-to-calculate-and-save-layer-weights-per"
---
The need to extract and store layer weights per epoch is a common requirement in deep learning experimentation, enabling detailed analysis of model learning dynamics and serving as a crucial step in tasks like pruning, knowledge distillation, or even simply observing convergence patterns. I've frequently employed this technique, particularly when debugging training instability or when fine-tuning networks with very specific layer freezing strategies. A typical implementation involves leveraging callback mechanisms provided by common deep learning frameworks. The challenge lies in efficiently capturing and organizing this data for subsequent analysis.

The core process hinges on accessing model weights, typically stored as tensors, at the end of each training epoch. These weights are then serialized, often into a format suitable for storage and later retrieval. Let's break down the process by focusing on a general implementation utilizing Keras and TensorFlow, which are frameworks I've used extensively. The fundamental operations are accessing weights, organizing them per layer and epoch, and then serializing the resulting data structure.

I usually construct a custom callback for this purpose. This allows us to integrate seamlessly into the training loop without modifying the core training procedure. Let's examine the steps involved through a practical code example:

```python
import tensorflow as tf
import numpy as np
import os
import json

class WeightRecorderCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, filename_prefix='weights_epoch_'):
      super(WeightRecorderCallback, self).__init__()
      self.save_dir = save_dir
      self.filename_prefix = filename_prefix
      os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        weights_dict = {}
        for layer in self.model.layers:
            if hasattr(layer, 'get_weights'):
                layer_weights = layer.get_weights()
                if layer_weights:
                    weights_dict[layer.name] = [w.tolist() for w in layer_weights]
        filepath = os.path.join(self.save_dir, f'{self.filename_prefix}{epoch+1}.json')
        with open(filepath, 'w') as f:
             json.dump(weights_dict, f)
```

Here, the `WeightRecorderCallback` inherits from `tf.keras.callbacks.Callback`, allowing it to be passed directly to the model's `fit` method. The `__init__` method initializes the storage location and filename prefixes. The `on_epoch_end` method, triggered at the end of each epoch, iterates through each layer in the model. It utilizes a conditional statement to check if the layer supports the `get_weights` method. If so, it retrieves the layer weights, converts them to lists (necessary for JSON serialization), and stores them in a dictionary indexed by the layer name. Finally, the entire dictionary is saved as a JSON file.  This format is chosen for its easy readability and cross-platform compatibility. The file naming convention includes the epoch number, simplifying later access to each epoch's corresponding weights.

This version avoids saving every weight tensor as a separate file which, I've found, can quickly become unwieldy. Instead, the JSON format encapsulates all layer weights for a given epoch into a single, organized document.

The above example saves all accessible weights, meaning both trainable and non-trainable. If you require only the trainable weights, a modification is needed:

```python
import tensorflow as tf
import numpy as np
import os
import json

class TrainableWeightRecorderCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, filename_prefix='trainable_weights_epoch_'):
        super(TrainableWeightRecorderCallback, self).__init__()
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        weights_dict = {}
        for layer in self.model.layers:
            if hasattr(layer, 'trainable_weights'):
              trainable_weights = layer.trainable_weights
              if trainable_weights:
                weights_dict[layer.name] = [w.numpy().tolist() for w in trainable_weights]
        filepath = os.path.join(self.save_dir, f'{self.filename_prefix}{epoch+1}.json')
        with open(filepath, 'w') as f:
          json.dump(weights_dict, f)
```

This modified version introduces a key change. Instead of relying on `layer.get_weights()`, we access the `layer.trainable_weights` attribute. Further, we explicitly convert the tensor to a NumPy array using `w.numpy()` before converting to a list. This is critical for ensuring the tensor data is compatible with serialization into JSON, as TensorFlow tensors cannot be directly serialized. This approach guarantees that only trainable parameters are extracted, effectively removing non-trainable elements such as batch normalization parameters that are not updated during training.

If, during experimentation, you need to selectively record the weights of particular layers and not all, a filter can be added:

```python
import tensorflow as tf
import numpy as np
import os
import json

class SelectiveWeightRecorderCallback(tf.keras.callbacks.Callback):
  def __init__(self, save_dir, layers_to_record, filename_prefix='selected_weights_epoch_'):
    super(SelectiveWeightRecorderCallback, self).__init__()
    self.save_dir = save_dir
    self.layers_to_record = layers_to_record
    self.filename_prefix = filename_prefix
    os.makedirs(self.save_dir, exist_ok=True)


  def on_epoch_end(self, epoch, logs=None):
        weights_dict = {}
        for layer in self.model.layers:
            if layer.name in self.layers_to_record:
                if hasattr(layer, 'get_weights'):
                    layer_weights = layer.get_weights()
                    if layer_weights:
                         weights_dict[layer.name] = [w.tolist() for w in layer_weights]

        filepath = os.path.join(self.save_dir, f'{self.filename_prefix}{epoch+1}.json')
        with open(filepath, 'w') as f:
            json.dump(weights_dict, f)
```

Here, the `__init__` method now accepts a `layers_to_record` parameter, which should be a list of layer names whose weights are to be captured. The conditional statement within `on_epoch_end` filters the layers based on this list, storing only the weights from the specified layers.  This is extremely beneficial in situations where one only needs to analyze a subset of a larger model or to avoid the processing and storage overhead of weights that are not relevant to the current investigation.

For more detailed understanding of these techniques, I'd recommend exploring the TensorFlow documentation on Keras callbacks, particularly the sections on creating custom callbacks and accessing model layers. The official Keras API documentation offers an exhaustive overview of the different properties and methods available for each layer type, and how to extract weight tensors. Furthermore, the source code for TensorFlow itself offers insightful implementation details on how different layers manage their internal parameters. Finally, while I've shown JSON as the output format, alternative serialization formats like Protocol Buffers or HDF5 could be considered if you require more efficient or compact storage, at the cost of increased processing complexity. Understanding the trade-offs is crucial when you are dealing with very large models and extended training durations.
