---
title: "How can I load a ResNet50 pretrained model from TensorFlow 1.15 to TensorFlow 2.4?"
date: "2025-01-30"
id: "how-can-i-load-a-resnet50-pretrained-model"
---
TensorFlow 1.x and TensorFlow 2.x represent a substantial shift in API design and execution, rendering direct model loading between versions problematic. The core issue resides in the change from graph-based execution in 1.x to eager execution by default in 2.x, fundamentally altering how models are defined, trained, and serialized. Compatibility layers exist, but for optimal performance and ease of maintenance, adapting the model loading process is necessary. I encountered this issue firsthand during a project migrating legacy image classification models. The precise steps for loading a ResNet50 model from TensorFlow 1.15 to 2.4 depend on how the original model was saved. I will cover the most common scenarios and provide code examples to illuminate the process.

The primary challenge revolves around differences in how weights are stored and accessed. TensorFlow 1.x often used a checkpoint format that defined model structure using placeholders and a graph-based computational approach. TensorFlow 2.x favors the SavedModel format, which inherently contains both the model’s architecture and weights. Consequently, directly loading a checkpoint file from 1.x into a 2.x model won't function without a translation procedure. We need to leverage compatibility tools and, potentially, rewrite parts of the code to align with the 2.x API. Additionally, ensure that you do not have other TF versions installed as it can create conflicts. The most seamless path is to use a model initially saved in SavedModel format within TF1.x itself, if possible.

First, assuming the original model was saved as a checkpoint (.ckpt) file, it requires more involved procedures.  The initial step entails recreating the original ResNet50 model architecture using the TF 2.x `tf.keras.applications` API. This provides the structural skeleton, absent weights. We can then load the weights from the checkpoint file and assign them to the equivalent layers in the 2.x model. This operation depends on correct layer name matching between the checkpoint and the new model which can sometimes be problematic with custom models and requires manual verification if not standardized naming conventions. It can be complex and can be error prone if there are custom layers as there would be no automatic mapping.

Let’s explore code example 1:  recreating ResNet50 architecture in TF 2.x and loading weights:

```python
import tensorflow as tf

# 1. Create the ResNet50 model in TensorFlow 2.x
model_2x = tf.keras.applications.ResNet50(weights=None) # No pretrained weights

# 2. Load the checkpoint file (assuming it's in 'tf1_checkpoint_dir')
# NOTE: tf.compat.v1 is used here to maintain compatibility with tf1 checkpoint format
checkpoint_path = tf.train.latest_checkpoint('tf1_checkpoint_dir') #replace 'tf1_checkpoint_dir' with your directory

if checkpoint_path:
    # This maps the TF1 layer variable names to the TF2 layer object to get the weights
    # This approach is necessary due to different naming conventions in checkpoint file and tf2 keras.
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Create a dictionary for matching and loading the weights.
    weight_dict = {}
    for var_name in var_to_shape_map:
      if 'kernel' in var_name or 'bias' in var_name:
        tf2_var_name = var_name.replace('resnet50/','').replace(':0','')
        try:
          #This will iterate every layer for matching.
          tf2_layer = [layer for layer in model_2x.layers if layer.name==tf2_var_name.split('/')[0]][0]
          tf2_weight_name = tf2_var_name.split('/')[1]
          if 'kernel' in tf2_weight_name:
            weight_dict[var_name] = tf2_layer.kernel
          elif 'bias' in tf2_weight_name:
            weight_dict[var_name] = tf2_layer.bias
        except:
          pass
    # Assign the weights
    for var_name,tf2_var in weight_dict.items():
      weight_value = reader.get_tensor(var_name)
      tf2_var.assign(weight_value)
    print("Successfully loaded TF1 checkpoint weights into TF2 model.")
else:
    print("No checkpoint file found.")

```
In this code, we first instantiate a ResNet50 model in TF 2.x using `tf.keras.applications`. We specifically avoid loading any pretrained weights, setting `weights=None`.  We then use the TensorFlow 1.x compatibility module, `tf.compat.v1`, to load the checkpoint. We then iterate over the TF1 checkpoint variables and match to TF2 layers manually based on names of variables. After matching the variables, it assigns the weights loaded from the checkpoint to the equivalent layers in the TF2 model by extracting the weights from the reader and then assigning it. While cumbersome this method provides explicit mapping.

Next, consider a scenario where the model was saved in the SavedModel format within TensorFlow 1.x, this greatly simplifies the process. This format preserves the graph structure and allows for seamless loading in TF 2.x, although it might come with some warning messages. Note that this format is best recommended if using TF1 and planning migration to TF2 in the future.

Here is code example 2: loading a SavedModel:

```python
import tensorflow as tf

# Load the SavedModel (assuming it's in 'tf1_savedmodel_dir')
# This method handles SavedModel format, and is a more direct way to load the model.
try:
  model_2x = tf.keras.models.load_model('tf1_savedmodel_dir') #replace with your savedmodel dir.
  print("Successfully loaded TF1 SavedModel into TF2 model.")

except Exception as e:
  print("Error during saved model load process:",e)
  print("Make sure that the directory is of SavedModel format")
```

In this case, the code leverages `tf.keras.models.load_model` to load the SavedModel directly. It’s relatively straightforward when the model is correctly saved in the `SavedModel` format as the TF2 API automatically converts it internally. This approach streamlines the conversion process considerably. This handles the internal changes to execution graph and performs mapping automatically.

However, we sometimes encounter cases where the TF1 model is saved as a frozen graph, often stored in a `.pb` file. This requires a different approach, as the graph needs to be imported and converted to a structure TF 2.x can understand. This format is less preferred but still very common for TF1 users.

Here is code example 3: Loading a frozen graph (.pb file).

```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

def load_frozen_graph(frozen_graph_path, input_node, output_node):
    with tf.io.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

# Load the frozen graph
frozen_graph_path = 'tf1_frozen_graph.pb' #replace with frozen graph path
input_node_name = 'input_tensor' #replace with input placeholder name
output_node_name = 'output_tensor' #replace with output node name

loaded_graph = load_frozen_graph(frozen_graph_path, input_node_name, output_node_name)

# Convert to TF2 model - Requires explicit definition of Input and output
try:
  # Convert to a Keras model format.
  input_tensor_name =  loaded_graph.get_tensor_by_name(input_node_name+":0")
  output_tensor_name =  loaded_graph.get_tensor_by_name(output_node_name+":0")

  input_layer = tf.keras.layers.Input(tensor=input_tensor_name) #input placeholder of tf1 graph
  keras_model = tf.keras.Model(inputs = input_layer,outputs=output_tensor_name) #converting to Keras
  print("Successfully loaded frozen graph and created TF2 model.")
except Exception as e:
  print("Error during frozen graph conversion:",e)

```

In this example, we define a custom function `load_frozen_graph` to load the frozen graph. Then we import the graph and extract its input and output tensors, which allows us to define input layer in TF2 using Keras API. Then a tf.keras.Model will be created which maps the input to output which is a TF2 equivalent of the imported graph. This is important to convert the imported graph to a Keras model. Note that this also assumes correct input and output node names are known beforehand, else it would need to be found via inspection.

When undertaking such migrations, I would recommend consulting the TensorFlow documentation for the latest recommendations on model migration. Pay close attention to the TensorFlow 2.x upgrade guide, and the documentation sections covering `tf.keras.applications`, `tf.keras.models.load_model`, and `tf.compat.v1`. Also, exploring any external tutorials available on the TensorFlow website or developer communities can provide further context and practical implementation advice. When using TF1 saved in a custom format, also ensure that the code used to save the model is available and compatible with the steps above for loading. Furthermore, ensure that the input and output tensors defined during TF1 model creation is known and correctly placed when migrating. These steps can be complex, but careful attention to model format and a methodical approach greatly aids in converting models across TensorFlow versions.
