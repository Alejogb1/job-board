---
title: "How can I access trained weights from a TensorFlow model?"
date: "2025-01-30"
id: "how-can-i-access-trained-weights-from-a"
---
Accessing trained weights from a TensorFlow model is a common requirement when deploying, analyzing, or fine-tuning models. In my experience, the approach varies significantly depending on how the model was constructed – whether using the Keras API or lower-level TensorFlow operations – and what format those weights were initially saved in. It’s crucial to understand that weights, at their core, are numerical tensors representing the learned parameters of the model's layers, and accessing them involves interacting with TensorFlow's internal representation.

Primarily, TensorFlow provides two methods for saving and restoring weights: checkpoint files (.ckpt) and the SavedModel format. Checkpoint files are TensorFlow's native format, primarily designed for saving and restoring weights during training. They contain variable data, allowing for resuming training from a particular point. SavedModel, on the other hand, is a more comprehensive format designed for production deployment and includes not just the weights but also the computational graph, input signatures, and other relevant information. This distinction is vital because accessing weights directly differs based on the format. I’ll focus primarily on accessing the raw weight tensors, rather than re-loading the weights into an existing graph for inference.

**1. Accessing Weights from Keras Models**

When using the Keras API, weight access is streamlined. Keras models store weights within individual layers. You can retrieve these weights by iterating through the layers and accessing their `weights` attribute. This attribute returns a list of tensors; for most dense, convolutional, and recurrent layers, there are generally two tensors: the kernel weights (the connection weights) and the bias. Let's demonstrate.

```python
import tensorflow as tf

# Example Keras model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dense(5, activation='softmax')
])

# Build the model so weights are instantiated
model.build()

# Iterate through the layers and extract weights
for layer in model.layers:
  print(f"Layer name: {layer.name}")
  if layer.weights: # Only layers with trainable weights have weights
    for i, weight_tensor in enumerate(layer.weights):
       print(f"  Weight type {i}: {weight_tensor.name} , shape {weight_tensor.shape}")

```
*Commentary:*

In this example, we constructed a very simple sequential model. We then iterate through each layer using `model.layers`. A check for `layer.weights` is crucial; layers such as dropout, pooling, and input layers generally do not hold trainable weights. For layers that do, we loop through `layer.weights`. Each of these `weight_tensor` objects is a TensorFlow Tensor, holding the numerical data, which you could, for example, convert to NumPy using `.numpy()` for other analysis. We can also inspect each weight's shape and name for precise identification.

**2. Accessing Weights from a Checkpoint File**

If the model weights have been saved to a checkpoint file, you’ll need to use `tf.train.load_checkpoint` and navigate the tensor names. Checkpoints use a custom file format with an associated index. To access individual tensors, you must use the precise tensor name used when training, and restore those specific tensors using `tf.train.load_checkpoint`. The problem is, we don't know the names of the tensors in advance so a utility function is required.

```python
import tensorflow as tf

def list_checkpoint_variables(checkpoint_path):
    """Lists variable names from a given TensorFlow checkpoint."""
    reader = tf.train.load_checkpoint(checkpoint_path)
    return reader.get_variable_to_shape_map().keys()

# Create a dummy directory
checkpoint_dir = "tmp_checkpoint_dir"
if not tf.io.gfile.exists(checkpoint_dir):
    tf.io.gfile.makedirs(checkpoint_dir)

# Create and save some dummy variables to checkpoint file
v1 = tf.Variable(tf.random.normal(shape=(10,10)), name="v1")
v2 = tf.Variable(tf.random.normal(shape=(5,10)), name="layer1/dense_kernel")

checkpoint = tf.train.Checkpoint(v1=v1, v2=v2)
checkpoint.save(checkpoint_dir + "/my_model")

# List names of tensors in checkpoint
checkpoint_path =  checkpoint_dir + "/my_model"
variable_names = list_checkpoint_variables(checkpoint_path)
print("Checkpoint variable names:", variable_names)

# Restoring the variables
reader = tf.train.load_checkpoint(checkpoint_path)

for name in variable_names:
    tensor = reader.get_tensor(name)
    print(f"Variable name: {name}, shape: {tensor.shape}")


```
*Commentary:*

This example presents the more involved process of checkpoint restoration. First, a utility function, `list_checkpoint_variables`, uses `tf.train.load_checkpoint` and `get_variable_to_shape_map()` to collect variable names in the checkpoint file. We create dummy variables (v1 and v2) and save them using a `tf.train.Checkpoint` object to emulate a model save. We list all of the names using the utility function, then we use `reader.get_tensor(name)` to load a specific weight tensor from the checkpoint given its name. Accessing a specific weight relies entirely on the knowledge of the internal variable names that TensorFlow assigned during training. This is less convenient for model exploration compared to Keras, but it provides low level access for debugging.

**3. Accessing Weights from a SavedModel**

The SavedModel format is comprehensive and typically used for deployment. While it includes the computational graph and input/output signatures, you can still obtain raw weight tensors, but it's often more involved. You need to load the SavedModel and navigate through the concrete functions and variables. This is typically done not for weight extraction but for inferencing.

```python
import tensorflow as tf

# Build the simple Keras model we used in example 1
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dense(5, activation='softmax')
])

model.build()

# Save to SavedModel format
save_path = "saved_model"
tf.saved_model.save(model, save_path)


# Load the SavedModel
loaded_model = tf.saved_model.load(save_path)
concrete_function = loaded_model.signatures["serving_default"]

# Access the variables, usually not exposed directly
for var in concrete_function.variables:
    print(f"Variable name: {var.name}, shape {var.shape}")

```
*Commentary:*

This example creates a SavedModel from a trained Keras model, then loads the SavedModel. You'll notice that the "variables" are not the direct weight tensor objects like we had in the Keras example, and there's an added layer of abstraction due to the concrete function. To actually get the weights you have to loop through the variables of `concrete_function.variables`. You typically don't load a SavedModel in this manner to retrieve the weights but rather to load up a model for inferencing, so this is an unconventional extraction. This way of accessing weights from SavedModel format is generally less convenient than checkpoint files or direct access from Keras objects, but may be used for model introspection and low level analysis.

**Resource Recommendations**

To further explore this topic, consider consulting the following resources:

*   The official TensorFlow documentation on the Keras API provides comprehensive details on model creation, saving, and weight manipulation. Pay attention to the sections discussing layers and `tf.keras.Model` objects.

*   The TensorFlow guide on checkpointing provides information on how TensorFlow models are saved and restored in `tf.train.Checkpoint` format. Understanding variable tracking and restoration in this context is crucial.

*   The SavedModel documentation, again from TensorFlow, describes the intricacies of the SavedModel format, how it encodes models and functions, and how it interacts with specific TensorFlow concepts such as variable management, concrete functions, and signatures.

*   Tutorials on TensorFlow's website and other platforms which describe weight inspection, often from the standpoint of visualizing them or performing model surgery. These tutorials can offer practical perspectives on the use of the techniques described.

In summary, accessing trained weights from TensorFlow models requires a contextual approach. Keras models offer convenient access through their layer structure. Checkpoint files require knowledge of the internal naming convention. SavedModel typically doesn't lead to direct weight extraction, but it can be done using the specific `concrete_function` and variables. A deep understanding of TensorFlow's internal representation of weights, its saving methods, and the specific APIs available to you depending on how your model was created, is crucial for successful weight extraction.
