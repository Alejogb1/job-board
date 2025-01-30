---
title: "How can TensorFlow restore only specific variables from a checkpoint?"
date: "2025-01-30"
id: "how-can-tensorflow-restore-only-specific-variables-from"
---
TensorFlow’s checkpoint mechanism, while crucial for saving and restoring model states, defaults to restoring *all* variables when using functions like `tf.train.Checkpoint.restore`. However, selective restoration, focusing on only certain variables, is a common requirement in scenarios like transfer learning, adversarial training, or fine-tuning pre-trained models. I've encountered this need several times during development, especially when transitioning between architectures or leveraging pre-existing weights. The solution hinges on leveraging `tf.train.Checkpoint`’s internal structure and employing explicit variable selection.

The standard `tf.train.Checkpoint.restore` method, when called, attempts to restore all variables present in the provided checkpoint file to variables within the current checkpoint object. If variables are missing or have different shapes, TensorFlow throws errors or issues warnings. This behavior is problematic when only a subset of a checkpoint's data is needed. The key insight is that checkpoint files store not just variable values, but also the *names* of those variables. These names are integral for mapping saved data to the appropriate tensors in the current TensorFlow graph. My approach to achieving selective restoration relies on manipulating this mapping process.

Instead of blindly restoring everything, I first construct a new, *partial* `tf.train.Checkpoint` object that contains only the variables I want to restore. I then restore the checkpoint into this new object, effectively filtering the restoration process. This approach avoids the errors associated with shape mismatches or missing variables, allowing for highly controlled weight transfer and loading.

**Example 1: Restoring Weights of Shared Layers**

Consider a situation where two models, `model_A` and `model_B`, share a convolutional base but have different classification heads. Model A has been trained, and we wish to initialize Model B’s base using those trained weights, while leaving Model B's head initialized randomly. This situation is commonly faced in transfer learning scenarios, and careful variable selection is essential.

```python
import tensorflow as tf

# Assume model_A and model_B are defined with a shared base
# For simplicity, let's use placeholders
class Model(tf.keras.Model):
    def __init__(self, base_layers, head_layers):
        super(Model, self).__init__()
        self.base_layers = base_layers
        self.head_layers = head_layers

    def call(self, inputs):
        x = inputs
        for layer in self.base_layers:
            x = layer(x)
        for layer in self.head_layers:
            x = layer(x)
        return x


# Create the shared base layers and model A.
shared_base = [tf.keras.layers.Conv2D(32, 3), tf.keras.layers.MaxPool2D()]
head_A = [tf.keras.layers.Dense(10)]
model_A = Model(shared_base, head_A)

# Create model B with same shared layers but a different head
head_B = [tf.keras.layers.Dense(5)]
model_B = Model(shared_base, head_B)

# Create a checkpoint for model A
ckpt_A = tf.train.Checkpoint(model=model_A)

# Simulate training and saving checkpoint (we initialize weights here for illustration)
_ = model_A(tf.random.normal((1,28,28,3)))
ckpt_A.save("ckpt_model_A")

# Build the checkpoint for model B to restore the shared base layers
# This can be done either as individual tensors or through layers, depending on how they are initially constructed.
restorable_vars = []
for layer in model_B.base_layers:
    for var in layer.trainable_variables:
        restorable_vars.append(var)


# Filterable checkpoint based on trainable variables of the shared base
partial_ckpt_B = tf.train.Checkpoint(vars_to_restore = restorable_vars)
partial_ckpt_B.restore(tf.train.latest_checkpoint("ckpt_model_A"))
```

In this example, I explicitly constructed a `restorable_vars` list containing only the trainable variables of model B’s shared base layers. A partial checkpoint `partial_ckpt_B` is then created to restore data from the model A checkpoint. This ensures that only the convolutional base weights are loaded, leaving model B's classification head with its original random initialization. This approach avoids errors due to the differing output layer dimensions in the two models, allowing for a straightforward implementation of transfer learning.

**Example 2: Restoring Specific Variables by Name**

Another common scenario involves models where layers are organized with consistent naming conventions, making it possible to selectively restore weights based on substring matches in their names. For instance, one might want to freeze the weights of a particular embedding layer but update the rest of the model.

```python
import tensorflow as tf

class NamedModel(tf.keras.Model):
    def __init__(self):
        super(NamedModel, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(100, 10, name='embedding_layer')
        self.dense_layer_1 = tf.keras.layers.Dense(64, name='dense_layer_1')
        self.dense_layer_2 = tf.keras.layers.Dense(10, name='dense_layer_2')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        return x

model_named = NamedModel()
ckpt_named = tf.train.Checkpoint(model=model_named)

# Simulate training and saving checkpoint
_ = model_named(tf.random.uniform((1, 10), minval=0, maxval=99, dtype=tf.int32))
ckpt_named.save('ckpt_named_model')

# Find variables not part of the embedding layer using string filtering on variable names
vars_to_restore = []
for var in model_named.trainable_variables:
    if 'embedding_layer' not in var.name:
        vars_to_restore.append(var)


partial_ckpt_named = tf.train.Checkpoint(vars_to_restore=vars_to_restore)
partial_ckpt_named.restore(tf.train.latest_checkpoint("ckpt_named_model"))

```

Here, the model's trainable variables are filtered using string matching on their names. This approach is helpful when a naming convention is consistent across models and specific layer types are targeted. This selective restoration ensures that only variables other than those belonging to the embedding layer are restored from the saved checkpoint.

**Example 3: Using Dictionaries to Map Variables Explicitly**

For complex situations where variable names or organization may not be consistent, I use dictionaries to create explicit mappings between checkpoint variable names and the current model’s variables. This method is the most precise and allows me to handle variations in model structure or variable names that have changed over time. This provides the greatest flexibility, though comes with the added burden of ensuring variable correctness through manual mapping.

```python
import tensorflow as tf

class VariableMappedModel(tf.keras.Model):
  def __init__(self):
    super(VariableMappedModel, self).__init__()
    self.dense1_old = tf.keras.layers.Dense(64, name='old_dense1')
    self.dense2 = tf.keras.layers.Dense(10, name='dense2')


  def call(self, inputs):
    x = self.dense1_old(inputs)
    x = self.dense2(x)
    return x



model_mapped_old = VariableMappedModel()
ckpt_mapped_old = tf.train.Checkpoint(model=model_mapped_old)

# Simulate training and saving checkpoint
_ = model_mapped_old(tf.random.normal((1,10)))
ckpt_mapped_old.save("ckpt_mapped_old_model")

# Create a new model that slightly differs from the old model
class VariableMappedModelNew(tf.keras.Model):
  def __init__(self):
    super(VariableMappedModelNew, self).__init__()
    self.dense1_new = tf.keras.layers.Dense(64, name='new_dense1') # Different name
    self.dense2 = tf.keras.layers.Dense(10, name='dense2')


  def call(self, inputs):
    x = self.dense1_new(inputs)
    x = self.dense2(x)
    return x


model_mapped_new = VariableMappedModelNew()

# Construct the variable mapping dictionary
var_mapping = {}
for var_old in model_mapped_old.trainable_variables:
    if var_old.name == 'old_dense1/kernel:0':
        var_mapping[var_old.name] = model_mapped_new.dense1_new.kernel
    if var_old.name == 'old_dense1/bias:0':
        var_mapping[var_old.name] = model_mapped_new.dense1_new.bias
    if var_old.name == 'dense2/kernel:0':
        var_mapping[var_old.name] = model_mapped_new.dense2.kernel
    if var_old.name == 'dense2/bias:0':
        var_mapping[var_old.name] = model_mapped_new.dense2.bias

partial_ckpt_mapped = tf.train.Checkpoint(**var_mapping)
partial_ckpt_mapped.restore(tf.train.latest_checkpoint('ckpt_mapped_old_model'))
```

This example demonstrates the use of a dictionary to map variable names from the old model to the corresponding variables in the new model. This approach is essential when model structures have evolved or variables have been renamed between saved checkpoints and the current model. This allows for precisely controlled weight transfer even when the underlying structures of the saved checkpoint and the present model are dissimilar.

**Resource Recommendations**

For deeper understanding of model saving and restoring techniques, I strongly recommend reviewing the following resources:

*   **TensorFlow Core Documentation:** The official TensorFlow documentation provides comprehensive details on the `tf.train.Checkpoint` API, outlining all its functionalities, including those used in selective restoration. Studying the examples and API documentation is key.
*   **Advanced TensorFlow Tutorials:** Many online platforms and university courses offer resources on transfer learning and fine-tuning, which demonstrate the practical applications of checkpoint management and highlight the necessity of partial loading for effective model reutilization.
*   **TensorFlow Community Forums:** Engaging with the TensorFlow community provides opportunities to explore diverse approaches for model management, learn from other practitioners' experiences, and understand various pitfalls one might encounter while working with checkpoints.

In summary, restoring only specific variables from a TensorFlow checkpoint requires a more strategic approach than the default restoration. I have found that utilizing partial `tf.train.Checkpoint` objects with either explicit variable lists, variable name filtering, or explicit mapping dictionaries are effective and highly versatile solutions. These techniques offer the necessary control for managing checkpoint data when transfer learning, fine-tuning, or working with evolving model architectures, thereby ensuring robustness and flexibility when training and deploying TensorFlow models.
