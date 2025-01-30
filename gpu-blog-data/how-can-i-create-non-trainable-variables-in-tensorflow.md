---
title: "How can I create non-trainable variables in TensorFlow v2?"
date: "2025-01-30"
id: "how-can-i-create-non-trainable-variables-in-tensorflow"
---
TensorFlow v2's flexibility in variable management sometimes necessitates the creation of variables that remain fixed throughout the training process.  These non-trainable variables, crucial for holding parameters like pre-trained embeddings or fixed hyperparameters, are distinct from trainable variables updated during backpropagation.  My experience optimizing large-scale NLP models highlighted the critical role of efficiently managing these immutable values, avoiding unintended gradient updates and streamlining computational graphs.  This response details how to achieve this using TensorFlow v2's functionalities.


**1.  Clear Explanation:**

The core mechanism for creating non-trainable variables in TensorFlow v2 hinges on the `trainable` attribute within the `tf.Variable` constructor.  By setting this attribute to `False`, we explicitly instruct TensorFlow to exclude the variable from the optimization process. This prevents any gradient-based updates during training, ensuring the variable's value remains constant. This contrasts with the default behavior where `trainable` is implicitly set to `True`, leading to updates based on the calculated gradients during backpropagation.  This seemingly simple distinction is vital for managing model parameters that should remain fixed, enhancing both model performance and reproducibility. Furthermore, maintaining consistent values in non-trainable variables is particularly important when working with complex architectures, preventing unexpected and difficult-to-debug behaviour during training.

Beyond simply setting `trainable=False`, it is crucial to ensure that the variable's initialization occurs outside any training loop or within a separate, non-trainable scope to guarantee complete separation from the trainable parameters.  Attempting to modify a non-trainable variable during training, even indirectly, could lead to unpredictable results or errors.  Proper scoping and separation guarantee the integrity of the model's training process.

In practical terms, a non-trainable variable acts as a constant within the computational graph. It contributes to the computation but remains unaffected by gradient calculations, preserving its initial value throughout the model's training lifetime.  This is distinct from constant tensors, which are more limited in their functionality within a computational graph, particularly for more complex operations involving updates or modifications from external sources.


**2. Code Examples with Commentary:**

**Example 1: Simple Non-Trainable Scalar Variable**

```python
import tensorflow as tf

# Create a non-trainable scalar variable
non_trainable_scalar = tf.Variable(5.0, trainable=False, name="non_trainable_constant")

# Verify trainable status
print(f"Is the variable trainable? {non_trainable_scalar.trainable}")

# Observe that attempting to assign a new value won't trigger errors.
non_trainable_scalar.assign(10.0) # No error, value changes directly
print(f"Value of non-trainable scalar: {non_trainable_scalar.numpy()}")


#Attempting to access gradients will show lack of update.
with tf.GradientTape() as tape:
  loss = non_trainable_scalar * 2

grads = tape.gradient(loss, non_trainable_scalar)
print(f"Gradient of non-trainable scalar: {grads}") # Output: None
```

This example demonstrates the fundamental method: initializing a `tf.Variable` with `trainable=False`.  The output explicitly confirms its non-trainable status, and attempts to assign a new value will change the variable without error.  Crucially,  the gradient calculation shows `None`, highlighting its exclusion from the backpropagation process.

**Example 2: Non-Trainable Variable within a Model**

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding_matrix = tf.Variable(tf.random.normal([10, 5]), trainable=False, name="embedding_matrix")
        self.dense_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return self.dense_layer(embedded)

# Initialize the model
model = MyModel()

# Verify the embedding matrix's trainable status
print(f"Is the embedding matrix trainable? {model.embedding_matrix.trainable}")

# Compile and train (embedding_matrix will remain unchanged). This example is simplified for demonstration.  
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()

for i in range(10):
  with tf.GradientTape() as tape:
    outputs = model(tf.constant([0,1,2,3,4,5,6,7,8,9])) #Example input
    loss = loss_fn(outputs, tf.constant([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Here, a pre-trained word embedding matrix (`embedding_matrix`) is incorporated as a non-trainable variable within a Keras model.  This showcases how to seamlessly integrate fixed parameters into larger model architectures, maintaining their immutability during training. The training loop updates only the trainable dense layer weights, leaving the embeddings unchanged.


**Example 3:  Non-Trainable Variable for Hyperparameter Management**

```python
import tensorflow as tf

learning_rate = tf.Variable(0.001, trainable=False, name="learning_rate")

#Define the optimizer with a non-trainable learning rate.

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#Demonstrating the ability to change non-trainable hyperparameters during training
learning_rate.assign(0.0001)

#Rest of training loop would proceed using the updated learning rate.  This is an illustrative example.  
#A more realistic scenario would involve conditional changes in learning rate based on performance.
```

This illustrates the use of a non-trainable variable to manage a hyperparameter (learning rate).  Modifying the `learning_rate` variable directly updates the optimizer, demonstrating how non-trainable variables facilitate dynamic control over training aspects without affecting the trainable model parameters.

**3. Resource Recommendations:**

The official TensorFlow documentation provides comprehensive guidance on variable management.  Exploring the specifics of the `tf.Variable` constructor and the Keras model building APIs will offer more in-depth understanding.  Furthermore, studying examples of advanced model architectures that employ pre-trained embeddings or fixed hyperparameters will showcase practical applications of non-trainable variables.  A thorough understanding of automatic differentiation and backpropagation mechanisms within TensorFlow will further solidify the conceptual understanding of why and how non-trainable variables function within the broader context of model training.
