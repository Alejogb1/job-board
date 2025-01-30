---
title: "How can TensorFlow 2 graphs be saved and loaded?"
date: "2025-01-30"
id: "how-can-tensorflow-2-graphs-be-saved-and"
---
TensorFlow 2's shift towards eager execution significantly altered the paradigm of graph construction and saving compared to TensorFlow 1.x.  The key fact to understand is that while TensorFlow 2 retains the ability to define and execute graphs, it primarily operates in eager mode by default. This necessitates a different approach to saving and loading computational graphs than the methods employed in its predecessor.  My experience optimizing large-scale image recognition models highlighted the crucial differences and best practices in this area.


**1.  Understanding TensorFlow 2's Saving Mechanism**

TensorFlow 2 employs the `tf.saved_model` API as the primary mechanism for saving and restoring models. This API transcends the limitations of saving only the graph structure.  Instead, it saves the entire model's state, including:

* **The model's architecture:** The structure of the layers, including their types, parameters, and connections.
* **The model's weights and biases:** The learned parameters that define the model's behavior.
* **The optimizer's state:** The internal state of the optimizer used during training (e.g., momentum, learning rate).
* **Custom objects:**  Any custom layers, functions, or classes used in the model.  This is particularly important for reproducibility and deployment.

This comprehensive approach ensures that a saved model can be readily restored and used for inference, further training, or any other downstream task without requiring the original code to be present.  This differs significantly from the older `tf.train.Saver` approach which primarily saved weights and often lacked the ability to reliably restore custom elements.


**2. Code Examples and Commentary**

Let's illustrate the saving and loading process with three distinct examples, progressively increasing in complexity.

**Example 1: Saving a Simple Sequential Model**

```python
import tensorflow as tf

# Define a simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model (necessary for saving the optimizer state)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save the model to a SavedModel directory
model.save('saved_model/simple_model')

# Load the saved model
restored_model = tf.keras.models.load_model('saved_model/simple_model')

# Verify that the loaded model is identical (optional)
print(model.summary())
print(restored_model.summary())
```

This example showcases the basic process.  The `model.save()` function conveniently handles the creation of the `SavedModel` directory.  The `load_model` function automatically handles loading the architecture, weights, and optimizer state.  Note the compilation step; it's essential for saving the optimizer state if you intend to resume training.


**Example 2: Saving a Model with a Custom Layer**

```python
import tensorflow as tf

class MyCustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyCustomLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(64, activation='relu')

    def call(self, inputs):
        return self.dense(inputs)

model = tf.keras.Sequential([
    MyCustomLayer(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.save('saved_model/custom_layer_model')

restored_model = tf.keras.models.load_model('saved_model/custom_layer_model', custom_objects={'MyCustomLayer': MyCustomLayer})

print(model.summary())
print(restored_model.summary())
```

This example highlights the importance of `custom_objects`.  Because `MyCustomLayer` is not a standard Keras layer, we must explicitly provide it during loading using the `custom_objects` argument.  Failure to do so will result in an error.  This emphasizes the robustness of `tf.saved_model` in handling user-defined components.


**Example 3: Saving and Loading a Functional API Model with Multiple Inputs and Outputs**

```python
import tensorflow as tf

input_a = tf.keras.Input(shape=(784,))
input_b = tf.keras.Input(shape=(10,))

x = tf.keras.layers.Dense(64, activation='relu')(input_a)
x = tf.keras.layers.concatenate([x, input_b])
output = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.save('saved_model/functional_model')

restored_model = tf.keras.models.load_model('saved_model/functional_model')

print(model.summary())
print(restored_model.summary())

```

This example demonstrates the versatility of `tf.saved_model` with more complex model architectures using the functional API.  Multiple inputs and outputs are handled seamlessly, ensuring that the model's complete functionality is preserved during saving and loading.  This is critical for handling scenarios where data preprocessing or feature engineering steps are incorporated directly into the model.


**3. Resource Recommendations**

For a comprehensive understanding of TensorFlow 2's saving and loading mechanisms, I recommend studying the official TensorFlow documentation, specifically the sections detailing the `tf.saved_model` API and the `tf.keras.models.save_model` and `tf.keras.models.load_model` functions.  Furthermore,  exploring the examples provided in the TensorFlow tutorials will provide practical experience in implementing these techniques for various model architectures and complexities. Carefully reviewing best practices regarding version control and consistent environment setups for reproducible results is also crucial.  Finally, understanding the underlying structure of the SavedModel directory itself provides a deeper insight into its inner workings.
