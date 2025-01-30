---
title: "How can I transfer pre-trained TensorFlow models?"
date: "2025-01-30"
id: "how-can-i-transfer-pre-trained-tensorflow-models"
---
Transferring pre-trained TensorFlow models is a cornerstone of modern deep learning, enabling the effective application of sophisticated models to new tasks and datasets without requiring extensive training from scratch. The core principle rests on leveraging the learned representations from a model trained on a large dataset (e.g., ImageNet) and repurposing those features for a new, often smaller, dataset. This process drastically reduces training time, computational resources, and the amount of labeled data needed.

The fundamental idea revolves around the hierarchical nature of convolutional neural networks (CNNs), a common architecture for image-based tasks. Earlier layers in a CNN tend to learn generic features, such as edges, corners, and simple textures, which are broadly applicable across different image classification tasks. Later layers, conversely, learn task-specific features that are highly tailored to the dataset on which they were trained. When performing transfer learning, the typical approach is to freeze the weights of the earlier layers, preserving the generic feature extraction capability, and then train only the later layers, adapting them to the new dataset and classification task.

Here's a breakdown of how this is typically implemented with TensorFlow:

First, you load the pre-trained model, usually from TensorFlow Hub or TensorFlow's model garden. These models are generally saved in SavedModel format, facilitating their loading without requiring the original model's source code. You need to specify which layers to freeze and which to train. Often, the trainable layers involve a custom classification head specific to your task. This involves creating a new fully connected layer or a series of layers and connecting them to the unfrozen output of the pre-trained model. The key advantage is that these new layers are initialized randomly and then trained using a small training set on your target domain, using an appropriate loss function. Gradient updates are then applied only to these trainable layers during the backpropagation phase.

Here are three specific code examples to illustrate this process, each with varying degrees of complexity:

**Example 1: Simple Image Classification using a Pre-trained ResNet**

This example illustrates a rudimentary transfer learning scenario. The pre-trained ResNet50 model is used, with the last fully connected layer replaced with a new layer tailored to a classification problem of a different class size. The entire pre-trained section (excluding the classification layers) is frozen.

```python
import tensorflow as tf
import tensorflow_hub as hub

# Load pre-trained ResNet50 from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
feature_extractor_layer = hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)

# Create a new classification head
num_classes = 10 # Example: 10 target classes
model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Prepare a dummy dataset (replace this with your actual dataset)
# This dataset is simply for demonstration purposes and does not represent a real dataset.
import numpy as np
x_train = np.random.rand(100, 224, 224, 3)
y_train = np.random.randint(0, num_classes, 100)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

# Train only the trainable layers
model.fit(x_train, y_train, epochs=5)
```

This code first downloads the ResNet50 model from TensorFlow Hub, explicitly setting the `trainable` parameter to false within the `hub.KerasLayer`. A custom set of layers that are trainable are then added, adapting the architecture to a new classification task. The model is compiled, and then training occurs only on the newly added layers. Note, that a realistic dataset and training loop are needed to achieve accuracy on practical scenarios.

**Example 2: Fine-Tuning a Portion of a Pre-trained Model**

In this example, instead of freezing the entire pre-trained model's feature extraction layers, a portion of it is unfrozen for fine-tuning. This allows the model to further adapt its learned feature representations, potentially resulting in better performance but with an increased risk of overfitting to the new data.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Load pre-trained VGG16 model (without the top classification layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers initially
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze a specific number of layers starting from the end of the pre-trained model
fine_tune_at = 15
for layer in base_model.layers[fine_tune_at:]:
   layer.trainable = True

# Build a new classification head
num_classes = 5 # Example: 5 target classes
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False) # Pass training=False during inference/transfer learning setup.
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Prepare a dummy dataset (replace with actual data)
import numpy as np
x_train = np.random.rand(100, 224, 224, 3)
y_train = np.random.randint(0, num_classes, 100)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)


# Train with the unfrozen layers
model.fit(x_train, y_train, epochs=10)
```

This code loads a VGG16 model, then systematically freezes all its layers. After that, it unfrees a specific number of layers at the tail end of the model.  It is important to use `training=False` when setting up the transfer model during the integration stage, but not when running the final training loop.  This avoids the pre-trained model from learning unexpected changes in data due to operations like batch-normalization. The new classification layers are built and the model trained with a lower learning rate, reflecting the fine-tuning approach. This example demonstrates more advanced control over trainable parameters within the architecture.

**Example 3: Using a Pre-trained Model for Feature Extraction only**

This example illustrates a scenario where the entire pre-trained model is used exclusively as a feature extractor, and those features are subsequently used for training another classifier such as a Support Vector Machine or a simple linear regression model (instead of directly training it from the outputs of the pretrained model). This can be useful when the downstream task can be achieved with simpler models or requires a specific type of classifier.

```python
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load pre-trained InceptionV3 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"
feature_extractor = hub.KerasLayer(model_url, input_shape=(299, 299, 3), trainable=False)

# Prepare a dummy dataset (replace with your actual data)
num_samples = 100
x_data = np.random.rand(num_samples, 299, 299, 3)
y_data = np.random.randint(0, 2, num_samples)

# Extract features using the pre-trained model
features = feature_extractor(x_data)

# Split the extracted features and labels into training and testing set
x_train, x_test, y_train, y_test = train_test_split(features, y_data, test_size=0.2, random_state=42)

# Train an SVM classifier on extracted features
svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(x_test)

# Evaluate the SVM's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM accuracy: {accuracy}')
```

This code utilizes a pre-trained InceptionV3 model purely for feature extraction.  Instead of retraining any layers inside the Tensorflow model, it takes the features produced by that model and then trains an external SVM classifier using the generated feature vectors.  This demonstrates the flexibility in combining pre-trained deep learning models with traditional machine learning algorithms.

For continued study and implementation of transfer learning, I recommend exploring TensorFlow tutorials on their official website and relevant machine learning textbooks which cover convolutional neural networks.  Furthermore, actively searching and utilizing well-documented, publicly available model zoos and data set repositories can provide invaluable context to this domain. Understanding common strategies in the literature can help in adapting these approaches to specific problem domains. Examining existing solutions and implementations on various platforms such as Github can also be insightful.
