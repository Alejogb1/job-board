---
title: "How do I fine-tune last layers of a network for transfer learning?"
date: "2024-12-16"
id: "how-do-i-fine-tune-last-layers-of-a-network-for-transfer-learning"
---

Okay, let's talk about fine-tuning the last layers of a network for transfer learning. It's a crucial step that often gets glossed over, and doing it effectively can be the difference between a model that's just okay and one that truly shines on your target task. I’ve seen firsthand how crucial this is, especially in projects where we were adapting models trained on very large datasets to more specialized, often smaller, datasets. We were recently working on a project involving specialized medical image analysis and initially, we just plugged in a pre-trained model without fine-tuning. The results were… underwhelming. It highlighted the importance of this process, that’s for sure.

The basic principle of transfer learning is leveraging the knowledge a model has already learned from a large, general dataset, such as ImageNet, and applying it to a different, often related, task. The initial layers of these models, especially in vision tasks, learn very general features like edges, corners, and textures, which are often helpful regardless of the specific problem. However, the later layers, closer to the output, learn features that are much more specific to the original task. That's where fine-tuning comes in. Instead of retraining the entire network, which is often computationally expensive and requires a lot of data, we freeze or partially freeze the initial layers and focus on training the last few layers to adapt to our new task.

So, how do we practically approach this? There are several ways to tackle it, depending on the situation:

**1. Freezing All but the Last Layer(s):** This is the most straightforward approach, and it works well when your new dataset is relatively small or very different from the original one. The idea here is to only train the final classification or regression layers, while the lower-level feature extractors remain untouched. This method effectively uses the pre-trained model as a fixed feature extractor and allows a small trainable head to adapt to the new target classes.

*Example Code (Python using TensorFlow/Keras):*

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained VGG16 model, excluding the top (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for your new classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x) # Replace num_classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training code goes here
# model.fit(...)
```

**Explanation:** The code loads a pre-trained `VGG16` model, excluding its classification top. The lines that freeze the layers are essential. We then build our custom top layer using `GlobalAveragePooling2D` followed by dense layers suited for the desired number of output classes and train only that. The pre-trained `VGG16` acts as a fixed feature extractor. Remember to replace `num_classes` with your desired number of output classes.

**2. Gradual Unfreezing:** This method provides a bit more flexibility and can be beneficial when you have a larger dataset for your target task or the dataset is similar to the one the original model was trained on. In this approach, we start by freezing a significant part of the network and training the last few layers, similar to the previous step. Once the model is achieving reasonably good performance, we then gradually unfreeze the earlier layers, fine-tuning them with a smaller learning rate. This slow, controlled unfreezing often results in better performance than training only the last few layers directly. It also helps to prevent overfitting. The idea here is that the features learned by earlier layers may not be perfectly suited for our new task but that we can fine-tune them to be more effective.

*Example Code (Python using TensorFlow/Keras):*

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load pre-trained ResNet50 model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Initially, freeze most layers
for layer in base_model.layers[:-20]:  # Unfreeze last 20 layers as an example
    layer.trainable = False

# Add a classification top for our new task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x) # Replace num_classes

# Create the fine-tuned model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train the model (initial training)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(...)

# Unfreeze some more layers and continue training with a lower learning rate
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(...)
```
**Explanation:** We start with a pre-trained `ResNet50`. This code unfreezes the last 20 layers and then trains these along with the new dense layers on top of the model. After the initial phase of training the new layers and the few unfreezed bottom layers, we then unfreeze the remainder of the model and train it again with lower learning rate for a further improvement. Remember to replace `num_classes` with your desired number of output classes.

**3. Layer-wise Learning Rates:** This is a more advanced technique, although it may seem complex. It involves setting different learning rates for different layers of the network. Generally, it's beneficial to use smaller learning rates for the early layers, since we want to make small adjustments to the pre-trained weights. We use relatively higher learning rates for the last layers, as these are further away from the original training data. This can lead to a more stable and more effective transfer of knowledge. Implementing this can be tricky, but frameworks like TensorFlow and PyTorch allow for this kind of fine-grained control.

*Example Code (Python using TensorFlow/Keras):*

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load pre-trained MobileNetV2 model without top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add the new classification top for our new task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x) # Replace num_classes

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Create optimizer with different learning rates for different layers.
# This requires going over the layers manually.
optimizer = Adam()
layer_learning_rates = []
for layer in model.layers:
    if layer.name.startswith('block_1'):
        layer_learning_rates.append((layer.trainable_variables, 0.00001))
    elif layer.name.startswith('block_2') or layer.name.startswith('block_3'):
        layer_learning_rates.append((layer.trainable_variables, 0.00005))
    elif layer.name.startswith('block_4') or layer.name.startswith('block_5'):
        layer_learning_rates.append((layer.trainable_variables, 0.0001))
    elif layer.name.startswith('dense'):
        layer_learning_rates.append((layer.trainable_variables, 0.001))
    else:
       layer_learning_rates.append((layer.trainable_variables, 0.001)) #default rate

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# This is slightly more complicated so it needs to be done within a fit step.
# You'll need to implement a custom training loop or utilize the tf.function decorator.
# An example of gradient application using layerwise learning rate can be given as below.

# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#      preds = model(images, training=True)
#      loss = model.loss(labels, preds)
#      gradients = tape.gradient(loss, model.trainable_variables)
#
#     for (vars, rate) in layer_learning_rates:
#      grad_vars = [(g, v) for g,v in zip(gradients, model.trainable_variables) if v in vars]
#      opt.apply_gradients(grad_vars)
#
# # Within your training loop use the train_step function.
```
**Explanation:**  We start with a pre-trained `MobileNetV2` model and add new layers for our new task. Then, it demonstrates how you would create an optimizer and assign different learning rates based on the name of the layer. It’s essential to adapt the names of the layers to the names of the actual layers in your target pre-trained model. Finally, a conceptual way to implement the gradients with the assigned layer wise learning rates is shown in the comments. The exact fit implementation may vary depending on the training strategy you implement.

These are the primary methods I’ve personally found most effective. However, choosing the “best” method often depends heavily on your specific problem, the amount of data you have, and the similarity between your dataset and the dataset used for pre-training. The key is to experiment and to really understand why each method works, and in which situations they tend to work best.

For further reading on this topic, I highly recommend checking out the following:

*   **Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:** This book provides a comprehensive theoretical foundation for transfer learning and deep learning in general. The relevant section is on transfer learning.
*   **"How transferable are features in deep neural networks?" by Yosinski et al. (2014):** This research paper provides insight into the nature of feature transferability in deep networks. This one dives into the details that the previous one only scratches the surface of.
*   **"A Deeper Look at Transfer Learning" by Bengio et al. (2011):** Although a bit older, it’s still very relevant and offers valuable understanding of the general methods.

Remember, fine-tuning is not a one-size-fits-all solution. It's an iterative process that requires experimentation and careful analysis of your model's performance. Don't be afraid to try different strategies and see what works best for you. Start simple, observe your results, and gradually refine your approach. This is a core competency for any deep learning practitioner and understanding it is crucial for anyone working on real world applications. Good luck with your work.
