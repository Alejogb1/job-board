---
title: "How does MobileNet utilize Grad-CAM for transfer learning?"
date: "2025-01-30"
id: "how-does-mobilenet-utilize-grad-cam-for-transfer-learning"
---
MobileNet's integration with Grad-CAM for transfer learning hinges on leveraging the pre-trained convolutional base to identify crucial regions within an input image that contribute most significantly to the classification outcome, even when applied to a novel target domain. Grad-CAM, or Gradient-weighted Class Activation Mapping, provides a visual interpretation of a convolutional neural network's decision-making process, showing which areas of the input were most influential in predicting a specific class. I’ve utilized this method in several projects involving object localization in medical imaging and environmental monitoring; the ability to pinpoint areas of focus is invaluable, especially for debugging and understanding model behavior after transfer learning.

The core process involves passing an input image through the MobileNet model, often pre-trained on ImageNet, up to a specified convolutional layer. This layer acts as a feature extractor, transforming the image into a series of feature maps. Instead of using the output from the fully connected layers (which are frequently discarded during transfer learning), the convolutional feature maps are targeted. The specific choice of layer is crucial; the earlier layers capture lower-level features, like edges and corners, while later layers encode more abstract concepts. Generally, targeting layers closer to the end of the convolutional base provides a better representation of features learned relevant to the classification task.

Once the feature maps are obtained, Grad-CAM computes the gradient of the score for a particular class (identified by backpropagation through the network) with respect to these feature maps. These gradients represent the importance of each feature map in making a prediction for that particular class. Crucially, these gradients are then globally average-pooled across the spatial dimensions, creating a set of weights that signify the importance of each individual feature map. The weights are then used to produce a weighted linear combination of the original feature maps. The resulting weighted feature map is then passed through a ReLU activation function to only emphasize the relevant regions and avoid the display of negative impact. The final output is then upsampled back to the input image's spatial dimensions, allowing us to visualize the regions the model focused on for its decision. This allows a human observer to see where the model is “looking” within the image and evaluate if this is reasonable for the intended classification.

For transfer learning, this process is particularly useful. When adapting MobileNet (or any other pre-trained CNN) to a new dataset, the original fully connected layers are often replaced with a new set that matches the desired classification problem. The pre-trained convolutional layers, having learned generic visual features from a large dataset, remain frozen or are fine-tuned for the specific task. During this adaptation, it is crucial to assess if the model is leveraging these pre-trained feature extractors effectively for the new task. Grad-CAM helps identify if the model is making decisions based on correct features within the new input domain or if it needs further training or architectural modification. It allows developers to visualize what the network has learned to focus on.

Here are some code examples to illustrate how Grad-CAM is implemented, using Python with TensorFlow/Keras, along with explanations:

**Example 1: Basic Grad-CAM implementation with a single target class**

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

def create_gradcam(img_path, model, last_conv_layer_name, class_idx):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_score = preds[:, class_idx]

    grads = tape.gradient(class_score, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.matmul(last_conv_layer_output, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    return superimposed_img

# Example usage
model = MobileNet(weights='imagenet')
last_conv_layer_name = 'conv_pw_13'  # Target layer before global average pooling
image_path = 'sample_image.jpg' # Replace with your image path
class_index = 243  # Class index for 'cat' in ImageNet
gradcam_img = create_gradcam(image_path, model, last_conv_layer_name, class_index)
plt.imshow(gradcam_img/255)
plt.show()
```

*Commentary*: This example demonstrates the core functionality. It loads a MobileNet model, processes the input image, finds gradients, performs weighted averaging and upscales the resulting heatmap to the original input image size. The heatmap is then applied as an overlay with a color scheme for visual clarity, effectively highlighting the input regions of interest based on the specific class of interest. Note that this example specifically targets class 243 (cat).

**Example 2: Grad-CAM with multiple target classes for a multiclass classification**

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

def create_gradcam_multiple_classes(img_path, model, last_conv_layer_name, class_indices):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    superimposed_imgs = []

    for class_idx in class_indices:
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_score = preds[:, class_idx]

        grads = tape.gradient(class_score, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = tf.matmul(last_conv_layer_output, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap)

        heatmap = heatmap.numpy()
        img_original = cv2.imread(img_path)
        img_original = cv2.resize(img_original, (224, 224))
        heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img_original
        superimposed_imgs.append(superimposed_img)
    return superimposed_imgs

# Example Usage:
model = MobileNet(weights='imagenet')
last_conv_layer_name = 'conv_pw_13'
image_path = 'sample_image.jpg'
class_indices = [243, 888, 345]  # Class indices for "cat", "dog" and "bird" respectively
gradcam_imgs = create_gradcam_multiple_classes(image_path, model, last_conv_layer_name, class_indices)

for i, img in enumerate(gradcam_imgs):
    plt.figure()
    plt.title(f'Class index: {class_indices[i]}')
    plt.imshow(img/255)
    plt.show()
```
*Commentary*: This demonstrates Grad-CAM for a multiclass scenario. Instead of a single class, the code calculates and displays heatmaps for multiple classes (e.g., ‘cat’, ‘dog’, and ‘bird’). This is useful when needing to understand the network’s attention across different target classes and assess whether there are any overlaps in the areas of attention for different categories within an image.

**Example 3: Integration of Grad-CAM with fine-tuned model for new classes**

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2

# Dummy Fine-tuning (replace with your fine-tuning code)
base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) # Example with custom layers
predictions = Dense(5, activation='softmax')(x) # 5 new classes
fine_tuned_model = Model(inputs=base_model.input, outputs=predictions)
fine_tuned_model.load_weights("path/to/fine_tuned_weights.h5") # Load your fine-tuned weights

def create_gradcam_finetuned(img_path, model, last_conv_layer_name, class_idx):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_score = preds[:, class_idx]

    grads = tape.gradient(class_score, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.matmul(last_conv_layer_output, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()
    img_original = cv2.imread(img_path)
    img_original = cv2.resize(img_original, (224, 224))
    heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_original

    return superimposed_img


# Example Usage:
last_conv_layer_name = 'conv_pw_13'
image_path = 'sample_image.jpg' # Replace with your path
class_index = 3  # Index of the fine-tuned class you want to examine.
gradcam_img = create_gradcam_finetuned(image_path, fine_tuned_model, last_conv_layer_name, class_index)
plt.imshow(gradcam_img/255)
plt.show()
```
*Commentary*: This example shows how Grad-CAM can be used after fine-tuning a MobileNet on a new dataset. In place of the generic ImageNet pre-trained classification head, a new set of dense layers are implemented for the transfer learning task, and the corresponding weight are loaded. The Grad-CAM implementation remains the same, effectively illustrating the regions that the *fine-tuned* model focuses on for classifying the new classes. This allows for further investigation of the model performance after it has been fine-tuned on the new classification target.

For further exploration, I recommend focusing on several key concepts. Firstly, delving into the backpropagation algorithm will give a deeper understanding of how gradients are computed in the first place. Secondly, exploring the various types of convolutional layers will reveal how their features affect different levels of abstraction. Thirdly, understanding how the choice of activation functions impact model performance is necessary to fully understand why the ReLU is chosen here. Finally, studying how different CNN architectures are designed will improve the understanding of the benefits and limitations of MobileNet specifically. These resources, readily available in textbooks and online courses, will provide a more robust foundation for effectively utilizing and interpreting Grad-CAM in conjunction with transfer learning models. This combination of practical application (as demonstrated) and further conceptual study will prove invaluable for successfully implementing image classification models.
