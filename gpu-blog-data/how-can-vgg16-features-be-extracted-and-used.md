---
title: "How can VGG16 features be extracted and used as input for other models like ResNet, ViT-Keras?"
date: "2025-01-30"
id: "how-can-vgg16-features-be-extracted-and-used"
---
Convolutional Neural Networks (CNNs), particularly those pre-trained on large datasets such as ImageNet, offer a potent strategy for feature extraction. I’ve leveraged this principle extensively across various computer vision projects, finding it a highly effective approach to transfer learning and model initialization. Specifically, the VGG16 architecture, despite its age, provides a robust foundation for generating feature maps that can be repurposed for subsequent, more complex models like ResNet or Vision Transformers (ViTs). These feature maps encapsulate learned spatial hierarchies, allowing downstream networks to focus on higher-level abstractions.

The process generally unfolds in three primary stages: loading the pre-trained VGG16 model, selectively extracting its feature maps, and preparing this output for ingestion into the target model. The goal isn’t to retrain the VGG16's convolutional layers. Instead, we leverage their already learned filters. We’re essentially using VGG16 as a feature extraction machine before plugging the output into the network we intend to train.

The selection of which layer's feature map to use is critical and dependent on the specific task. Early convolutional layers tend to capture low-level features such as edges and corners, whereas the deeper layers capture more complex, object-centric patterns. Often, I've found that the output from the final convolutional layer, just before the fully connected layers, produces a good balance between generality and task relevance. We typically discard the fully connected layers in the VGG16 model entirely when using it for feature extraction. The rationale is that these layers are highly specific to the ImageNet dataset's 1000 classes, while our target model may be tailored for an entirely different use case.

The extracted features are essentially a set of multi-dimensional arrays. We can then reshape these arrays into a format suitable for the target model. For example, if we are using a ResNet model which processes input in a structured way, these arrays are reshaped to align with what ResNet expects. Or, if the downstream model is a Vision Transformer (ViT), the feature maps are usually converted into a sequence of vectors. This step might involve flattening, normalization or the use of techniques such as patch embedding depending on the specific target architecture requirements.

To illustrate, consider a scenario where I need to classify images of different types of medical scans using ResNet50. I will leverage VGG16 for feature extraction.

Here is the first code example which utilizes TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

# Load VGG16 model, excluding the top (fully connected) layers.
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the VGG16 layers to prevent retraining.
for layer in vgg16_base.layers:
    layer.trainable = False

# Specify input for the base model and the feature map to extract.
inputs = Input(shape=(224, 224, 3))
features = vgg16_base(inputs)

# Flatten the feature maps, making them suitable as an input for ResNet
flattened_features = Flatten()(features)

# Create the feature extractor model.
feature_extractor = Model(inputs=inputs, outputs=flattened_features)

# The output shape will depend on which feature maps are extracted.
print(f"Shape of the feature map: {feature_extractor.output_shape}")

```

This code loads the VGG16 model, explicitly omitting its fully-connected layers through the `include_top=False` argument, which is vital for this workflow. The loop that freezes the layers prevents inadvertent retraining of the VGG16 weights. We define the input and output of our new feature extraction model and create the final `feature_extractor` model. The output of this model are the flattened feature maps from the last convolutional block. This extracted feature data can then be used as input for another model.

The second code example provides a specific example of how to pass these extracted features to a basic ResNet-like model.

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model

# Assuming we have the feature extractor from the previous code.
# Let’s create an input layer for ResNet which receives flattened feature maps.

input_features = Input(shape=feature_extractor.output_shape[1:]) # Shape is inferred from feature extractor
x = Dense(256)(input_features)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(10)(x)  # Final output for a 10 class problem.

# Construct the ResNet-like classification head.
resnet_model = Model(inputs=input_features, outputs=x)

# Connect the VGG16 feature extractor with the ResNet model.
full_model = Model(inputs=feature_extractor.input, outputs=resnet_model(feature_extractor.output))


# Display the model summary.
print("Model Summary")
full_model.summary()
```

This snippet demonstrates how to create a simple ResNet-like model that takes the flattened output from the VGG16 feature extractor as input. The dense layers will then learn to classify the images based on these pre-extracted features. We then link these two models together to create a complete pipeline. The key here is that `resnet_model` uses `feature_extractor.output` as input which was defined in previous example. Note also that we keep the VGG16 portion frozen during the training of `full_model`. The `full_model.summary()` command provides an excellent way to check that the shape of the model and the number of parameters are as expected.

For Vision Transformers (ViT), the approach differs slightly because ViTs expect sequence-like data. The next code example shows how we can leverage VGG16 for feature extraction with a ViT model:

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Reshape, Conv2D, LayerNormalization
from tensorflow.keras.models import Model

# Same VGG16 loading as before.
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg16_base.layers:
    layer.trainable = False

inputs = Input(shape=(224, 224, 3))
features = vgg16_base(inputs)

# The last conv layer has shape, let's say (7,7,512), we will flatten that into
# a sequence of vectors.

# First a Conv2D transformation to allow channel-wise transformation.
# In a typical ViT, a linear projection is used here, but we will
# use a 1x1 convolution which is equivalent.
features_transformed = Conv2D(filters = 768, kernel_size = (1,1), padding='same')(features)

# Reshape the features.
# The last dim is the embedding dimension, the first is the sequence length (which is 7 * 7 = 49).
# The new shape is (49, 768).
reshaped_features = Reshape((-1,768))(features_transformed)


# Add a layer normalisation.
normalized_features = LayerNormalization()(reshaped_features)


# Build feature extraction model.
feature_extractor_vit = Model(inputs=inputs, outputs=normalized_features)

# Print the output shape to use with a ViT model later.
print("Shape of features for ViT input:", feature_extractor_vit.output_shape)

```

Here, instead of merely flattening the feature maps, we first project them using a 1x1 convolution to a new embedding dimension (in this case, 768, a common embedding size in ViT models). This serves as a learnable linear embedding step. Then the feature maps are reshaped to create a sequence of vectors, a format ViTs expect. I've then included a Layer Normalisation operation on this data as this is also commonly used before passing data to a ViT encoder. The resultant output of this `feature_extractor_vit` model can then be used as input to a Vision Transformer by simply passing the extracted features to the ViT's input.

In summary, extracting VGG16 features requires careful consideration of the target model’s input requirements. We can take the VGG16's pre-trained weights, specifically those from the convolutional layers, and use them to generate meaningful feature maps. These are then either flattened or reshaped, transformed and/or normalized to be used as input for the model one would like to train, whether it is ResNet or a Vision Transformer. This approach avoids the need for training all layers from scratch, greatly accelerating the training process and improving performance especially on tasks with limited datasets.

For further exploration, I would recommend the following: the official TensorFlow and Keras documentation, which provides excellent insights on model creation and usage. Standard computer vision textbooks and online courses often have detailed sections on transfer learning and feature extraction with pre-trained models. Finally, reviewing research papers on CNN and ViT architectures will help to better understand the practical applications of these techniques.
