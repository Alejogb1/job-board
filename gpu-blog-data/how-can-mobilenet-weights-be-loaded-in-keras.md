---
title: "How can MobileNet weights be loaded in Keras using a specific input tensor?"
date: "2025-01-30"
id: "how-can-mobilenet-weights-be-loaded-in-keras"
---
The core challenge in loading MobileNet weights with a specific input tensor in Keras arises from the inherent dependency between the pre-trained model’s input layer shape and the shape expected by its weights. A pre-trained MobileNet model, typically expecting an input tensor of shape `(224, 224, 3)`, will reject any input tensor of a different shape when loading the weights, unless specific adjustments are made.  My experience working on a real-time object detection project using customized input sizes highlighted the need for this understanding.

The straightforward approach of loading weights directly into a model with a modified input tensor leads to a `ValueError` because the weight shapes are incompatible. The initial input layer of the pre-trained MobileNet model and the custom input layer are not identical. Thus, you can’t simply load the standard weights into the altered layer. Instead, we need to create a “bridge” or a partially loaded model. We do this by separating the feature extraction portion of the model, which is adaptable to input sizes, from the classification layers which need specific input dimensions after the feature extraction. We leverage functional API for this manipulation and then the custom input tensor will effectively “feed” the new, adjusted model.

The following steps outline the process: First, we define a custom input tensor using `keras.Input()`. We must then construct the MobileNet model *without* its top classification layer (which has fixed input dimensions). We then use this 'partial' MobileNet to obtain the output tensor that results from passing our custom input through the feature extracting layers. This output tensor is then used as the input to our custom classification layers that now account for the altered input size. Finally, we load the MobileNet weights into only the initial feature extraction model we constructed. We are intentionally bypassing the classification layers so that the weights of these layers that were expecting a certain input dimension are not disturbed, and then we use a new custom classification layers that now has the dimensions of the output after feature extraction. This approach preserves the benefits of pre-trained weights while adapting to various input sizes.

**Code Example 1: Creating a MobileNet feature extractor with custom input shape**

```python
import tensorflow as tf
from tensorflow import keras

def create_mobilenet_feature_extractor(input_shape):
    """
    Creates a MobileNet feature extractor without its top classification layer.

    Args:
        input_shape (tuple): The shape of the input tensor (height, width, channels).

    Returns:
         tuple: input_tensor and output_tensor of partial MobileNet model
    """
    input_tensor = keras.Input(shape=input_shape)

    base_model = keras.applications.MobileNet(
        input_tensor=input_tensor,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    output_tensor = base_model.output
    return input_tensor, output_tensor


input_shape = (160, 160, 3) # Example custom input size
input_tensor, feature_tensor = create_mobilenet_feature_extractor(input_shape)


print("Custom Input Tensor Shape:", input_tensor.shape)
print("Feature Tensor Shape after partial MobileNet:", feature_tensor.shape)


```

This first example defines a function that instantiates a MobileNet model using the Keras Applications API. The crucial part is setting `include_top=False`. This option removes the classification layers that impose the 224x224 input requirement. By specifying the input tensor using `input_tensor`, we ensure that our custom shape is respected and used as the first layer, instead of the default input tensor. The function returns two tensors the input tensor of our modified model and the output tensor of this model, which will be connected to our custom layers that will be added later. I have observed that specifying 'avg' for the pooling argument greatly simplifies the handling of the tensor dimensions coming out of the feature extraction part, and we don't have to flatten them before passing to a Dense layer. This allows this process to support arbitrary input dimensions.

**Code Example 2: Adding Custom Classification Layers**

```python
def add_custom_classification_head(feature_tensor, num_classes):
    """
    Adds custom classification layers on top of the MobileNet feature tensor.

    Args:
        feature_tensor (tensor): Output tensor from the MobileNet feature extractor.
        num_classes (int): The number of output classes.

    Returns:
        keras.Model: A Keras model with the custom classification layers
    """
    x = keras.layers.Dense(128, activation='relu')(feature_tensor)
    x = keras.layers.Dropout(0.5)(x)
    output_tensor = keras.layers.Dense(num_classes, activation='softmax')(x)

    return output_tensor

num_classes = 10  # Example number of classes for classification
output_tensor = add_custom_classification_head(feature_tensor, num_classes)

print("Custom Classification Output Shape:", output_tensor.shape)

```

In this example, we add new layers to the feature tensor extracted in the previous step. I have used dense layers, however, more complicated layers such as LSTMs can be used if required. Crucially, these new layers are not initialized using ImageNet weights, so they need to be trained using labeled data. We instantiate a new model using the input tensor and the final output tensor so that we can train the model with custom data.

**Code Example 3: Creating the complete model**

```python
def create_full_model(input_shape, num_classes):
    """
    Creates the complete Keras model combining the feature extractor and custom classification layers

    Args:
        input_shape (tuple): The shape of the input tensor (height, width, channels)
        num_classes (int): The number of output classes.
    
    Returns:
        keras.Model: A complete Keras model.
    """

    input_tensor, feature_tensor = create_mobilenet_feature_extractor(input_shape)
    output_tensor = add_custom_classification_head(feature_tensor, num_classes)

    full_model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return full_model


full_model = create_full_model(input_shape, num_classes)
full_model.summary()

```
This final example integrates the two previous examples to show how a full model is created. The function now returns a `keras.Model` that is ready to be trained. By inspecting the model summary using `full_model.summary()`, it becomes clear how the `input_tensor` flows through the initial partial MobileNet model and finally to the classification layers.

This approach of partial model loading is powerful.  If more than just classification is needed, the output tensor of our feature extraction model can be fed into other models as well. It is important to remember to only load the ImageNet weights into the initial MobileNet part of the model, and to train the newly initialized layers from scratch.

When working with pre-trained models, I typically rely on two primary resources for information and guidance: first, the official Keras documentation provides a good overview of each model in the Keras Applications module, including the expected input sizes and parameter options. Second, TensorFlow tutorials and official guides offer a broader understanding of pre-trained model handling within the TensorFlow ecosystem and can be useful for troubleshooting unexpected behaviors. While specific examples often need adaptation, understanding the general approach significantly streamlines the process. These resources collectively provide essential details for handling model architectures, weight loading, and model manipulation. Understanding these principles will allow you to implement robust and flexible deep learning models even with complicated pre-trained weights.
