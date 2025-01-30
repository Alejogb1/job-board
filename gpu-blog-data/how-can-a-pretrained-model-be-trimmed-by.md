---
title: "How can a pretrained model be trimmed by removing its top layers and saved as a new, smaller model?"
date: "2025-01-30"
id: "how-can-a-pretrained-model-be-trimmed-by"
---
Transfer learning often involves leveraging the feature extraction capabilities of a large, pre-trained neural network. However, deployment scenarios frequently demand models with reduced computational cost and memory footprint. Trimming the top layers, those closer to the output, is a common strategy for model downsizing while preserving the learned representations captured in the lower layers. This process hinges on correctly manipulating the model architecture and ensuring that the resultant, truncated model can be saved and loaded correctly. I've performed this process numerous times across different frameworks in past projects, and the core concepts tend to remain similar.

The fundamental principle is to dissect the original model, identify the desired point of truncation, and then construct a new model using only the layers leading up to that point. This new model effectively functions as a feature extractor, transforming input data into high-level, discriminative feature representations. The top layers, typically comprising fully connected layers in classification tasks, are responsible for mapping these extracted features to the final classification output. By removing them, the model loses its classification ability, but retains the powerful feature engineering capabilities that underpin it. The resulting reduced model can then be employed in various ways including as a component of a larger more complex machine learning pipeline.

The procedure involves the following steps: first, the original model must be loaded from disk or initialized based on the pre-trained configuration. Subsequently, the model architecture needs to be inspected to determine the names or indices of the layers targeted for removal. The actual truncation usually involves one of two common patterns, either directly building a new sequential model with the layers desired to be kept or making the network graph that represents the model and then removing nodes that correspond to unwanted layers and connecting the remaining nodes. The retained layers are then used to assemble the smaller model, and the modified model is saved to disk for further use. Careful attention must be paid to the specific frameworks' API, such as Keras, PyTorch or TensorFlow, since they have their own implementation of network graphs.

The key considerations are: preserving layer weights, ensuring compatibility with the target application and finally, managing input/output dimensions. When truncating a model, I always make sure to extract and transplant the weights and biases from the source model to the equivalent layers in the new reduced model. Failing to do so means that the transfer learning does not happen. Furthermore, the output shape of the truncated layer becomes the new model’s output shape. If the downstream task has a different input shape, the new reduced model will require adjustment. Additionally, saving and loading must be handled using the framework’s recommended tools to ensure proper persistence and loading of the reduced models, as opposed to saving data objects that happen to look like trained weights and biases.

Below are three code examples using three different frameworks. These examples illustrate how to trim a model, save the reduced model and test it.

**Example 1: Keras (TensorFlow Backend)**

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model

# Load a pre-trained VGG16 model (without top classification layers).
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Identify the last convolutional layer (before flatten layer)
last_conv_layer = base_model.get_layer('block5_pool')

# Create a new model up to this layer
truncated_model = Model(inputs=base_model.input, outputs=last_conv_layer.output)


#Save the truncated model
truncated_model.save('truncated_vgg16.h5')

# Verify that the model was correctly saved:
loaded_model = tf.keras.models.load_model('truncated_vgg16.h5')
print("Output Shape of Original Model: ",base_model.output.shape)
print("Output Shape of Truncated Model: ", truncated_model.output.shape)
print("Output Shape of loaded Model: ", loaded_model.output.shape)
```

*   **Commentary:** This Keras example leverages the `Model` class to create a new model with the same inputs as the original VGG16 but with outputs derived from the desired intermediate layer, `block5_pool`. This method builds a copy of the underlying computational graph while retaining the loaded weights from the pre-trained network. The new model is then saved to a `.h5` file using the Keras API. After saving, a quick check confirms that the shapes are correct.

**Example 2: PyTorch**

```python
import torch
import torchvision.models as models

# Load a pre-trained ResNet50 model
base_model = models.resnet50(pretrained=True)

# Identify the layer at which to truncate (before the average pooling)
truncated_model = torch.nn.Sequential(*list(base_model.children())[:-2])


#Save the model
torch.save(truncated_model.state_dict(), 'truncated_resnet50.pth')

#Verify that the model was correctly saved:
loaded_model = models.resnet50(pretrained=False) #Load a new model, without pre-trained weights
truncated_list = list(loaded_model.children())[:-2]
loaded_truncated_model = torch.nn.Sequential(*truncated_list)
loaded_truncated_model.load_state_dict(torch.load('truncated_resnet50.pth'))

print("Output Shape of Original Model: ", base_model.fc.in_features)
print("Output Shape of Truncated Model: ", truncated_model(torch.randn(1, 3, 224, 224)).shape)
print("Output Shape of loaded Model: ", loaded_truncated_model(torch.randn(1, 3, 224, 224)).shape)
```

*   **Commentary:** This PyTorch example uses the `torch.nn.Sequential` container to extract the layers up to but not including the last two layers. The truncated ResNet50 model is saved by extracting the `state_dict`, which contains all learnable parameters, using `torch.save`. When loading, it loads state dictionary to the non pre-trained ResNet50 that has had the top two layers removed.  The shapes of both saved and loaded truncated models are checked using dummy inputs.

**Example 3: TensorFlow (Low-Level)**

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np


# Load a pretrained ResNet50
base_model = ResNet50(weights='imagenet', include_top = False, input_shape=(224, 224, 3))


# Identify the last conv layer
last_conv_layer = base_model.get_layer('conv5_block3_out')

#Create graph of network until conv5_block3_out
truncated_model = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)

#Get the weights of the model layers
weights = []
for layer in truncated_model.layers:
    weights.append(layer.get_weights())

# Manually write each layer's weight and biases to a TF Checkpoint

checkpoint = tf.train.Checkpoint(model = truncated_model)
checkpoint.save('truncated_resnet50_checkpoint/')

# Load the checkpoint
new_checkpoint = tf.train.Checkpoint(model = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output))
new_checkpoint.restore(tf.train.latest_checkpoint('truncated_resnet50_checkpoint/'))


print("Output Shape of Original Model: ", base_model.output.shape)
print("Output Shape of Truncated Model: ", truncated_model(tf.random.normal([1, 224, 224, 3])).shape)
print("Output Shape of loaded Model: ", new_checkpoint.model(tf.random.normal([1, 224, 224, 3])).shape)
```

*   **Commentary:** This TensorFlow example builds on the basic idea of the first example, however instead of saving it as an h5 file, it saves it using TensorFlow's low-level checkpoint system. First a graph for the model is created. Then, rather than using an automated save, the checkpoint is saved using `tf.train.Checkpoint` and later restored in a similar manner. Lastly, the same check for the sizes is performed.

These examples demonstrate that, regardless of the framework, the approach to model trimming involves identifying the split point, extracting the layers leading up to it, and saving the resulting reduced model. The critical steps that are universal are extracting weights from layers and properly managing the graph in order to preserve the computational structure and the underlying numerical properties.

For resources, I suggest reviewing the official documentation for the deep learning frameworks (TensorFlow, PyTorch, Keras) which offer in depth explanations of model construction, manipulation and persistence. Additionally, publications on feature extraction within deep learning are useful. Understanding the underlying concepts, as opposed to simply copying and pasting code, is essential. Publications which cover feature extraction in computer vision and language models are particularly helpful. Finally, tutorials, such as those that are published by frameworks, that specifically cover transferring learning are useful.
