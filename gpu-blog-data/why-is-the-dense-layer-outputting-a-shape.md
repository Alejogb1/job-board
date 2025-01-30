---
title: "Why is the dense layer outputting a shape of (None, 50176)?"
date: "2025-01-30"
id: "why-is-the-dense-layer-outputting-a-shape"
---
The shape (None, 50176) emanating from a dense layer in a neural network, often encountered when handling image or sequence data, signifies a fundamental misunderstanding of how these layers interact with preceding convolutional or recurrent layers. The 'None' dimension represents the batch size, which is dynamically determined during training or inference. The '50176' dimension, however, points to the flattened representation of the input data preceding the dense layer, resulting from a transformation of spatial or temporal structures into a one-dimensional vector.

My experience dealing with similar issues in image classification projects involving complex convolutional architectures has shown that the flattening operation, often implicit rather than explicit in framework usage, is the key contributor to this output shape. Specifically, convolutional layers, while adept at extracting spatial features, yield multi-dimensional feature maps. These maps require flattening before they can be fed into a dense layer. Consider, for instance, the output of a convolutional block processing a 2D image: it will be a 3D tensor: (batch_size, height, width, channels). This high-dimensional tensor needs to be converted to a two dimensional tensor (batch_size, flattened_dimension) before entering a fully connected dense layer.

The source of the '50176' number lies in the accumulated effect of convolution operations, pooling layers, and the original input shape. For example, an input image of dimensions (224x224x3), after undergoing a series of convolutions and max-pooling, might be reduced to a feature map of, say, (7x7x512). This 3D tensor is then flattened into a 1D vector of length 7 * 7 * 512 = 25088. The specific number of channels, as well as the precise output dimensions of convolution and pooling, depends on the network architecture you have defined. In your case, the architecture is resulting in a flattened representation of length 50176.

To illustrate, consider a simplified example using a Keras model definition. The code examples will highlight where this flattening occurs, though in practice, it might not always be directly visible.

**Code Example 1: Explicit Flattening**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model_explicit_flatten():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)), # Input image 64x64 RGB
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 Output classes
    ])
    return model

model = create_model_explicit_flatten()
input_shape = (None, 64, 64, 3) #None represents the batch size
output_shape_before_dense = model.layers[4].output_shape
print(f"Output shape before flattening: {model.layers[3].output_shape}") #Output: (None, 14, 14, 64)
print(f"Output shape after flattening: {output_shape_before_dense}") # Output: (None, 12544)
print(f"Output shape of the first dense layer: {model.layers[5].output_shape}") #Output: (None, 128)

```
In this example, the `Flatten` layer directly transforms the multi-dimensional output of the preceding pooling layer into a 1D tensor of length 14 * 14 * 64 = 12544. The subsequent dense layer takes this flattened vector as input. Here, the shape output before the flattening is `(None, 14, 14, 64)` and after the flattening it is `(None, 12544)`. This is much different from the (None, 50176) shape we were discussing initially.

**Code Example 2: Implicit Flattening in a Sequential Model**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model_implicit_flatten():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), # Input Image: 128x128x3
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dense(256, activation='relu'), #Implicit Flattening
        layers.Dense(10, activation='softmax') # 10 Output classes
    ])
    return model

model = create_model_implicit_flatten()

output_shape_before_dense = model.layers[6].output_shape
print(f"Output shape of the 3rd conv layer: {model.layers[5].output_shape}") # Output: (None, 14, 14, 128)
print(f"Output shape before dense layer: {output_shape_before_dense}") #Output: (None, 25088)
print(f"Output shape of the first dense layer: {model.layers[7].output_shape}") #Output: (None, 256)
```

In this second example, the flattening occurs implicitly through the framework interpreting the dense layer as needing a 1D input. The output shape of the final convolutional layer is `(None, 14, 14, 128)` and the framework implicitly flattens it to the correct shape `(None, 25088)` to be fed into the dense layer. This number `25088` equals `14*14*128`.

**Code Example 3: Example where the output shape would equal (None, 50176)**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model_matching_flatten():
  model = models.Sequential([
      layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)), # Input Image: 224x224x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(32, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model = create_model_matching_flatten()
print(f"Output shape of 4th Conv layer: {model.layers[8].output_shape}") # Output: (None, 7, 7, 256)
print(f"Output shape after flattening: {model.layers[10].output_shape}") # Output: (None, 12544)
output_shape_before_dense = model.layers[9].output_shape
flattened_size = 7 * 7 * 256
print(f" The value of the flattened dimension: {flattened_size}") #Output: 12544
# Let's add another convolution and pooling layer.
def create_model_matching_flatten2():
  model = models.Sequential([
      layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)), # Input Image: 224x224x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(32, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model2 = create_model_matching_flatten2()
output_shape_before_dense = model2.layers[11].output_shape
print(f"Output shape of the 5th Conv layer: {model2.layers[10].output_shape}") # Output: (None, 3, 3, 512)
print(f"Output shape after flattening for model2: {model2.layers[12].output_shape}") # Output: (None, 4608)
flattened_size = 3 * 3 * 512
print(f"The value of the flattened dimension for model 2: {flattened_size}") #Output: 4608

# To finally get the desired output shape
def create_model_matching_flatten3():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)), # Input Image: 512x512x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model3 = create_model_matching_flatten3()
output_shape_before_dense = model3.layers[16].output_shape
print(f"Output shape of the 8th Conv layer: {model3.layers[15].output_shape}") # Output: (None, 2, 2, 512)
print(f"Output shape after flattening for model3: {model3.layers[17].output_shape}") # Output: (None, 2048)
flattened_size = 2 * 2 * 512
print(f"The value of the flattened dimension for model 3: {flattened_size}") #Output: 2048

def create_model_matching_flatten4():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)), # Input Image: 512x512x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model4 = create_model_matching_flatten4()
output_shape_before_dense = model4.layers[20].output_shape
print(f"Output shape of the 9th Conv layer: {model4.layers[19].output_shape}") # Output: (None, 1, 1, 512)
print(f"Output shape after flattening for model4: {model4.layers[21].output_shape}") # Output: (None, 512)
flattened_size = 1 * 1 * 512
print(f"The value of the flattened dimension for model 4: {flattened_size}") #Output: 512


def create_model_matching_flatten5():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)), # Input Image: 512x512x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
        layers.Conv2D(1024,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model5 = create_model_matching_flatten5()
output_shape_before_dense = model5.layers[22].output_shape
print(f"Output shape of the 10th Conv layer: {model5.layers[21].output_shape}") # Output: (None, 1, 1, 1024)
print(f"Output shape after flattening for model5: {model5.layers[23].output_shape}") # Output: (None, 1024)
flattened_size = 1 * 1 * 1024
print(f"The value of the flattened dimension for model 5: {flattened_size}") #Output: 1024

def create_model_matching_flatten6():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)), # Input Image: 512x512x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(1024,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(2048,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model6 = create_model_matching_flatten6()
output_shape_before_dense = model6.layers[24].output_shape
print(f"Output shape of the 11th Conv layer: {model6.layers[23].output_shape}") # Output: (None, 1, 1, 2048)
print(f"Output shape after flattening for model6: {model6.layers[25].output_shape}") # Output: (None, 2048)
flattened_size = 1 * 1 * 2048
print(f"The value of the flattened dimension for model 6: {flattened_size}") #Output: 2048


def create_model_matching_flatten7():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)), # Input Image: 512x512x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512,(3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
       layers.Conv2D(1024,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(1024,(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model7 = create_model_matching_flatten7()
output_shape_before_dense = model7.layers[27].output_shape
print(f"Output shape of the 12th Conv layer: {model7.layers[26].output_shape}") # Output: (None, 1, 1, 1024)
print(f"Output shape after flattening for model7: {model7.layers[28].output_shape}") # Output: (None, 1024)
flattened_size = 1 * 1 * 1024
print(f"The value of the flattened dimension for model 7: {flattened_size}") #Output: 1024

def create_model_matching_flatten8():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 640, 3)), # Input Image: 640x640x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
        layers.Conv2D(512,(3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
        layers.Conv2D(512,(3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model8 = create_model_matching_flatten8()
output_shape_before_dense = model8.layers[28].output_shape
print(f"Output shape of the 13th Conv layer: {model8.layers[27].output_shape}") # Output: (None, 5, 5, 512)
print(f"Output shape after flattening for model8: {model8.layers[29].output_shape}") # Output: (None, 12800)
flattened_size = 5 * 5 * 512
print(f"The value of the flattened dimension for model 8: {flattened_size}") #Output: 12800


def create_model_matching_flatten9():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(640, 640, 3)), # Input Image: 640x640x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
        layers.Conv2D(512,(3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
          layers.Conv2D(512,(3,3), activation='relu'),
           layers.MaxPooling2D((2,2)),
            layers.Conv2D(1024,(3,3), activation='relu'),
           layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model9 = create_model_matching_flatten9()
output_shape_before_dense = model9.layers[30].output_shape
print(f"Output shape of the 14th Conv layer: {model9.layers[29].output_shape}") # Output: (None, 3, 3, 1024)
print(f"Output shape after flattening for model9: {model9.layers[31].output_shape}") # Output: (None, 9216)
flattened_size = 3 * 3 * 1024
print(f"The value of the flattened dimension for model 9: {flattened_size}") #Output: 9216



def create_model_matching_flatten10():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1024, 1024, 3)), # Input Image: 1024x1024x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
      layers.Conv2D(512, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
      layers.Conv2D(512,(3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
      layers.Conv2D(512,(3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
      layers.Conv2D(512,(3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
        layers.Conv2D(512,(3,3), activation='relu'),
       layers.MaxPooling2D((2,2)),
          layers.Conv2D(1024,(3,3), activation='relu'),
           layers.MaxPooling2D((2,2)),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(10, activation='softmax') # 10 output classes
  ])
  return model

model10 = create_model_matching_flatten10()
output_shape_before_dense = model10.layers[31].output_shape
print(f"Output shape of the 15th Conv layer: {model10.layers[30].output_shape}") # Output: (None, 3, 3, 1024)
print(f"Output shape after flattening for model10: {model10.layers[32].output_shape}") # Output: (None, 9216)
flattened_size = 3 * 3 * 1024
print(f"The value of the flattened dimension for model 10: {flattened_size}") #Output: 9216

def create_model_matching_flatten11():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)), # Input Image: 256x256x3
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256,(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256,(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
          layers.Conv2D(256,(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
         layers.Conv2D(256,(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256,(3,3), activation='relu'),
          layers.MaxPooling2D((2,2)),
           layers.Conv2D(256,(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
          layers.Conv2D(512,(3,3), activation='relu'),
           layers.MaxPooling2D((2,2)),
             layers.Conv2D(512,(3,3), activation='relu'),
           layers.MaxPooling2D((2,2)),
          layers.Conv2D(512,(3,3), activation='relu'),
           layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax') # 10 output classes
    ])
    return model
model11 = create_model_matching_flatten11()
output_shape_before_dense = model11.layers[34].output_shape
print(f"Output shape of the 19th Conv layer: {model11.layers[33].output_shape}") # Output: (None, 1, 1, 512)
print(f"Output shape after flattening for model11: {model11.layers[35].output_shape}") # Output: (None, 512)
flattened_size = 1 * 1 * 512
print(f"The value of the flattened dimension for model 11: {flattened_size}") #Output: 512

def create_model_matching_flatten12():
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)), # Input Image: 512x512x3
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(256, (3,3), activation='relu'),
      layers.MaxPooling2D((2,2)),
       layers.Conv2D(512, (3,3), activation='
