---
title: "How do I create a deep learning model by concatenating two hidden layers of the same output shape of Resnet and VGG-16 using TensorFlow?"
date: "2024-12-15"
id: "how-do-i-create-a-deep-learning-model-by-concatenating-two-hidden-layers-of-the-same-output-shape-of-resnet-and-vgg-16-using-tensorflow"
---

Alright, so you're looking to merge the outputs of two different convolutional neural networks, specifically resnet and vgg-16, after they've gone through a few hidden layers, and you want to do this in tensorflow, achieving concatenation at the same output shape. i've been there, done that, got the t-shirt, and probably spent a few late nights staring at error messages because of it. it's not exactly rocket science, but there are some nuances to watch out for.

let's break it down. the core issue is grabbing specific intermediate layer outputs from both resnet and vgg-16, ensuring they have the same dimensions before concatenating, and then plugging that into your subsequent layers. the shape matching part is key, because that's where things usually fall apart. tensorflow expects consistent tensor shapes when you concatenate them along a specific axis.

i remember when i first tried this, i was working on a project that involved multimodal data, trying to fuse image features with audio features. i thought it was a brilliant idea, but i didn't account for the shape differences and i ended up with a rather verbose stack trace that i had to spend hours figuring out.

anyway, here's how you can approach this in tensorflow, with some specifics:

first off, let's load pre-trained resnet and vgg-16 models. tensorflow's `keras.applications` module makes this pretty straightforward. we want to make sure we can grab the layers at the points where we want to perform concatenation. this requires defining specific layer outputs to become the inputs to your concatenation layer.

```python
import tensorflow as tf
from tensorflow.keras.applications import resnet, vgg16
from tensorflow.keras.layers import concatenate

# load resnet
resnet_model = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# freeze the resnet base layers since they are pretrained
for layer in resnet_model.layers:
    layer.trainable = False

# load vgg16
vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# freeze the vgg16 base layers since they are pretrained
for layer in vgg_model.layers:
    layer.trainable = False

# example: select resnet's activation_40 output
resnet_output = resnet_model.get_layer('conv4_block6_out').output

# example: select vgg16's block5_conv3 output
vgg_output = vgg_model.get_layer('block5_conv3').output

print(f"resnet output shape:{resnet_output.shape}")
print(f"vgg output shape:{vgg_output.shape}")

```

now, in this first snippet, i'm loading the models with `include_top=false`, so we don't include the classification layer on the top which would be redundant here since we are just getting the middle layers for feature extraction which will go to another network. i am also freezing these base layers. this means that these convolutional network weights are not going to change during training and this will make training faster and better by not updating these learned weights. you'll need to pick the actual layers, the `layer_name` to grab the intermediate tensors. i’ve just used two as an example. what you need to do is use `resnet_model.summary()` and `vgg_model.summary()` to inspect the architecture and choose your target layers that will then be concatenated, these selected layers needs to have compatible tensor shapes for the concatenation to work. note the output shapes from the prints. they are tensors and have shape of batch\_size, height, width, channels. pay close attention to these because that is where a lot of the errors come from. they must match in height and width for the concatenation to happen across channels.

the key point here is `get_layer('layer_name').output`. this grabs the output tensor of the layer with that specific name. the name is the name of the layer as you can inspect in the output of `model.summary()`.

if the shape of these tensor outputs does not match along the height and width dimension and assuming that the first dimension is the batch size, then you need to reshape the tensors before concatenation. there are several ways you can do it, but one common way is through resizing the feature maps using an upsampling or downsampling layer like `tf.keras.layers.resizing`

let's add some resizing logic to the previous snippet. assuming the shapes do not match and need to be resized. here we make an assumption that we resize to the smaller shape of the two tensors. this works because in feature maps, the height and the width represents more abstract feature extraction in those areas, while the number of channels represent that amount of feature extraction in general. so resizing with an appropriate `tf.image.resize` interpolation method or using `tf.keras.layers.Resizing` is going to maintain some good representation in the areas where the features were originally extracted, but at the smaller scale.

```python
import tensorflow as tf
from tensorflow.keras.applications import resnet, vgg16
from tensorflow.keras.layers import concatenate, Resizing

# load resnet
resnet_model = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# freeze the resnet base layers since they are pretrained
for layer in resnet_model.layers:
    layer.trainable = False

# load vgg16
vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# freeze the vgg16 base layers since they are pretrained
for layer in vgg_model.layers:
    layer.trainable = False

# example: select resnet's activation_40 output
resnet_output = resnet_model.get_layer('conv4_block6_out').output

# example: select vgg16's block5_conv3 output
vgg_output = vgg_model.get_layer('block5_conv3').output

print(f"resnet output shape before resizing: {resnet_output.shape}")
print(f"vgg output shape before resizing: {vgg_output.shape}")

# resize if necessary

resnet_height=resnet_output.shape[1]
resnet_width=resnet_output.shape[2]
vgg_height=vgg_output.shape[1]
vgg_width=vgg_output.shape[2]

# get the minimum height and width for resize
min_height=min(resnet_height,vgg_height)
min_width=min(resnet_width,vgg_width)

# resize resnet output
if(resnet_height != min_height or resnet_width != min_width):
  resnet_output = Resizing(height=min_height,width=min_width)(resnet_output)

# resize vgg output
if(vgg_height != min_height or vgg_width != min_width):
  vgg_output = Resizing(height=min_height,width=min_width)(vgg_output)


print(f"resnet output shape after resizing: {resnet_output.shape}")
print(f"vgg output shape after resizing: {vgg_output.shape}")

```

now the resnet and vgg outputs should have the same height and width. it is crucial that you verify with the print statements in the previous snippets to have compatible shapes before concatenation. you can also use `tf.image.resize` but here i am using the `tf.keras.layers.Resizing` as it's easier to integrate in a functional keras model. in any case the principle of resizing the feature maps remains the same.

now, you concatenate. tensorflow makes this pretty straightforward with the `concatenate` layer and you specify which axis to use. typically you would use axis -1 which refers to the channels dimension of a tensor.

```python
import tensorflow as tf
from tensorflow.keras.applications import resnet, vgg16
from tensorflow.keras.layers import concatenate, Resizing, Input
from tensorflow.keras.models import Model

# load resnet
resnet_model = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# freeze the resnet base layers since they are pretrained
for layer in resnet_model.layers:
    layer.trainable = False

# load vgg16
vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# freeze the vgg16 base layers since they are pretrained
for layer in vgg_model.layers:
    layer.trainable = False


# example: select resnet's activation_40 output
resnet_output = resnet_model.get_layer('conv4_block6_out').output

# example: select vgg16's block5_conv3 output
vgg_output = vgg_model.get_layer('block5_conv3').output

print(f"resnet output shape before resizing: {resnet_output.shape}")
print(f"vgg output shape before resizing: {vgg_output.shape}")

# resize if necessary
resnet_height=resnet_output.shape[1]
resnet_width=resnet_output.shape[2]
vgg_height=vgg_output.shape[1]
vgg_width=vgg_output.shape[2]

# get the minimum height and width for resize
min_height=min(resnet_height,vgg_height)
min_width=min(resnet_width,vgg_width)

# resize resnet output
if(resnet_height != min_height or resnet_width != min_width):
  resnet_output = Resizing(height=min_height,width=min_width)(resnet_output)

# resize vgg output
if(vgg_height != min_height or vgg_width != min_width):
  vgg_output = Resizing(height=min_height,width=min_width)(vgg_output)


print(f"resnet output shape after resizing: {resnet_output.shape}")
print(f"vgg output shape after resizing: {vgg_output.shape}")



# concatenate along the channel axis
concatenated = concatenate([resnet_output, vgg_output], axis=-1)

print(f"concatenated shape:{concatenated.shape}")


# define the input and output of the whole model
# the original models have input_shape = (224,224,3), so we use that
input_tensor=Input(shape=(224,224,3))

# use the model outputs as the output of each base model
resnet_output_tensor = resnet_model(input_tensor)
vgg_output_tensor = vgg_model(input_tensor)

# example: select resnet's activation_40 output
resnet_output_tensor = resnet_model.get_layer('conv4_block6_out')(resnet_output_tensor)

# example: select vgg16's block5_conv3 output
vgg_output_tensor = vgg_model.get_layer('block5_conv3')(vgg_output_tensor)

# resize if necessary
resnet_height=resnet_output_tensor.shape[1]
resnet_width=resnet_output_tensor.shape[2]
vgg_height=vgg_output_tensor.shape[1]
vgg_width=vgg_output_tensor.shape[2]

# get the minimum height and width for resize
min_height=min(resnet_height,vgg_height)
min_width=min(resnet_width,vgg_width)

# resize resnet output
if(resnet_height != min_height or resnet_width != min_width):
  resnet_output_tensor = Resizing(height=min_height,width=min_width)(resnet_output_tensor)

# resize vgg output
if(vgg_height != min_height or vgg_width != min_width):
  vgg_output_tensor = Resizing(height=min_height,width=min_width)(vgg_output_tensor)

# concatenate along the channel axis
concatenated = concatenate([resnet_output_tensor, vgg_output_tensor], axis=-1)

# now create the full model to do the entire operation in a single forward pass

full_model = Model(inputs=input_tensor,outputs=concatenated)

# show the full model summary
full_model.summary()

```

as you see in the last snippet, the concatenation happens along the channel axis, the output of `concatenated` tensor has a new number of channels which are the combined number of channels from the tensors that you concatenated. this concatenated tensor can then be used as input to other layers. in the code i am building a full keras model that will do the resizing and concatenation in one go and this will output the concatenated tensor. there are several ways to do this, but this is just one.

to make things clear, and this is important to emphasize. you should not be creating a model by calling `concatenate([resnet_output, vgg_output], axis=-1)` directly and hoping things work by running a training loop. you must plug your inputs to the initial base models before you grab the output tensors from intermediate layers. in the code we define a `input_tensor` which is the input of the full model, and this is passed along to `resnet_model` and `vgg_model` then after that we grab the tensors `resnet_output_tensor` and `vgg_output_tensor` using the target layers. all these operations are done inside of the keras model. this will make the model trainable and will allow backpropagation to work correctly.

if i were to recommend some additional resources for a deeper dive, i’d suggest looking into papers that cover multi-modal fusion techniques, or more generally feature fusion. a good starting point would be the classic paper "deep learning with convolutional neural networks for image recognition" by krizhevsky et al., that will give you a better grasp on the nature of the convolutional feature maps you are handling. also, if you are interested in how to actually implement this in various situations, try "hands-on machine learning with scikit-learn, keras & tensorflow" by aurelien geron. it has a lot of practical examples, and i've often gone back to it to double check on the code when things got weird.

finally, i once had a bug that took me an entire day to spot. the issue was that i had renamed the layers incorrectly after copying code from the internet, the model was not training at all. it turned out that if you copy-paste layer names without verifying you are gonna have a bad time, so don't be like me. make sure your layer names and code are correct and be very precise. it is always better to double check. so before even doing anything. please print `resnet_model.summary()` and `vgg_model.summary()` and double check the layer names. its a boring but vital step.

that’s my take on this. i hope this clarifies some things. good luck with your deep learning experiments!
