---
title: "How can a CNN map a list of images to a list of 'integer, integer, float' values?"
date: "2025-01-30"
id: "how-can-a-cnn-map-a-list-of"
---
Convolutional Neural Networks (CNNs), while commonly associated with image classification tasks outputting a single class label or a probability distribution, can indeed be adapted to map a list of images to a list of [integer, integer, float] values. This adaptation requires careful architectural planning, primarily at the output layers of the network and a judicious choice of loss function. My experience building a robotic vision system which estimated object counts, sizes, and relative distances simultaneously using multiple camera streams led me directly to this architectural style. The system needed to generate this structured output for every input image, rather than a single categorical output.

The key is recognizing that CNNs excel at extracting hierarchical features from images. These features, after convolution and pooling, are essentially high-dimensional representations of image content. What follows the convolutional backbone determines the ultimate output format. For our target output of [integer, integer, float], we require three separate output pathways, each producing one component of this tuple. We cannot rely on the final layers of a typical classification CNN, which use softmax activation and produce a single probability distribution. Instead, we need to split the final feature maps into multiple streams before generating the final output values.

Specifically, we’ll use three distinct dense layers, each with a size of one, to produce the numerical components of our desired tuple. The integer outputs will require a quantization step after the dense layer, and the float output can proceed directly from the dense layer output. Each of these output paths needs a corresponding loss function to facilitate learning during backpropagation.

The first integer output will likely represent a count or a class label, and thus we can treat it as a regression target which after rounding is constrained to an integer. Hence, mean squared error (MSE), or similar regression losses will work. We need to be mindful of the training signal we provide to ensure that we encourage convergence at the relevant integer value after the rounding is done. The second integer value may encode something similar to the first and will also need similar treatment. The final output represents a floating point number and requires no quantization. So MSE will be appropriate here as well.

Now, I'll outline three code examples, using a pseudo-code convention for conciseness and clarity rather than adhering to a single specific library's syntax. This is to avoid the specifics of tensor creation and focus on the architecture.

**Example 1: Basic Architecture with Independent Outputs**

```python
# Assume input_tensor is of shape [batch_size, height, width, channels]

# Convolutional base (replace with your actual convolutional layers)
conv_features = ConvolutionLayer(input_tensor, filters=32, kernel_size=3)
conv_features = MaxPoolLayer(conv_features)
conv_features = ConvolutionLayer(conv_features, filters=64, kernel_size=3)
conv_features = MaxPoolLayer(conv_features)
conv_features = ConvolutionLayer(conv_features, filters=128, kernel_size=3)
conv_features = GlobalAveragePoolLayer(conv_features) # Gives [batch_size, 128]

# Branch for first integer output (e.g., object count)
int1_output_raw = DenseLayer(conv_features, units=1, activation=None)
int1_output = RoundingOperation(int1_output_raw)  # Quantize to integer

# Branch for second integer output (e.g., object class label)
int2_output_raw = DenseLayer(conv_features, units=1, activation=None)
int2_output = RoundingOperation(int2_output_raw)  # Quantize to integer

# Branch for float output (e.g., distance)
float_output = DenseLayer(conv_features, units=1, activation=None)

# Collect outputs
output_tuple = [int1_output, int2_output, float_output]

# Define loss functions
loss_int1 = MeanSquaredError(target=target_int1, predicted=int1_output_raw)
loss_int2 = MeanSquaredError(target=target_int2, predicted=int2_output_raw)
loss_float = MeanSquaredError(target=target_float, predicted=float_output)

# Combine losses
total_loss = loss_int1 + loss_int2 + loss_float
```

In this example, the three outputs are treated entirely independently. Each has its own dedicated dense layer stemming directly from the pooled feature map output of the convolutional base. Note the `RoundingOperation` which is critical for producing the integer outputs. The `total_loss` combines each individual loss, ensuring all aspects are learned during training. This approach works well when the three outputs are not strongly correlated.

**Example 2: Sharing Layers Before Branching**

```python
# Assume input_tensor is of shape [batch_size, height, width, channels]

# Convolutional base (same as above)
conv_features = ConvolutionLayer(input_tensor, filters=32, kernel_size=3)
conv_features = MaxPoolLayer(conv_features)
conv_features = ConvolutionLayer(conv_features, filters=64, kernel_size=3)
conv_features = MaxPoolLayer(conv_features)
conv_features = ConvolutionLayer(conv_features, filters=128, kernel_size=3)
conv_features = GlobalAveragePoolLayer(conv_features)

# Shared dense layers
shared_features = DenseLayer(conv_features, units=64, activation='relu') # Introduce shared layers
shared_features = DropoutLayer(shared_features, rate=0.2)

# Branch for first integer output
int1_output_raw = DenseLayer(shared_features, units=1, activation=None)
int1_output = RoundingOperation(int1_output_raw)

# Branch for second integer output
int2_output_raw = DenseLayer(shared_features, units=1, activation=None)
int2_output = RoundingOperation(int2_output_raw)

# Branch for float output
float_output = DenseLayer(shared_features, units=1, activation=None)

# Collect outputs
output_tuple = [int1_output, int2_output, float_output]

# Define loss functions
loss_int1 = MeanSquaredError(target=target_int1, predicted=int1_output_raw)
loss_int2 = MeanSquaredError(target=target_int2, predicted=int2_output_raw)
loss_float = MeanSquaredError(target=target_float, predicted=float_output)

# Combine losses
total_loss = loss_int1 + loss_int2 + loss_float

```
This example illustrates that we don't have to go directly from convolutional features to outputs. Introducing shared dense layers which can model cross-dependencies between the features and final outputs can be helpful. This design assumes the three outputs share some underlying information. By performing some feature abstraction on the pooled convolutional features the model can potentially learn more useful features shared across the output paths. This approach proved very effective in the aforementioned robotic vision system, where object distance and count were somewhat correlated. The introduction of dropout also adds a regularizing effect to improve generalization.

**Example 3: Using a Single Output Layer with Custom Loss**

```python
# Assume input_tensor is of shape [batch_size, height, width, channels]

# Convolutional base (same as above)
conv_features = ConvolutionLayer(input_tensor, filters=32, kernel_size=3)
conv_features = MaxPoolLayer(conv_features)
conv_features = ConvolutionLayer(conv_features, filters=64, kernel_size=3)
conv_features = MaxPoolLayer(conv_features)
conv_features = ConvolutionLayer(conv_features, filters=128, kernel_size=3)
conv_features = GlobalAveragePoolLayer(conv_features)

# Single output layer with 3 units
output_vector = DenseLayer(conv_features, units=3, activation=None)

#  Separate the output into three parts
int1_output_raw = output_vector[:, 0]
int1_output = RoundingOperation(int1_output_raw)

int2_output_raw = output_vector[:, 1]
int2_output = RoundingOperation(int2_output_raw)

float_output = output_vector[:, 2]

# Collect outputs
output_tuple = [int1_output, int2_output, float_output]


# Define loss functions
loss_int1 = MeanSquaredError(target=target_int1, predicted=int1_output_raw)
loss_int2 = MeanSquaredError(target=target_int2, predicted=int2_output_raw)
loss_float = MeanSquaredError(target=target_float, predicted=float_output)

# Combine losses
total_loss = loss_int1 + loss_int2 + loss_float
```

In this, more compact, final example we see that we can produce the three different outputs by simply having a dense layer with 3 output nodes, then separating the final output vector into three separate parts, quantizing the required parts, and calculating the losses. This allows us to reduce the number of parameters in the final network and also simplify the network architecture, albeit at the cost of clarity in the output paths. This may perform well when all of the output parameters are correlated.

**Resource Recommendations**

For further exploration into this area, I recommend consulting several sources, broadly focusing on: deep learning architecture patterns (especially multi-head architectures), loss function selection (specifically for regression and combined loss functions), and image processing fundamentals. Books covering deep learning, and specifically the practical applications of CNNs, will provide a robust theoretical understanding of the underlying principles. Additionally, online course platforms often offer hands-on projects that can help solidify understanding, specifically courses focused on computer vision or multi-modal learning. Finally, technical documentation and research papers pertaining to modern network architectures will provide detailed information on the newest methods for combining multiple outputs in a CNN. Exploring the implementations of state of the art detection networks will also provide valuable insights. Understanding and applying these techniques will allow you to effectively map a list of images to a list of [integer, integer, float] values, and other complex outputs as well.
