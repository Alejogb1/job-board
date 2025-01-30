---
title: "How can a trained auto-encoder's decoder be repurposed?"
date: "2025-01-30"
id: "how-can-a-trained-auto-encoders-decoder-be-repurposed"
---
Autoencoders, fundamentally designed for data compression and reconstruction, possess a decoder that can be effectively repurposed for generative tasks once training is complete. This inherent capability arises from the decoder's learned ability to map from a compressed, latent space back to the original input space. I’ve successfully leveraged this in several projects, most notably in image synthesis for a material design optimization tool. The key shift in perspective involves treating the latent vector, typically produced by the encoder, as a controllable input for the decoder, allowing it to generate new, varied outputs.

The core principle lies in the decoder’s capacity to generalize from the compressed latent representation. During autoencoder training, the encoder learns to reduce the high-dimensional input data into a lower-dimensional space (the latent space), while the decoder learns the inverse mapping, reconstructing the input from this compressed representation. Once trained, the decoder’s mapping is fixed; however, we are no longer constrained by the encoder's output. Instead, we can feed arbitrary latent vectors to the decoder, thereby generating potentially novel outputs that are within the distribution of the training data. The process relies on the decoder having captured meaningful relationships in the data through its training and being able to extrapolate from that. Therefore, the quality of the generated output is directly proportional to the richness of the trained model and the inherent structure within the latent space.

To facilitate repurposing, the nature of the latent space is important. If the latent space exhibits a smooth structure, interpolating between different latent vectors can yield meaningful transitions in the output space. This is particularly beneficial in applications like generating variations of images, sound textures, or molecular structures. Conversely, if the latent space is fragmented or unstructured, results may be less predictable or less meaningful.

Repurposing involves decoupling the encoder's input from the decoder. We bypass the encoder entirely, focusing on generating novel latent vectors and using them as inputs to the decoder. This separation gives us creative freedom but requires consideration of how we're generating these vectors. Strategies can include sampling from a pre-defined distribution (e.g., a Gaussian) or, in more sophisticated applications, using algorithms that optimize the latent space toward a desired output target.

Here are three code examples illustrating different methods of leveraging the decoder:

**Example 1: Random Latent Vector Generation**

```python
import torch
import torch.nn as nn

# Assume a trained decoder is stored in 'decoder'

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Define some placeholder parameters for demonstration
latent_dimension = 20
output_dimension = 784 # Assuming grayscale images of 28 x 28 pixels
decoder = Decoder(latent_dimension, output_dimension)


def generate_random(decoder_model, latent_size, output_shape):
    z = torch.randn(1, latent_size) # Generate a single latent vector sampled from a Gaussian distribution
    with torch.no_grad(): # No need to calculate gradients for inference
        generated_image = decoder_model(z)
    return generated_image.reshape(output_shape) # Reshape to original dimensions


output_shape_2d = (28, 28)
generated_image_tensor = generate_random(decoder, latent_dimension, output_shape_2d)

# This will produce a new image (tensor), not a reconstruction.
# We would typically then visualise generated_image_tensor as an image.
```
This first example demonstrates the most basic technique. The `generate_random` function samples a random vector from a Gaussian distribution (using `torch.randn`) and feeds it directly to the trained decoder. This results in generating new output that will likely share similar characteristics to the dataset used to train the autoencoder, but will be novel. This example also highlights the need to ensure that the output is in the correct shape after going through the decoder model which may need reshaping using `reshape`. The random latent vectors explore the latent space to create various results.

**Example 2: Latent Vector Interpolation**

```python
import torch
import torch.nn as nn

# Assume a trained decoder is stored in 'decoder'

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Define some placeholder parameters for demonstration
latent_dimension = 20
output_dimension = 784 # Assuming grayscale images of 28 x 28 pixels
decoder = Decoder(latent_dimension, output_dimension)


def interpolate(decoder_model, latent_size, output_shape, z1, z2, alpha):
    z = z1 * (1-alpha) + z2 * alpha # Linear interpolation between two vectors
    with torch.no_grad():
         generated_image = decoder_model(z)
    return generated_image.reshape(output_shape)


# Example usage
latent_vector_1 = torch.randn(1, latent_dimension)
latent_vector_2 = torch.randn(1, latent_dimension)

output_shape_2d = (28, 28)
interpolated_image_tensor = interpolate(decoder, latent_dimension, output_shape_2d, latent_vector_1, latent_vector_2, 0.5)
# The 'interpolated_image_tensor' will represent an image halfway between the images corresponding to the vectors 'latent_vector_1' and 'latent_vector_2'.

# Further samples with varying alpha values can be visualised
# example:
# interpolated_image_tensor_2 = interpolate(decoder, latent_dimension, output_shape_2d, latent_vector_1, latent_vector_2, 0.2)
# interpolated_image_tensor_3 = interpolate(decoder, latent_dimension, output_shape_2d, latent_vector_1, latent_vector_2, 0.8)
```
This example demonstrates latent vector interpolation. Instead of solely generating random vectors, we create two random latent vectors and use linear interpolation to create a vector between them. This allows for a smoother transition between data samples that are encoded in the latent space. The `alpha` parameter controls the degree of interpolation. This can generate smooth transitions between data samples that the network has been trained on, creating a blending of the data points. This method relies on the latent space being relatively smooth.

**Example 3: Latent Space Optimization**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume a trained decoder is stored in 'decoder'

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Define some placeholder parameters for demonstration
latent_dimension = 20
output_dimension = 784 # Assuming grayscale images of 28 x 28 pixels
decoder = Decoder(latent_dimension, output_dimension)


def optimize_latent_vector(decoder_model, latent_size, output_shape, target_vector, learning_rate=0.01, iterations=1000):

    latent_z = torch.randn(1, latent_size, requires_grad=True) # Start with a random latent vector

    optimizer = optim.Adam([latent_z], lr = learning_rate) # Optimiser for the latent vector

    for i in range(iterations):
        optimizer.zero_grad() # Ensure gradients are cleared at start of each iteration
        generated_output = decoder_model(latent_z)
        loss = nn.MSELoss()(generated_output.reshape(output_shape), target_vector) # Calculate the loss based on how far generated output is from the target
        loss.backward()  # Backpropagate loss to the latent vector
        optimizer.step() # Update latent vector to minimise loss

    with torch.no_grad(): # We do not want to calculate gradients for the final generation of data
         generated_image = decoder_model(latent_z)
    return generated_image.reshape(output_shape)

target_image = torch.rand(output_shape) # Define a random target as an example, the target could be some other criteria.

output_shape_2d = (28, 28)
optimized_image_tensor = optimize_latent_vector(decoder, latent_dimension, output_shape_2d, target_image)

# The 'optimized_image_tensor' will represent an image that is as close as possible to target_image according to the decoder.
```
This final example presents latent space optimization. It iteratively adjusts the latent vector to generate an output that aligns with a specific target output, unlike the random approach in the first example. This is achieved by using backpropagation to minimise a loss function that measures the distance between the decoder output and the target. We start with a random latent vector, which is then optimised using an optimiser. This enables exploration and generation in a more targeted way. The loss function may be varied to have more complex targets, such as perceptual losses that consider high level features. This could be used when it is desired to produce something that is as close to some target image as is possible.

For further information on autoencoders and latent space manipulations, several excellent resources are available. I found "Deep Learning with Python" by François Chollet, especially beneficial for providing a foundational understanding. Additionally, "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron gives a practical, example-driven perspective. Several reputable online courses offered by university platforms also delve into the specific application of autoencoders in generative modeling. Exploration of papers on variational autoencoders (VAEs) further extends the range of potential applications.
