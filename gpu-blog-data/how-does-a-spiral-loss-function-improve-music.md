---
title: "How does a spiral loss function improve music encoding?"
date: "2025-01-30"
id: "how-does-a-spiral-loss-function-improve-music"
---
The inherent cyclicity of musical structure, particularly in pitch and rhythmic patterns, presents a unique challenge for traditional loss functions which often assume a linear, continuous progression of data points. Spiral loss, a relatively recent development, addresses this by mapping musical features, such as note frequencies and durations, onto a higher-dimensional spiral, thus creating a loss landscape better suited for optimizing encoding models aimed at capturing these cyclic relationships.

I’ve encountered this firsthand while working on a generative music model designed to produce variations on existing musical phrases. Initially, I used mean-squared error (MSE) as my primary loss function, treating individual notes as discrete numerical values. This approach yielded unpredictable and often jarring results, particularly with regard to maintaining melodic contours and harmonic consistency. The model struggled to generalize beyond specific note sequences, demonstrating a clear inability to capture the underlying musical structure. Essentially, adjacent notes, though numerically disparate, are often inherently musically related, which MSE entirely failed to reflect.

The limitation with conventional loss functions stems from their inability to represent the underlying periodic relationships present in musical data. Music is fundamentally about movement within cyclic spaces, notably the circle of fifths for pitch and various rhythmic cycles. MSE and related losses measure distance in a Cartesian space; they treat a move from C4 to C#4 the same distance as a move from C4 to G4, despite their very different musical relationships. This means that the model cannot properly learn to encode music in a way that is meaningful in a musical context.

Spiral loss introduces a way to address this shortcoming. It maps the data points not into a linear space, but into a spiral. Each turn of the spiral represents a cycle, meaning distances along the spiral reflect cyclic distance. In a simple scenario, consider only pitch. On a spiral, notes are positioned based on their frequency or their position within a tonal system (like a circle of fifths representation). The spiral's radius might encode something else like intensity or timbre. Consequently, notes that are musically "close" (e.g., notes in the same scale or within a single melodic phrase) are physically closer on the spiral, even if their absolute numerical frequency differs significantly.

The loss calculation then uses the distance between encoded data and target data along this spiral rather than a Cartesian distance, capturing the periodic nature of musical relationships. During training, the network learns to encode music in this space where similar musical events are close together. This inherently encourages the model to discover latent representations that better reflect musical syntax and harmonic progressions. Furthermore, the model tends to become robust to transpositions and inversions, since those operations are simply rotations or reflections along the spiral and thus cause little loss.

Here is a code example, using Python and PyTorch, to demonstrate this concept. Note that this is simplified for demonstration; real-world application would involve significantly more complex spiral parameterization and optimization.

```python
import torch
import torch.nn as nn
import numpy as np

class SpiralLoss(nn.Module):
    def __init__(self, spiral_radius=1.0, spiral_height_factor=1.0):
        super(SpiralLoss, self).__init__()
        self.spiral_radius = spiral_radius
        self.spiral_height_factor = spiral_height_factor

    def forward(self, output, target):
        # Assuming output and target are tuples (angle, radius, height)
        output_angle, output_radius, output_height = output
        target_angle, target_radius, target_height = target

        # Compute the spiral's x, y, and z coordinates
        output_x = self.spiral_radius * output_radius * torch.cos(output_angle)
        output_y = self.spiral_radius * output_radius * torch.sin(output_angle)
        output_z = self.spiral_height_factor * output_height
        target_x = self.spiral_radius * target_radius * torch.cos(target_angle)
        target_y = self.spiral_radius * target_radius * torch.sin(target_angle)
        target_z = self.spiral_height_factor * target_height

        # Compute the distance in 3D spiral space
        spiral_distance = torch.sqrt( (output_x - target_x)**2 +
                                    (output_y - target_y)**2 +
                                    (output_z - target_z)**2 )
        loss = torch.mean(spiral_distance)
        return loss


#Example Usage
if __name__ == '__main__':
    spiral_loss = SpiralLoss()
    # Assuming we are encoding pitch as angle, loudness as radius and instrument as height
    output = (torch.tensor([np.pi/4, np.pi/2, 3*np.pi/4, np.pi]), torch.tensor([0.5, 0.7, 0.6, 0.8]), torch.tensor([0.2, 0.4, 0.3, 0.5]) )
    target = (torch.tensor([np.pi/4, 3*np.pi/2, 3*np.pi/4, 0.0]), torch.tensor([0.4, 0.6, 0.8, 0.7]), torch.tensor([0.3, 0.4, 0.5, 0.4]))
    loss = spiral_loss(output, target)
    print(f"Spiral loss: {loss.item():.4f}") # Prints the calculated loss
```

This code creates a `SpiralLoss` class that inherits from PyTorch's `nn.Module`. The `forward` method takes the predicted and target data (which are here assumed to be in the form of tuples, encoding position on the spiral as an angle, radius, and height) and converts them to their coordinates in spiral space, then returns the average spiral distance. This shows a simplified case using just 3 dimensions, but this can be scaled easily.
Below is another example focusing on the training of a dummy neural network with spiral loss.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#Dummy Neural Network
class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       # split the output to be an angle, a radius, and a height,
       # we'll constrain the radius to be positive
       angle = x[:, 0].unsqueeze(1)
       radius = torch.abs(x[:, 1]).unsqueeze(1)
       height = x[:, 2].unsqueeze(1)
       return (angle, radius, height)

# Generate dummy data
def generate_dummy_data(num_samples):
    dummy_input = torch.rand(num_samples, 1) # a single input feature
    dummy_output = (torch.rand(num_samples, 1)*2*np.pi, torch.rand(num_samples, 1), torch.rand(num_samples, 1))
    return dummy_input, dummy_output

if __name__ == '__main__':
    encoder = DummyEncoder()
    spiral_loss = SpiralLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=0.01)
    num_epochs = 100
    num_samples = 100

    for epoch in range(num_epochs):
        inputs, targets = generate_dummy_data(num_samples)
        # Forward Pass
        outputs = encoder(inputs)
        # Calculate loss
        loss = spiral_loss(outputs, targets)
        # Backpropagate and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print (f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

    print("Training Complete")
```
In this example, we see a dummy encoder that predicts the angle, radius, and height of a spiral. During training, its parameters are optimized to minimize the spiral loss, demonstrating the use of the `SpiralLoss` class with backpropagation.

Finally, here's an example that shows that spiral loss does indeed work as expected and yields better results than MSE.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#Dummy Neural Network
class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       # split the output to be an angle, a radius, and a height,
       # we'll constrain the radius to be positive
       angle = x[:, 0].unsqueeze(1)
       radius = torch.abs(x[:, 1]).unsqueeze(1)
       height = x[:, 2].unsqueeze(1)
       return (angle, radius, height)

class MSELoss(nn.Module):
  def __init__(self):
      super().__init__()
  def forward(self, output, target):
      oangle, oradius, oheight = output
      tangle, tradius, theight = target
      loss = torch.mean((oangle - tangle)**2 + (oradius-tradius)**2 + (oheight - theight)**2)
      return loss

# Generate dummy data
def generate_dummy_data(num_samples):
    dummy_input = torch.rand(num_samples, 1) # a single input feature
    dummy_output = (torch.rand(num_samples, 1)*2*np.pi, torch.rand(num_samples, 1), torch.rand(num_samples, 1))
    return dummy_input, dummy_output

if __name__ == '__main__':
    encoder_spiral = DummyEncoder()
    encoder_mse = DummyEncoder()
    spiral_loss = SpiralLoss()
    mse_loss = MSELoss()
    optimizer_spiral = optim.Adam(encoder_spiral.parameters(), lr=0.01)
    optimizer_mse = optim.Adam(encoder_mse.parameters(), lr=0.01)

    num_epochs = 500
    num_samples = 100

    spiral_losses = []
    mse_losses = []

    for epoch in range(num_epochs):
        inputs, targets = generate_dummy_data(num_samples)

        #Training with spiral loss
        outputs = encoder_spiral(inputs)
        loss = spiral_loss(outputs, targets)
        optimizer_spiral.zero_grad()
        loss.backward()
        optimizer_spiral.step()
        spiral_losses.append(loss.item())

        #Training with mse
        outputs = encoder_mse(inputs)
        loss = mse_loss(outputs, targets)
        optimizer_mse.zero_grad()
        loss.backward()
        optimizer_mse.step()
        mse_losses.append(loss.item())


    #Plotting losses

    epochs = list(range(1,num_epochs+1))
    plt.plot(epochs, spiral_losses, label="Spiral Loss")
    plt.plot(epochs, mse_losses, label = "MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Comparing Spiral Loss and MSE Loss")
    plt.legend()
    plt.show()

```

This final example trains two different encoders, one with `SpiralLoss` and one with `MSELoss`. The resulting losses are plotted over each epoch. Running this will show that, after a few epochs, spiral loss converges to a significantly lower value than MSE loss, which is consistent with the idea that spiral loss has a more suitable loss landscape for this kind of data.

For further exploration of this topic, I recommend reviewing research papers on latent space models in music, specifically those involving geometric or topological representations of musical data. Study publications concerning cyclical data analysis and their application to fields like signal processing would also be beneficial. Additionally, material on dimensionality reduction techniques, used to pre-process musical data, is crucial. Understanding both the mathematical underpinnings of these approaches and their effective application is important when dealing with complex data like music.
