---
title: "How can I plot contours of PyTorch functions?"
date: "2025-01-30"
id: "how-can-i-plot-contours-of-pytorch-functions"
---
Creating contour plots of PyTorch functions requires a careful consideration of how the function is evaluated and subsequently visualized. The core challenge arises because PyTorch functions, often part of a larger neural network, are designed to operate on tensors, not necessarily on the gridded data needed for contour plotting. Over my years developing numerical optimization routines, I’ve found the approach is generally to generate a grid, evaluate the function at each point, and then feed this data into a plotting library.

First, we must establish a domain upon which to evaluate the target PyTorch function. This domain is typically a 2D space defined by two axes, x and y. I've commonly used `torch.linspace` to generate equally spaced points along these axes, thereby producing a grid of coordinate pairs. Then, for each coordinate pair (x, y), the PyTorch function needs to be evaluated. It's essential to ensure that the input to the PyTorch function is a tensor with the correct shape and datatype – usually a float tensor, and that the output is scalar. This can sometimes mean reshaping or processing the output of a PyTorch model to obtain a single value for each input coordinate. The result of this evaluation will be a 2D tensor representing the function's values over the grid. This tensor can then be used by plotting libraries, such as Matplotlib, to generate the contour plot.

Here are three code examples with commentary, which should provide a practical guide:

**Example 1: Contour of a Simple Quadratic Function**

In this first example, I will demonstrate how to visualize a simple quadratic function. This is a good starting point because the mathematical form is well-understood.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def quadratic_function(x, y):
    return x**2 + y**2

# Define the grid range and number of points
x_range = torch.linspace(-5, 5, 100)
y_range = torch.linspace(-5, 5, 100)

# Create a meshgrid for evaluation
x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='ij')

# Evaluate the function over the grid
z_values = quadratic_function(x_grid, y_grid)

# Convert torch tensors to NumPy for plotting
x_np = x_grid.numpy()
y_np = y_grid.numpy()
z_np = z_values.numpy()

# Generate the contour plot
plt.contour(x_np, y_np, z_np, levels=20)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot of Quadratic Function")
plt.colorbar()
plt.show()
```

The code initially defines the `quadratic_function`. Then, `torch.linspace` establishes the range and number of points for both the x and y axes. The critical step is `torch.meshgrid`, which generates the x and y coordinate matrices. The function is then evaluated on these grid points, creating the `z_values` tensor. Finally, the data is converted to NumPy arrays since `matplotlib.pyplot.contour` expects NumPy arrays for input. The number of levels parameter in the contour plot dictates the number of contour lines to draw. A color bar is useful for interpretating these levels. This approach allows for the visualization of the simple paraboloid shape resulting from x^2+y^2.

**Example 2: Contour of a Function Involving Trigonometric Elements**

This example moves to a function which introduces some oscillations using trigonometric functions, adding a layer of complexity to the visualization.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def trig_function(x, y):
    return torch.cos(x) * torch.sin(y)

# Define grid ranges
x_range = torch.linspace(-5, 5, 100)
y_range = torch.linspace(-5, 5, 100)

# Generate meshgrid
x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='ij')

# Evaluate the function on the grid
z_values = trig_function(x_grid, y_grid)

# Convert tensors to NumPy
x_np = x_grid.numpy()
y_np = y_grid.numpy()
z_np = z_values.numpy()

# Generate the contour plot
plt.contourf(x_np, y_np, z_np, levels=20, cmap='viridis')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot of Trigonometric Function")
plt.colorbar()
plt.show()
```

Here, the structure of the code is nearly identical to the first example. However, the function being evaluated, `trig_function`, incorporates trigonometric components – cosine and sine. The change from `plt.contour` to `plt.contourf` is deliberate: it fills the area between the contour lines, thereby generating a filled contour plot for better visual perception of function variations. The cmap parameter selects the color scheme of the filled contour. Notice how the values now exhibit oscillatory patterns. This highlights how the grid method efficiently handles functions of varying complexity.

**Example 3: Contours From a Small PyTorch Model**

This final example showcases how to visualize the output of a small, dummy PyTorch model. This is particularly relevant when considering the visualization of activations within a neural network or training losses.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Dummy PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = SimpleModel()

# Define function to evaluate the model
def model_output(x, y):
    input_tensor = torch.tensor([[x,y]], dtype=torch.float32)
    output = model(input_tensor)
    return output.item()

# Define grid ranges
x_range = torch.linspace(-5, 5, 100)
y_range = torch.linspace(-5, 5, 100)

# Generate meshgrid
x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='ij')

# Evaluate the model on the grid
z_values = torch.empty_like(x_grid, dtype=torch.float32)
for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        z_values[i, j] = model_output(x_grid[i,j], y_grid[i,j])

# Convert tensors to NumPy
x_np = x_grid.numpy()
y_np = y_grid.numpy()
z_np = z_values.numpy()

# Generate the contour plot
plt.contour(x_np, y_np, z_np, levels=20)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot of Model Output")
plt.colorbar()
plt.show()
```

Here, a simple PyTorch model `SimpleModel`, consisting of a single linear layer, is defined. To evaluate the model over the grid, a helper function `model_output` is introduced, which takes x and y coordinates, converts them to a tensor, passes them through the model, and extracts the scalar output. The key change from the previous examples lies in the manner of evaluating the function. A double loop is necessary to evaluate each combination from the meshgrid individually because the model’s forward method expects a tensor of a specific shape. Consequently, the output tensor `z_values` is generated iteratively. The output shows the contour lines that result from a linear transformation of the input domain. This can be a simple example of seeing how the model is behaving across input space.

For further study, I suggest familiarizing yourself with the core concepts of PyTorch tensors, especially meshgrid and its implications. Resources detailing the `torch.autograd` module can be useful if you intend to visualize gradients of a function. Books or websites dedicated to the Matplotlib library will explain more about customization options for the contour plots, as I’ve only covered the very basics. Finally, a good understanding of numerical optimization is useful for interpreting the plots, particularly when trying to find minima or saddle points. I've found these concepts, when carefully approached, to create a very effective method for visualizing PyTorch function landscapes.
