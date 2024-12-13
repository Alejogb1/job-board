---
title: "polynomial regression torch neural network?"
date: "2024-12-13"
id: "polynomial-regression-torch-neural-network"
---

Okay so you wanna fit a polynomial using PyTorch and a neural net eh I've been down this road more times than I can count like way way back when I was still rocking a Pentium III and coding on a CRT monitor so let's get into it

First things first yeah you can totally do polynomial regression with a neural network it might seem like overkill I know since polynomial regression is a classic example of linear algebra done right but this way we can get practice making and fitting a very simple network I mean we could just use sklearn's polynomial features and a linear regression model but where is the fun in that

The key is that your neural network only needs to model the relationship between input `x` and output `y` which has a polynomial relationship If we have our input feature as just x we need to make a network capable of learning a polynomial like

`y = a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n`

where `a_i` are the coefficients it has to learn You could also think of this as `y = sum(a_i * x^i for i in range(n+1))` So a simple feedforward network works fine for this kind of problem. Lets start building one

Here's some boilerplate code to create a network in PyTorch that you can adapt

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class PolynomialNetwork(nn.Module):
    def __init__(self, degree):
        super(PolynomialNetwork, self).__init__()
        self.degree = degree
        self.layers = nn.Sequential()
        # one input one output for a regression task
        self.layers.add_module("linear_1", nn.Linear(1, 100))
        self.layers.add_module("relu_1", nn.ReLU())
        self.layers.add_module("linear_2", nn.Linear(100, 100))
        self.layers.add_module("relu_2", nn.ReLU())
        self.layers.add_module("linear_out", nn.Linear(100, 1))

    def forward(self, x):
        return self.layers(x)

def generate_polynomial_data(degree, num_samples, noise_std=0.2):
    coeffs = np.random.rand(degree + 1) # random coefficients for the polynomial function
    x = np.sort(np.random.rand(num_samples) * 2 - 1).astype(np.float32) # generating x from -1 to 1
    y = np.zeros(num_samples, dtype=np.float32)
    for i, coeff in enumerate(coeffs):
      y += coeff * x**i
    y += np.random.normal(0, noise_std, num_samples).astype(np.float32)
    return x.reshape(-1, 1), y.reshape(-1, 1)

if __name__ == '__main__':
    degree = 3
    num_samples = 100
    x_train, y_train = generate_polynomial_data(degree, num_samples)
    x_tensor = torch.tensor(x_train)
    y_tensor = torch.tensor(y_train)

    model = PolynomialNetwork(degree)
    criterion = nn.MSELoss() # mean squared error regression loss
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam optimizer good defaults in general

    num_epochs = 5000
    for epoch in range(num_epochs):
      optimizer.zero_grad()
      outputs = model(x_tensor)
      loss = criterion(outputs, y_tensor)
      loss.backward()
      optimizer.step()
      if (epoch+1) % 1000 == 0:
          print(f"Epoch: {epoch+1} Loss: {loss.item():.4f}")

    with torch.no_grad():
      x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
      x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
      y_predicted = model(x_test_tensor).numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, label='Training Data')
    plt.plot(x_test, y_predicted, color='red', label='Polynomial Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Polynomial Regression using Neural Network")
    plt.show()

```

This code defines a `PolynomialNetwork` that takes in the `degree` of the polynomial you are trying to learn but it is really not used and the number of layers and hidden units used can be varied according to your needs The network consists of a couple linear layers with relu activation in between and outputs one scalar since it is a regression task The `generate_polynomial_data` function generates some random data that follows an `x^n` polynomial pattern for training and for testing. It trains the network and then plots the results. This plot should show the predicted polynomial curve fitting pretty well with the training data.

Now for the fun part you might be wondering why use a neural net for this when a polynomial equation or regression would work We have to think about situations where a simple polynomial will not cut it So here is where neural nets shine you can make the architecture bigger or create custom architecture to handle complex data that a simple equation would not be able to handle for example if we have a non-linear transformation to `x` before calculating the output `y` a simple linear regression will fail so here is an example where we can use this simple example to show how we might extend it for more complex functions

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ComplexPolynomialNetwork(nn.Module):
    def __init__(self, degree):
        super(ComplexPolynomialNetwork, self).__init__()
        self.degree = degree
        self.layers = nn.Sequential()
        # one input one output for a regression task
        self.layers.add_module("linear_1", nn.Linear(1, 100))
        self.layers.add_module("relu_1", nn.ReLU())
        self.layers.add_module("linear_2", nn.Linear(100, 100))
        self.layers.add_module("relu_2", nn.ReLU())
        self.layers.add_module("linear_out", nn.Linear(100, 1))

    def forward(self, x):
      # non linear transformation of x before calculating y
      x = torch.sin(x * 3.0)
      x = torch.tanh(x*2)
      return self.layers(x)

def generate_complex_polynomial_data(degree, num_samples, noise_std=0.2):
    coeffs = np.random.rand(degree + 1) # random coefficients for the polynomial function
    x = np.sort(np.random.rand(num_samples) * 2 - 1).astype(np.float32) # generating x from -1 to 1
    y = np.zeros(num_samples, dtype=np.float32)
    for i, coeff in enumerate(coeffs):
      y += coeff * np.sin(x * 3)**i
    y += np.random.normal(0, noise_std, num_samples).astype(np.float32)
    return x.reshape(-1, 1), y.reshape(-1, 1)

if __name__ == '__main__':
    degree = 3
    num_samples = 100
    x_train, y_train = generate_complex_polynomial_data(degree, num_samples)
    x_tensor = torch.tensor(x_train)
    y_tensor = torch.tensor(y_train)

    model = ComplexPolynomialNetwork(degree)
    criterion = nn.MSELoss() # mean squared error regression loss
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam optimizer good defaults in general

    num_epochs = 5000
    for epoch in range(num_epochs):
      optimizer.zero_grad()
      outputs = model(x_tensor)
      loss = criterion(outputs, y_tensor)
      loss.backward()
      optimizer.step()
      if (epoch+1) % 1000 == 0:
          print(f"Epoch: {epoch+1} Loss: {loss.item():.4f}")

    with torch.no_grad():
      x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
      x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
      y_predicted = model(x_test_tensor).numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, label='Training Data')
    plt.plot(x_test, y_predicted, color='red', label='Polynomial Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Complex Polynomial Regression using Neural Network")
    plt.show()
```

Here I modified the model so that it will now take `x` apply a non linear `sin` and `tanh` transformation and use the transformed variable to train the network We also changed the way the data is generated such that it is also now non linear So we can see that now a simple regression model will not work here But a neural network with some extra layers is able to approximate this non-linearity.

One important note and this is something that tripped me a long time ago When you are training you should always normalize your inputs and outputs this will improve the learning process and if you have not done this before I suggest you implement a quick min max normalization that can easily be implemented in this example Here is the function

```python
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
```

So you would normalize your `x_train` and `y_train` with this and use it during training and then reverse this process for predictions

If you are going deeper into this topic I would suggest checking out some good material first like the deep learning book by goodfellow I have it sitting on my shelf in paper format but I am sure there is a digital version around and it is a very good read you should also check out the mathematical theory behind neural networks because it might be daunting to do it but it will allow you to build deeper intuitions about these things I also recommend looking into the original Adam paper it is very well explained and a very powerful optimizer And if you want to go all the way back I would suggest reading about the back propagation algorithm from Rumelhart et al

You also asked about the best architecture for this well there isn't a universal answer it really depends on your data I once spent 3 full weeks experimenting with different layer sizes and activations on a similarly sounding project and finally I figured out that a simpler architecture worked better. My biggest recommendation is to start simple then slowly iterate and make it more complex. So don't just go for it and start adding convolutions and residual connections try simple linear layers first. You'll be surprised what simple things you can achieve with simple models. Don't overcomplicate it. It's like the old saying "why is six afraid of seven" you will find it when you learn the basics first (get it because 7 8 9)

So to wrap up yes you can use neural nets for polynomial regression and it is a good way to learn to work with them and they are useful for complex non-linear functions also make sure you normalize your data and start with something simple and then scale up. Good luck and feel free to ask if anything else comes up.
