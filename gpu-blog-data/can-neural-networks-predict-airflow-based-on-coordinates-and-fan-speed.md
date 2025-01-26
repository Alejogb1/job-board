---
title: "Can neural networks predict airflow based on coordinates and fan speed?"
date: "2025-01-26"
id: "can-neural-networks-predict-airflow-based-on-coordinates-and-fan-speed"
---

The feasibility of employing neural networks to predict airflow given coordinate positions and fan speed is strongly indicated by the underlying physics of fluid dynamics, which are governed by consistent relationships amenable to function approximation. Having spent considerable time developing numerical simulation models for HVAC systems and control algorithms, I've encountered similar modeling challenges and observed that neural networks provide a viable alternative to computationally expensive finite element methods for approximating such complex relationships.

Fundamentally, the problem involves mapping a multi-dimensional input space (x, y, z coordinates and fan speed) to a vector representing airflow (typically a magnitude and direction, or component vectors). The complex interplay of fluid behavior – influenced by boundary conditions like the fan geometry, surrounding environment, and air properties – creates a non-linear relationship suitable for modeling with a neural network. The network learns to implicitly capture these interactions from training data generated from simulations or empirical measurements.

The core challenge is not the inherent theoretical possibility, but rather the practical considerations of data acquisition, network architecture, and training efficacy. Neural networks, particularly feedforward types like Multilayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs), are capable of function approximation for high-dimensional inputs, if properly configured and trained. The effectiveness hinges on providing sufficient, representative training data covering the full operational parameter space. Insufficient data can lead to overfitting or generalization failures outside the training regime.

The chosen architecture and hyperparameters of the network also critically affect performance. Simple MLPs with fully connected layers may suffice for basic scenarios with limited spatial variation in airflow, but complex flows often benefit from the use of CNNs or more specialized architectures like graph neural networks (GNNs). These architectures incorporate inductive biases appropriate for handling spatial data. CNNs, for example, are inherently translationally invariant and can learn patterns that recur at various spatial locations, a useful property for fluid flow modeling, while GNNs are efficient at capturing the relationships between points in a given space.

Consider a case where we want to model airflow in a rectangular room from a single fan. The inputs could be defined as (x,y,z, fan_speed) and the output could be the x,y,z components of the air velocity.

Here's an example using a simple MLP in Python with TensorFlow/Keras, focusing on clarity:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Define the model architecture
def build_mlp_model(input_shape, output_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(output_shape)  # Linear output for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# 2. Generate synthetic training data for example
def generate_training_data(num_samples):
    # Input range for x,y,z: [0,1], fan_speed [0,5]
    X = np.random.rand(num_samples, 4)  # x,y,z, fan_speed
    X[:,3] = X[:,3]*5 # Rescale fan_speed
    y = np.zeros((num_samples, 3)) #  velocity x,y,z
    # A simple approximation function for example:
    for i in range(num_samples):
       y[i,0] = 0.5*X[i,3] * (0.5-X[i,0]) # Velocity in x direction, based on fan_speed and x coordinate
       y[i,1] = 0.1*X[i,3] * (0.5-X[i,1]) # Velocity in y direction
       y[i,2] = 0.01*X[i,3] * (0.5-X[i,2]) # Velocity in z direction
    return X, y

# 3. Train and evaluate the model
input_shape = (4,) # x,y,z, fan_speed
output_shape = 3 # vx, vy, vz
num_samples = 10000
X_train, y_train = generate_training_data(num_samples)
model = build_mlp_model(input_shape, output_shape)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)


# 4. Make prediction
new_point = np.array([[0.1, 0.2, 0.3, 2]]) # Test point: x,y,z, fan_speed
predicted_velocity = model.predict(new_point)
print("Predicted airflow vector (vx, vy, vz):",predicted_velocity)
```

This example showcases a basic workflow, but requires several improvements for real-world application. Firstly, the training data here is simplistic and not generated from an accurate fluid dynamics simulator. A dataset from Computational Fluid Dynamics (CFD) simulation or from actual experimental measurements would be vital. Secondly, the MLP architecture is rudimentary. For more complex geometries or flows, experimenting with a deeper or wider MLP, or CNNs would be necessary. Finally, hyperparameters such as the learning rate, network layers, and activation functions require iterative optimization.

Here is an example of a CNN implementation to introduce spatial correlations.  This model assumes we receive the 3D position as an input to generate a sparse 3D grid where the fan speed is projected on the corresponding grid location and zero elsewhere. We then feed this sparse grid to a 3D CNN.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Define CNN model
def build_cnn_model(input_shape, output_shape):
  model = keras.models.Sequential([
      keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', input_shape=input_shape),
      keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
      keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), activation='relu'),
      keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(output_shape)
  ])
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return model


# 2. Generate synthetic training data
def generate_training_data_sparse_grid(num_samples, grid_size):
    X_coords = np.random.rand(num_samples, 3)  # x,y,z coords in [0,1]
    fan_speeds = np.random.rand(num_samples) * 5 # fan speed between 0 to 5
    X_sparse_grid = np.zeros((num_samples, grid_size, grid_size, grid_size, 1)) #Initialize
    for i in range(num_samples):
        coords_idx = np.floor(X_coords[i]*grid_size).astype(int) # Convert [0,1] to grid indices
        X_sparse_grid[i,coords_idx[0],coords_idx[1],coords_idx[2],0] = fan_speeds[i] #Put fan speed at position
    y = np.zeros((num_samples, 3))  # velocity x,y,z
    # Simple function approximation
    for i in range(num_samples):
        y[i,0] = 0.5 * fan_speeds[i] * (0.5 - X_coords[i,0])
        y[i,1] = 0.1 * fan_speeds[i] * (0.5 - X_coords[i,1])
        y[i,2] = 0.01 * fan_speeds[i] * (0.5-X_coords[i,2])
    return X_sparse_grid, y

# 3. Train and evaluate the model
grid_size = 16 #Grid dimensions
input_shape = (grid_size, grid_size, grid_size, 1)
output_shape = 3
num_samples = 1000
X_train, y_train = generate_training_data_sparse_grid(num_samples, grid_size)
model = build_cnn_model(input_shape, output_shape)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# 4. Make prediction with a new input
new_coord = np.array([[0.1, 0.2, 0.3]])
new_fan_speed = np.array([2])
new_grid = np.zeros((1, grid_size, grid_size, grid_size, 1))
new_coord_idx = np.floor(new_coord*grid_size).astype(int)
new_grid[0,new_coord_idx[0,0],new_coord_idx[0,1],new_coord_idx[0,2],0] = new_fan_speed
predicted_velocity = model.predict(new_grid)
print("Predicted airflow vector (vx, vy, vz):", predicted_velocity)

```

This CNN example assumes the position data is already transformed to a spatial grid. Data pre-processing is important because neural networks can't use position values directly, thus an appropriate conversion is necessary. The position to a grid is a basic method to convert coordinate points to a spatially meaningful input. Depending on the use case, other approaches to use the coordinates and fan speed together may be more suitable.

Finally, a recurrent neural network (RNN) example demonstrates how to incorporate a time series to take into account the effects of temporal changes in the fan speed. Although the airflow at a certain point in time is influenced by the present speed, the past speed may also matter for capturing transient effects.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Define the RNN model
def build_rnn_model(input_shape, output_shape):
    model = keras.models.Sequential([
        keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences = True),
        keras.layers.LSTM(64, activation='relu', return_sequences = False),
        keras.layers.Dense(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# 2. Generate synthetic time series data
def generate_training_data_timeseries(num_samples, seq_length):
   X_coords = np.random.rand(num_samples, 3)  # x,y,z coordinates, static
   fan_speeds = np.random.rand(num_samples, seq_length) * 5 #Fan speeds, time sequence
   X = np.zeros((num_samples, seq_length, 4)) # Combine coordinates and speed for time sequence
   for i in range(num_samples):
    for t in range(seq_length):
        X[i,t,0:3]=X_coords[i]
        X[i,t,3] = fan_speeds[i,t]
   y = np.zeros((num_samples, 3)) #  velocity x,y,z
   # Simple approximation function
   for i in range(num_samples):
       y[i,0] = 0.5*np.mean(fan_speeds[i,:]) * (0.5-X_coords[i,0]) # Velocity in x
       y[i,1] = 0.1*np.mean(fan_speeds[i,:]) * (0.5-X_coords[i,1]) # Velocity in y
       y[i,2] = 0.01*np.mean(fan_speeds[i,:]) * (0.5-X_coords[i,2])# Velocity in z
   return X, y

# 3. Train and evaluate the model
input_shape = (None, 4) # Time series dimension needs to be variable
output_shape = 3
seq_length = 10 #Time steps
num_samples = 10000
X_train, y_train = generate_training_data_timeseries(num_samples, seq_length)
model = build_rnn_model(input_shape, output_shape)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose = 0)


# 4. Make prediction with a new time sequence
new_coord = np.array([[0.1, 0.2, 0.3]]) #New xyz coordinates
new_fan_speeds = np.random.rand(1,seq_length) * 5 #Fan speeds, time sequence
new_X = np.zeros((1, seq_length, 4))
for t in range(seq_length):
   new_X[0,t,0:3]=new_coord
   new_X[0,t,3] = new_fan_speeds[0,t]
predicted_velocity = model.predict(new_X)
print("Predicted airflow vector (vx, vy, vz):", predicted_velocity)

```
This last example shows how to include the time factor in predicting the airflow. Note that while the model input includes the time history of the fan speed, the final output is a single vector of air velocity. One limitation with this approach is that it assumes the position is fixed during the time sequence. If the position were to vary, this must be incorporated into the network’s architecture. The choice between RNNs and other architectures depends on the specific nature of the flow and the desired level of temporal modeling.

In conclusion, predicting airflow from coordinates and fan speed using neural networks is achievable, provided sufficient high-quality training data and a properly chosen model architecture. The examples serve as a starting point; further research, and rigorous experimentation are crucial for developing robust solutions. I would recommend focusing on resources on numerical methods for fluid dynamics, deep learning fundamentals, and practical guidance in TensorFlow or PyTorch. Books focused on fluid simulations, such as those employing CFD, along with specialized tutorials on recurrent and graph neural networks, would greatly assist in building more complex and accurate models. Finally, careful consideration of model evaluation metrics, including both accuracy and generalization capability, must be a priority in any practical implementation.
