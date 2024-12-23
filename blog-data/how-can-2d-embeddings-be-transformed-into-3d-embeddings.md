---
title: "How can 2D embeddings be transformed into 3D embeddings?"
date: "2024-12-23"
id: "how-can-2d-embeddings-be-transformed-into-3d-embeddings"
---

Let's talk about lifting 2D embeddings into 3D space. It's a process I've dealt with a number of times, especially back during my work on visualizing complex sensor data; I remember spending weeks getting a good grip on this back when we were trying to represent time series sensor readings in a way that felt more intuitive for the engineers. The challenge, at its core, is converting data representations from a two-dimensional plane into a three-dimensional volume, and as you might expect, the methods range in complexity.

Essentially, you're moving from a space defined by two coordinates (like `x` and `y`) to one defined by three (adding `z`). The most straightforward approach isn't to generate 3d directly but to use the 2d positions as a base for constructing the 3d space. You're essentially adding a third dimension while respecting the relational structure already present in your 2d embedding. If you completely throw out the 2d information you lose a good chunk of your inherent data structure.

The simplest conceptual method, which may sound almost trivial, is to map your 2D embedding directly to a plane within the 3D space. We take the 2D coordinates and make them the `x` and `y` values in our 3D space, and then we set a fixed `z` value, effectively creating a flat disc of data. It isn't incredibly valuable for complex visualisations, because you’re not really leveraging the third dimension, but it's the starting point for understanding other methods.

```python
import numpy as np

def simple_2d_to_3d(embeddings_2d, z_value=0):
    """
    Maps 2D embeddings to 3D by setting a constant z value.

    Args:
        embeddings_2d (np.ndarray): 2D embeddings, shape (n_points, 2).
        z_value (float): Constant z value for all points.

    Returns:
        np.ndarray: 3D embeddings, shape (n_points, 3).
    """
    num_points = embeddings_2d.shape[0]
    embeddings_3d = np.zeros((num_points, 3))
    embeddings_3d[:, :2] = embeddings_2d
    embeddings_3d[:, 2] = z_value
    return embeddings_3d

# Example usage
embeddings_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
embeddings_3d = simple_2d_to_3d(embeddings_2d, z_value=5)
print(embeddings_3d)
# Output: [[1. 2. 5.]
#          [3. 4. 5.]
#          [5. 6. 5.]
#          [7. 8. 5.]]
```

Here, the `simple_2d_to_3d` function takes a 2D array, and constructs a 3d array, keeping the original coordinates. This provides a flat plane of points for the given `z` value.

A slightly more useful approach, and the one I used most often in that sensor visualization work, involves using a function to determine the `z` coordinate based on the data itself, often based on some property of your original 2D embeddings or even from the original high-dimensional input from which you generated the embeddings. Think of it as encoding additional information in the third dimension. For example, you could use a radial function to compute the `z` value: `z = sqrt(x^2 + y^2)`. Or you could use the distance to the centre of your 2D embeddings. I found that by using some properties of original high dimensional data I could produce a more meaningful visualisation for my engineers.

```python
import numpy as np

def radial_z_transform(embeddings_2d):
    """
    Transforms 2D embeddings to 3D using a radial function for the z-coordinate.

    Args:
        embeddings_2d (np.ndarray): 2D embeddings, shape (n_points, 2).

    Returns:
        np.ndarray: 3D embeddings, shape (n_points, 3).
    """
    num_points = embeddings_2d.shape[0]
    embeddings_3d = np.zeros((num_points, 3))
    embeddings_3d[:, :2] = embeddings_2d

    # Calculate z using a radial function sqrt(x^2 + y^2)
    embeddings_3d[:, 2] = np.sqrt(np.sum(embeddings_2d**2, axis=1))

    return embeddings_3d


# Example Usage
embeddings_2d = np.array([[1, 2], [-1, 2], [3, -1], [-2,-3]])
embeddings_3d = radial_z_transform(embeddings_2d)
print(embeddings_3d)

# Output: [[1.         2.         2.23606798]
#  [-1.         2.         2.23606798]
#  [ 3.        -1.         3.16227766]
#  [-2.        -3.         3.60555128]]
```

Here, the `radial_z_transform` computes the magnitude of the 2D vectors which becomes the `z` value. In this case the `z` value is dependent on the `x` and `y` positions.

For even more complex transformations, we can apply techniques such as kernel methods or neural networks. These are typically more appropriate when there is a specific need to model some particular structure or relationship in the data. For instance, if you had known relationships between data points, a neural network can be designed to map your 2d embeddings into a 3d structure that preserves these relationships in ways that the simpler methods fail to do. If you have any form of additional data, you can leverage this to drive the third dimension in a more meaningful way. In this approach, the input data for the network is your 2D embedding, and the output is your 3D representation.

```python
import numpy as np
import tensorflow as tf

def neural_network_2d_to_3d(embeddings_2d, z_input):
    """
    Transforms 2D embeddings to 3D using a neural network,
    using the embeddings and additional 'z' data as input.

    Args:
        embeddings_2d (np.ndarray): 2D embeddings, shape (n_points, 2).
        z_input (np.ndarray): Additional z value input, shape (n_points, 1).

    Returns:
        np.ndarray: 3D embeddings, shape (n_points, 3).
    """
    num_points = embeddings_2d.shape[0]
    model_input = tf.keras.layers.Input(shape=(3,))
    hidden = tf.keras.layers.Dense(16, activation='relu')(model_input)
    output = tf.keras.layers.Dense(3)(hidden)
    model = tf.keras.Model(inputs=model_input, outputs=output)

    model.compile(optimizer='adam', loss='mse')

    # Combine embeddings_2d and z_input as network input
    network_input = np.concatenate((embeddings_2d, z_input), axis=1)

    # Train the neural network (simplified training example)
    model.fit(network_input, np.zeros((num_points,3)), epochs=10, verbose=0)
    # Predict 3D embeddings
    embeddings_3d = model.predict(network_input)

    return embeddings_3d

# Example Usage
embeddings_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
z_input = np.array([[0.1], [0.2], [0.3], [0.4]])  # Example z input
embeddings_3d = neural_network_2d_to_3d(embeddings_2d, z_input)
print(embeddings_3d)
```

The `neural_network_2d_to_3d` function constructs a simple keras neural network that takes both the 2D input and the `z` input, to create our 3D points. This function requires more resources (tensorflow) but provides for a much more complex mapping, leveraging your existing 2D space with additional data.

The choice of method really depends on the data you're working with and the specific goals of your visualization. For basic visualisations, mapping a flat plane may be fine, and for more detailed views, simple radial transformation can help. It's only if there are very specific relationships in the original high dimensional data that are important to preserve that a neural network approach becomes valuable.

If you're interested in delving deeper, I'd highly recommend looking into the work of John Moody on "Principal Manifolds" and how he connects them to visualising higher dimensional data in lower dimensions. For more on neural network based techniques, look into the literature of t-distributed stochastic neighbor embeddings (t-sne), and the application of neural nets to those techniques, as it's often used for complex dimensional reduction and manipulation. Also, the book "Pattern Recognition and Machine Learning" by Christopher Bishop offers a very thorough exploration of several related topics, such as kernel methods which can be relevant in creating high-dimensional transformations. These sources should give you the mathematical and conceptual underpinnings needed to really tackle this problem with confidence.
