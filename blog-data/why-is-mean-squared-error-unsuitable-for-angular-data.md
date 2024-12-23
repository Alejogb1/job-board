---
title: "Why is mean squared error unsuitable for angular data?"
date: "2024-12-23"
id: "why-is-mean-squared-error-unsuitable-for-angular-data"
---

Let's dive into the intricacies of error metrics, specifically why mean squared error (mse) falls short when dealing with angular data. It's a topic I've encountered quite frequently in my work, particularly when developing algorithms for orientation estimation using sensor data. In one particular project, we were tracking the movement of a robotic arm, and the initial use of mse on the angular components led to perplexing results. It was a classic lesson in selecting the appropriate metric for the task at hand.

The core issue lies in how mse calculates error. It measures the average squared difference between predicted and actual values. Mathematically, this is represented as:

mse = (1/n) * Σ(predicted_i - actual_i)^2

where n is the number of data points.

This equation works remarkably well for linear or scalar values. If the predicted value is close to the actual, the squared difference is small, and mse reflects this accuracy. However, angular data, such as angles represented in degrees or radians, introduces a crucial concept: circularity. Angles wrap around. For instance, 359 degrees is essentially the same direction as 1 degree; they are only 2 degrees apart, not 358. The mse, however, treats 359 and 1 as if they are significantly different. This misinterpretation stems from the fact that mse assumes a linear scale, and it penalizes any large numerical difference, irrespective of the actual directional proximity on a circle or sphere.

To illustrate this, imagine two angles: 350 degrees and 10 degrees. Intuitively, these are close, separated by 20 degrees along the smaller arc. If we were dealing with linear data, the mse would correctly reflect the difference (340 units), but here, it would incorrectly quantify this as a very large error because (350 - 10)^2 = 115,600. A more appropriate difference should take into account that 350 degrees is, in fact, "close" to 10 degrees. Squaring this incorrect difference magnifies the problem further.

Here’s a practical example using python code to demonstrate the issue:

```python
import numpy as np

def calculate_mse(predictions, actuals):
  """Calculates the mean squared error."""
  predictions = np.array(predictions)
  actuals = np.array(actuals)
  return np.mean((predictions - actuals)**2)

# Problematic example using naive angle difference
predictions = np.array([350, 20, 170])
actuals = np.array([10, 25, 160])

naive_mse = calculate_mse(predictions, actuals)
print(f"Naive MSE: {naive_mse}") # High error despite close angular proximity
```

In this code, the naive mse is artificially high due to the linear difference calculation, masking the fact that two out of three pairs of angles are very close.

A better way to deal with the circular nature of angular data is to calculate the *angular difference*, considering the shortest arc on the circle. We can achieve this by considering the difference between two angles modulo 360 degrees (or 2π radians).

Let’s extend our python example to show how angular differences can be accurately calculated and used in an mse calculation:

```python
import numpy as np

def calculate_angular_difference(predictions, actuals, degrees=True):
  """Calculates the shortest angular difference (modulo 360 or 2pi)."""
  predictions = np.array(predictions)
  actuals = np.array(actuals)

  if degrees:
    diff = (predictions - actuals) % 360
    diff = np.abs((diff + 180) % 360 - 180) # Shortest arc
  else:
    diff = (predictions - actuals) % (2 * np.pi)
    diff = np.abs((diff + np.pi) % (2*np.pi) - np.pi)

  return diff


def calculate_angular_mse(predictions, actuals, degrees=True):
    """Calculates the mean squared error using angular differences."""
    angular_diff = calculate_angular_difference(predictions,actuals, degrees)
    return np.mean(angular_diff**2)

predictions = np.array([350, 20, 170])
actuals = np.array([10, 25, 160])

angular_mse = calculate_angular_mse(predictions, actuals)
print(f"Angular MSE: {angular_mse}") # Reduced error due to correct difference calculation

predictions_rad = np.radians(predictions)
actuals_rad = np.radians(actuals)
angular_mse_rad = calculate_angular_mse(predictions_rad,actuals_rad, degrees=False)
print(f"Angular MSE (radians): {angular_mse_rad}")
```

Here, the angular mse provides a much more reasonable representation of the error, reflecting the intuitive notion that the angles are quite close. The code works for both degrees and radians.

In my experience, using this approach dramatically improved the performance of the robotic arm's orientation estimation. The system's control loop became far more accurate when feedback was based on the appropriate angular error metric rather than the misleading linear mse.

Beyond the simple angle differences, more complex data structures, such as quaternions (used to represent 3d rotations) need even more care when measuring error. Instead of comparing the individual components of the quaternion directly, it's often better to consider the *angle of rotation* represented by the quaternion (which is an angle around a specific axis), because two different quaternions can represent the exact same rotation.

For example, suppose you have a predicted quaternion `q_predicted` and an actual quaternion `q_actual`. We can define the error as the rotation angle that transforms `q_predicted` into `q_actual` . This is mathematically equivalent to the angle of rotation associated with the quaternion product `q_actual * q_predicted.conjugate()`.

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_angle_error(q_predicted, q_actual):
    """Calculates the angular difference between two quaternions."""
    r_predicted = R.from_quat(q_predicted)
    r_actual = R.from_quat(q_actual)
    r_error = r_actual * r_predicted.inv()
    return r_error.magnitude()

def calculate_quaternion_mse(predictions_quat, actuals_quat):
    """Calculates MSE based on angle of rotation between two quaternions"""
    angular_errors = [quaternion_angle_error(q_pred, q_act) for q_pred, q_act in zip(predictions_quat, actuals_quat)]
    return np.mean(np.array(angular_errors)**2)

predicted_quaternions = np.array([[0.707, 0, 0, 0.707], [0.866, 0, 0.5, 0], [1, 0, 0, 0]]) # some example rotations
actual_quaternions = np.array([[0.707, 0, 0, 0.707], [0.966, 0, 0.259, 0], [0.707, 0.707, 0, 0]]) # some example rotations

mse_quaternion = calculate_quaternion_mse(predicted_quaternions, actual_quaternions)
print(f"Quaternion MSE: {mse_quaternion}") # reduced error using angle difference in quaternions.
```

Here, the error is not computed from the difference of quaternion components, but from the angle of the rotation that is needed to transform the predicted to the actual quaternions. It reflects the error more appropriately in many applications.

In summary, while mse is a powerful and widely used error metric, it’s unsuitable for angular data due to its reliance on linear differences, which fails to capture the circular nature of angles. It's crucial to select metrics that are tailored to the specific data structure and application. When dealing with angular data, calculating the angular difference or employing error measures specific to quaternion-based orientations (like the angle of rotation) provides a much more accurate and reliable representation of the system's performance. For further study, I strongly recommend delving into "Probabilistic Robotics" by Sebastian Thrun, Wolfram Burgard, and Dieter Fox, which contains excellent sections on sensor data processing and error metrics. Also, “3D Math Primer for Graphics and Game Development” by Fletcher Dunn and Ian Parberry provides a solid foundation in the mathematical representation of rotations. Finally, research papers focusing on orientation estimation within sensor fusion literature often detail the various nuances of angular error measurement. These are great places to deepen understanding in this area.
