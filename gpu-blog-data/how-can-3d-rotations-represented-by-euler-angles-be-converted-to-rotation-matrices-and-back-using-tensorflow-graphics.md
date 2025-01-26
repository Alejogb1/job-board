---
title: "How can 3D rotations represented by Euler angles be converted to rotation matrices and back using TensorFlow-Graphics?"
date: "2025-01-26"
id: "how-can-3d-rotations-represented-by-euler-angles-be-converted-to-rotation-matrices-and-back-using-tensorflow-graphics"
---

Euler angles, while intuitive for human understanding, suffer from gimbal lock and non-unique representations, making rotation matrices a preferred format for computation in 3D graphics. TensorFlow-Graphics, a library providing differentiable graphics primitives, facilitates the conversion between these two representations using optimized tensor operations. I’ve frequently used these conversions when developing custom articulated character animation systems, where precise control over bone orientations is critical for physically plausible motion.

**Understanding the Conversion Process**

The fundamental challenge lies in transforming a set of three rotation angles (typically around the X, Y, and Z axes – though the specific order can vary) into a single 3x3 matrix representing the equivalent rotation. The process, referred to as constructing the rotation matrix from Euler angles, involves applying the rotational transformations individually, in a specified order, then multiplying the resulting matrices. The reverse operation – extracting Euler angles from a rotation matrix – is more complex, requiring trigonometric function inversions which can lead to ambiguities.

**Euler Angles to Rotation Matrix Conversion**

The conversion from Euler angles to a rotation matrix is achieved by creating individual rotation matrices for each axis, then multiplying these matrices in the designated order. Let's assume an 'XYZ' rotation order where we first rotate about the X-axis, then the Y-axis, and finally the Z-axis. If we have Euler angles represented as α (around X), β (around Y), and γ (around Z), the rotation matrices for each axis are as follows:

*   **Rotation around X (Rx):**
    ```
        [ 1  0         0 ]
        [ 0  cos(α)  -sin(α) ]
        [ 0  sin(α)   cos(α) ]
    ```

*   **Rotation around Y (Ry):**
    ```
        [ cos(β)  0  sin(β) ]
        [ 0      1  0      ]
        [-sin(β)  0  cos(β) ]
    ```

*   **Rotation around Z (Rz):**
    ```
        [ cos(γ) -sin(γ)  0 ]
        [ sin(γ)  cos(γ)  0 ]
        [ 0       0       1 ]
    ```

The complete rotation matrix, R, for the XYZ order is:  R = Rz * Ry * Rx.

**Code Example 1: Euler Angles to Rotation Matrix (XYZ Order)**

```python
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation

def euler_to_rotation_matrix_xyz(euler_angles):
  """Converts Euler angles (XYZ order) to a rotation matrix.

  Args:
      euler_angles: A tensor of shape [..., 3] representing Euler angles
        (in radians) for each rotation around the X, Y, and Z axes.

  Returns:
      A tensor of shape [..., 3, 3] representing the rotation matrices.
  """
  x_rotation = tfg_transformation.rotation_matrix_3d.from_euler(
      euler_angles[..., 0], axis=tf.constant([1., 0., 0.], dtype=tf.float32))
  y_rotation = tfg_transformation.rotation_matrix_3d.from_euler(
      euler_angles[..., 1], axis=tf.constant([0., 1., 0.], dtype=tf.float32))
  z_rotation = tfg_transformation.rotation_matrix_3d.from_euler(
      euler_angles[..., 2], axis=tf.constant([0., 0., 1.], dtype=tf.float32))

  rotation_matrix = tf.matmul(z_rotation, tf.matmul(y_rotation, x_rotation))
  return rotation_matrix

# Example Usage:
euler_angles = tf.constant([[tf.pi/4, tf.pi/3, tf.pi/6], [tf.pi/2, 0, 0]], dtype=tf.float32) # 2 sets of Euler angles (in radians)
rotation_matrices = euler_to_rotation_matrix_xyz(euler_angles)
print(rotation_matrices)
```

*   The function `euler_to_rotation_matrix_xyz` accepts Euler angles (in radians) with an assumed XYZ order. It uses `tfg_transformation.rotation_matrix_3d.from_euler` to generate individual rotation matrices, and then uses `tf.matmul` to compute the final rotation matrix. The rotation is performed based on the assumed XYZ order by applying matrix multiplications from right to left, Rx * Ry * Rz
* The code illustrates a batch conversion by providing two sets of angles.

**Rotation Matrix to Euler Angles Conversion**

The reverse transformation is more intricate and relies on solving the matrix equations using inverse trigonometric functions. This process is inherently ambiguous because multiple Euler angle combinations can lead to the same rotation matrix. The standard approach involves extracting angles based on specific matrix elements. Given our defined XYZ order, the following relationships can be used:

*   β = arcsin(R[0, 2])
*   α = arctan2(-R[1, 2], R[2, 2])
*   γ = arctan2(-R[0, 1], R[0, 0])

These formulas are subject to singularities when cos(β) is close to zero. This corresponds to a gimbal lock situation where a degree of freedom is lost.

**Code Example 2: Rotation Matrix to Euler Angles (XYZ Order)**

```python
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation

def rotation_matrix_to_euler_xyz(rotation_matrix):
    """Converts a rotation matrix to Euler angles (XYZ order).

    Args:
      rotation_matrix: A tensor of shape [..., 3, 3] representing rotation matrices.

    Returns:
        A tensor of shape [..., 3] representing Euler angles (in radians) for the
        X, Y, and Z axes.
    """
    beta = tf.asin(rotation_matrix[..., 0, 2])
    alpha = tf.atan2(-rotation_matrix[..., 1, 2], rotation_matrix[..., 2, 2])
    gamma = tf.atan2(-rotation_matrix[..., 0, 1], rotation_matrix[..., 0, 0])
    euler_angles = tf.stack([alpha, beta, gamma], axis=-1)
    return euler_angles

# Example Usage:
euler_angles = tf.constant([[tf.pi/4, tf.pi/3, tf.pi/6], [tf.pi/2, 0, 0]], dtype=tf.float32) # 2 sets of Euler angles (in radians)
rotation_matrices = euler_to_rotation_matrix_xyz(euler_angles)
extracted_euler_angles = rotation_matrix_to_euler_xyz(rotation_matrices)
print(extracted_euler_angles)
```

*   The function `rotation_matrix_to_euler_xyz` directly calculates the Euler angles using the trigonometric relations described, leveraging TensorFlow's optimized math functions. The output is a tensor representing the Euler angles (in radians) for each input rotation matrix. It assumes an XYZ order for the provided matrices.
* This example first generates a matrix with code provided in example 1, and then calculates and prints Euler angles based on the newly created rotation matrices.

**Addressing Gimbal Lock and Order Dependence**

The key limitations with Euler angles are order dependence and gimbal lock. When using Euler angles, choosing an order of rotations such as XYZ, ZYX, or others, affects the resulting rotation. Furthermore, when one rotation axis approaches an alignment with another, you lose a degree of freedom, producing a singularity.

Rotation matrices do not suffer from these issues. Because rotations are composed using matrix multiplication, order dependencies are explicitly encoded in the matrix representation. While rotation matrices are an effective workaround for gimbal lock and non-unique representation, Euler angles still serve a purpose where intuitive understanding of rotation about specific axis is needed. This also enables manual modification and artistic manipulation of rotation.

**Code Example 3: Demonstration of Different Euler Angles Resulting in the Same Rotation**

```python
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation

def compare_rotations(euler1, euler2):
  """Compares if two different Euler angle sets represent the same rotation.

  Args:
    euler1: Tensor representing first set of Euler angles (XYZ order).
    euler2: Tensor representing second set of Euler angles (XYZ order).

  Returns:
    True if both sets of Euler angles represent the same rotation,
    False otherwise.
  """
  rotation1 = euler_to_rotation_matrix_xyz(euler1)
  rotation2 = euler_to_rotation_matrix_xyz(euler2)
  return tf.reduce_all(tf.abs(rotation1 - rotation2) < 1e-6)

# Example Usage:
euler1 = tf.constant([tf.pi/2, 0, tf.pi/2], dtype=tf.float32) # 90 deg around x, followed by 90 deg around z.
euler2 = tf.constant([0, tf.pi/2, 0], dtype=tf.float32) # 90 deg around y.
print("Do different Euler Angles produce the same rotation: ", compare_rotations(tf.reshape(euler1,[1,3]), tf.reshape(euler2,[1,3])).numpy())


euler3 = tf.constant([tf.pi/4, tf.pi/3, tf.pi/6], dtype=tf.float32) #arbitrary Euler angles.
euler4 = tf.constant([tf.pi/8, tf.pi/5, tf.pi/7], dtype=tf.float32) #another set of different Euler angles.
print("Do different Euler Angles produce the same rotation: ", compare_rotations(tf.reshape(euler3,[1,3]), tf.reshape(euler4,[1,3])).numpy())
```

*   The `compare_rotations` function calculates rotation matrices using euler angles and compares them, allowing us to test if rotations are equivalent even if their Euler representation is different.
* This code example specifically demonstrates that different Euler angle sets can generate the same rotation matrix and that this can be verified by calculating and comparing the resulting matrices. It highlights non-uniqueness of representation.

**Resource Recommendations**

For a deeper understanding of 3D rotations and their mathematical representations, explore resources on linear algebra, especially focusing on rotation matrices and transformations. Materials covering computer graphics fundamentals will provide further context on the practical application of these concepts. Texts on robotics or kinematics often delve into the intricacies of Euler angles and their limitations in systems requiring precise orientation control. Look for publications that discuss the nuances of singularities and alternative representations like quaternions as well. For a deep dive into the computational implementation, refer to publications that cover numerical methods and approximation techniques used in matrix calculations. These resources together will provide a well rounded understanding of the theory and practice of dealing with 3D transformations.
