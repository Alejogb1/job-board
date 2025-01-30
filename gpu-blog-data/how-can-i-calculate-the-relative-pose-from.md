---
title: "How can I calculate the relative pose from two 4x4 affine matrices?"
date: "2025-01-30"
id: "how-can-i-calculate-the-relative-pose-from"
---
The core challenge in calculating the relative pose from two 4x4 affine matrices lies in understanding that these matrices represent transformations, not just points in space.  Direct subtraction isn't meaningful; instead, we need to compose the inverse of the first transformation with the second to obtain the relative transformation. My experience working on robotic arm calibration frequently involved precisely this computation, requiring robust handling of potential numerical instability.  The following explains this process and provides illustrative examples.


**1. Explanation:**

An affine transformation matrix, often represented as a 4x4 matrix, encodes both rotation and translation.  Let's denote the two given affine matrices as  `T1` and `T2`.  `T1` transforms points from frame A to frame B, while `T2` transforms points from frame B to frame C.  Our goal is to find the transformation `T_rel` that directly transforms points from frame A to frame C. This transformation, representing the relative pose, can be determined through matrix multiplication.  Specifically, we compute the inverse of `T1` (`T1⁻¹`) and then multiply it by `T2`:

`T_rel = T2 * T1⁻¹`


This equation accurately reflects the transformation chain: first, we reverse the transformation from A to B using `T1⁻¹`, effectively moving points from frame B back to frame A. Then, we apply the transformation from B to C using `T2`, resulting in the desired transformation from A to C.  Importantly, the order of multiplication matters; reversing it would yield an incorrect result.  The inverse of an affine transformation matrix can be efficiently computed by exploiting its block structure.  For a general affine transformation matrix of the form:

```
T = | R  t |
    | 0  1 |
```

where `R` is a 3x3 rotation matrix and `t` is a 3x1 translation vector, the inverse is:

```
T⁻¹ = | Rᵀ -Rᵀt |
      | 0    1  |
```

where `Rᵀ` is the transpose of `R`, which is equivalent to its inverse in the case of a rotation matrix (assuming it's a proper rotation). The efficiency gained through this direct calculation avoids the computational overhead and potential numerical instability associated with a general matrix inversion algorithm.



**2. Code Examples:**

The following examples demonstrate the relative pose calculation in three different programming environments: Python (using NumPy), MATLAB, and C++.  Each example assumes the affine transformation matrices are already available.  Error handling, crucial in real-world applications, has been omitted for brevity, but should always be incorporated.


**2.1 Python (NumPy):**

```python
import numpy as np

# Example affine transformation matrices
T1 = np.array([[1, 0, 0, 1],
               [0, 1, 0, 2],
               [0, 0, 1, 3],
               [0, 0, 0, 1]])

T2 = np.array([[0, 1, 0, 4],
               [1, 0, 0, 5],
               [0, 0, 1, 6],
               [0, 0, 0, 1]])

# Extract rotation and translation components from T1
R1 = T1[:3, :3]
t1 = T1[:3, 3].reshape(3,1)

# Extract rotation and translation components from T2
R2 = T2[:3, :3]
t2 = T2[:3, 3].reshape(3,1)

# Calculate the inverse of T1
T1_inv = np.eye(4)
T1_inv[:3, :3] = R1.T
T1_inv[:3, 3] = -R1.T @ t1

# Compute the relative transformation
T_rel = T2 @ T1_inv

print("Relative Transformation Matrix:\n", T_rel)
```

This code leverages NumPy's efficient linear algebra operations for matrix manipulation.  The extraction of rotation and translation components allows for a computationally efficient inverse calculation.


**2.2 MATLAB:**

```matlab
% Example affine transformation matrices
T1 = [1 0 0 1; 0 1 0 2; 0 0 1 3; 0 0 0 1];
T2 = [0 1 0 4; 1 0 0 5; 0 0 1 6; 0 0 0 1];

% Extract rotation and translation components from T1
R1 = T1(1:3, 1:3);
t1 = T1(1:3, 4);

% Extract rotation and translation components from T2
R2 = T2(1:3, 1:3);
t2 = T2(1:3, 4);

% Calculate the inverse of T1
T1_inv = eye(4);
T1_inv(1:3, 1:3) = R1';
T1_inv(1:3, 4) = -R1' * t1;

% Compute the relative transformation
T_rel = T2 * T1_inv;

disp('Relative Transformation Matrix:');
disp(T_rel);
```

MATLAB's built-in functions and matrix operations offer a concise and efficient solution, mirroring the Python implementation's structure.


**2.3 C++ (Eigen):**

```cpp
#include <iostream>
#include <Eigen/Dense>

int main() {
  // Example affine transformation matrices
  Eigen::Matrix4d T1, T2;
  T1 << 1, 0, 0, 1,
        0, 1, 0, 2,
        0, 0, 1, 3,
        0, 0, 0, 1;

  T2 << 0, 1, 0, 4,
        1, 0, 0, 5,
        0, 0, 1, 6,
        0, 0, 0, 1;

  // Extract rotation and translation components from T1
  Eigen::Matrix3d R1 = T1.block<3,3>(0,0);
  Eigen::Vector3d t1 = T1.block<3,1>(0,3);

  // Extract rotation and translation components from T2
  Eigen::Matrix3d R2 = T2.block<3,3>(0,0);
  Eigen::Vector3d t2 = T2.block<3,1>(0,3);

  // Calculate the inverse of T1
  Eigen::Matrix4d T1_inv = Eigen::Matrix4d::Identity();
  T1_inv.block<3,3>(0,0) = R1.transpose();
  T1_inv.block<3,1>(0,3) = -R1.transpose() * t1;

  // Compute the relative transformation
  Eigen::Matrix4d T_rel = T2 * T1_inv;

  std::cout << "Relative Transformation Matrix:\n" << T_rel << std::endl;
  return 0;
}
```

This C++ example utilizes the Eigen library, a powerful linear algebra library offering similar functionality and efficiency to NumPy and MATLAB's built-in tools.  It demonstrates effective use of Eigen's block operations for efficient matrix manipulation.



**3. Resource Recommendations:**

For a deeper understanding of affine transformations and matrix operations, I recommend consulting standard linear algebra textbooks and resources focused on computer graphics and robotics.  Specific texts on numerical methods will be valuable for understanding the stability considerations inherent in matrix inversion.  Furthermore, the documentation for the specific linear algebra libraries used (NumPy, MATLAB, Eigen) provides extensive details on their functionalities and optimizations.  Understanding homogeneous coordinates and their applications in 3D transformations is also essential.
