---
title: "eigen quaternion product?"
date: "2024-12-13"
id: "eigen-quaternion-product"
---

 so you want to talk about eigen quaternion products right I've been there trust me a long time ago actually back when I was still hacking away on my first flight simulator project that was a mess of linear algebra and well quaternions were the less messy part believe it or not It's a beast when you start mixing eigen concepts and these 4D things but totally manageable once you get your head around the fundamental operations

so the core issue here is you're dealing with rotations basically rotations in 3D space are notoriously awkward with matrices if you try to chain a bunch of rotations with standard rotation matrices you often run into gimbal lock and numerical drift over time which are no fun at all So quaternions to the rescue they offer a compact and stable way to represent rotations So your eigenvector problem which likely deals with linear transformations in a space that might be a rotation space is being described by quaternions instead of matrices

Now you aren't directly taking "eigenvectors of a quaternion" that's not a thing You typically deal with eigen stuff with matrices because they are inherently linear operators But you can certainly have an eigen problem where the solution is represented by a quaternion it could be for instance the rotation axis you're looking for or something relating to a covariance matrix that describes a rotational component of data For example you have a covariance matrix describing data that is rotated in 3D space you might perform an eigenvalue decomposition on the covariance to obtain principal axes and the principal rotation associated to one of these directions will be your quaternion solution

So what do we do if we need to find a product of eigen-related quantities and a quaternion well it usually comes down to a few common patterns It usually goes like this

*   **Rotation application:** the eigen-stuff typically gives you a rotation axis or a rotation angle which is used to build your quaternion then you have that rotation quaternion and you apply it to some other vector or object

*   **Quaternion composition:** sometimes you have two rotations represented by quaternions where one of them is derived from eigenvector analysis and you are trying to combine them into a single rotation that’s a quaternion product

*   **Transforming a vector or frame:** your eigen derived information might be a transform between two frames or vector spaces where one might be represented by a quaternion

Let's go through some code examples to hammer this down since I know you asked for that you just said "code snippets as examples that should work" and I always deliver that like the time I delivered a half cooked machine learning model to a client after an all nighter anyway

 first the classic rotation application This is where you get your rotation quaternion and you need to rotate a 3D vector We assume that your eigen analysis gave you all the elements needed to construct this quaternion its axis and its rotation angle for example

```python
import numpy as np

def quaternion_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis) # make sure it's unit
    s = np.sin(angle/2)
    c = np.cos(angle/2)
    return np.array([c, s*axis[0], s*axis[1], s*axis[2]])

def apply_rotation(q, v):
  # Convert quaternion to rotation matrix
  q_w, q_x, q_y, q_z = q
  R = np.array([
    [1 - 2*q_y**2 - 2*q_z**2, 2*q_x*q_y - 2*q_w*q_z, 2*q_x*q_z + 2*q_w*q_y],
    [2*q_x*q_y + 2*q_w*q_z, 1 - 2*q_x**2 - 2*q_z**2, 2*q_y*q_z - 2*q_w*q_x],
    [2*q_x*q_z - 2*q_w*q_y, 2*q_y*q_z + 2*q_w*q_x, 1 - 2*q_x**2 - 2*q_y**2]
  ])
  return np.dot(R, v)

axis = np.array([1, 0, 0])
angle = np.pi/2 # 90 degrees
q = quaternion_from_axis_angle(axis, angle)
v = np.array([1, 2, 3])

rotated_v = apply_rotation(q, v)
print(rotated_v)
```

This snippet shows how you take an axis angle that you got from your eigen-problem for example and then you rotate an arbitrary vector using this quaternion which is built from said eigen results The `apply_rotation` function does the heavy lifting of converting the quaternion to a rotation matrix then apply it to the vector

Next up is quaternion composition where you have two or more rotations represented as quaternions and you want to combine them into a single rotation you would use a quaternion product to compose them

```python
import numpy as np

def quaternion_multiply(q1, q2):
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  w = w1*w2 - x1*x2 - y1*y2 - z1*z2
  x = w1*x2 + x1*w2 + y1*z2 - z1*y2
  y = w1*y2 - x1*z2 + y1*w2 + z1*x2
  z = w1*z2 + x1*y2 - y1*x2 + z1*w2
  return np.array([w, x, y, z])


axis1 = np.array([0, 0, 1])
angle1 = np.pi/4
q1 = quaternion_from_axis_angle(axis1, angle1)

axis2 = np.array([1, 0, 0])
angle2 = np.pi/2
q2 = quaternion_from_axis_angle(axis2, angle2)

composed_q = quaternion_multiply(q1, q2) # first rotate by q2 then by q1. It's the same as matrix product.
v = np.array([1,0,0])
composed_v = apply_rotation(composed_q, v)
print(composed_v)
```

Here `quaternion_multiply` does the core quaternion product and then in the example you have two rotations then you apply a vector using the composed rotation Again we assume that one or both quaternions are coming from eigenvalue analysis of some kind

And finally let's look at how to use a quaternion as a frame transformation I’ve used this many times when dealing with transformations between sensor frames and world frames

```python
import numpy as np

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def transform_vector(q, v):
    q_v = np.array([0, v[0], v[1], v[2]])
    q_inv = quaternion_conjugate(q) # conjugate is the inverse for a rotation
    rotated_q = quaternion_multiply(quaternion_multiply(q,q_v), q_inv)
    return rotated_q[1:4]

q = quaternion_from_axis_angle(np.array([0,1,0]), np.pi/2)
vector_in_frame = np.array([1,0,0])
vector_in_world = transform_vector(q,vector_in_frame)
print(vector_in_world)
```

In this last example you would transform a vector from one frame to another using the quaternion that describes the relative rotation between these frames it's super helpful for changing coordinate systems and again one of these frames could very well be coming from eigenanalysis of some data

Now for resources if you are new to this kind of thing I would recommend “3D Math Primer for Graphics and Game Development” It's a solid book for brushing up on the fundamentals of linear algebra and how it ties into 3D transformations and has a whole section on quaternions If you want to dive deeper into the math behind it all “Robotics: Modelling Planning and Control” is a pretty robust text although it might be a bit dense if you are not deep into robotics and control theory the quaternion theory is very well treated

And since you are making me do a joke like you asked once I was trying to debug a quaternion multiplication and I was pulling my hair out only to find out I had swapped the components and was doing the equivalent of multiplying a vector by it self It felt very awkward I tell you

So yeah that’s the gist of it really Eigen stuff and quaternions it might look complicated but it all comes down to careful application of basic math principles and of course making sure you get your quaternion products right so you do not multiply a vector by itself like I did that one time. Let me know if you have any specific cases in mind or more questions I’m always happy to help
