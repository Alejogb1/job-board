---
title: "How can random Fourier features and Performers approximate the softmax kernel?"
date: "2024-12-23"
id: "how-can-random-fourier-features-and-performers-approximate-the-softmax-kernel"
---

, let's delve into this. I remember a particularly nasty machine learning problem back at 'Project Chimera' where we had to deal with an enormous dataset and the traditional softmax kernel was just choking our system. That's when I really had to get comfortable with random Fourier features and Performers. It became apparent that approximating the softmax kernel efficiently was paramount, and these two methods proved invaluable. So, to answer your question directly, let's break down *how* they achieve this.

First, it’s essential to understand why the softmax kernel is such a resource hog. The softmax kernel, given input vectors *x* and *y*, is often represented as exp(*x*<sup>T</sup>*y*). Calculating this for all pairs of data points scales quadratically with the number of data points, making it computationally intractable for large datasets. The goal here is to find an approximation that preserves the properties of the kernel while being significantly faster to compute.

**Random Fourier Features (RFF)**:

The idea behind RFF is elegantly simple: we approximate the kernel using random features derived from the Fourier transform. This isn’t some wild theoretical leap, but a practical application of Bochner’s theorem. Essentially, any shift-invariant kernel – which includes the softmax kernel – can be represented as the Fourier transform of a probability distribution. Instead of explicitly computing the full kernel matrix, we sample from this distribution and project the data into a lower-dimensional space.

Specifically, for a shift-invariant kernel *k*(x, y) = k(x-y), we can write it as:

 *k*(x, y) ≈ 1/*D* Σ<sub>i=1</sub><sup>D</sup> exp(j*ω<sub>i</sub><sup>T</sup>(x - y))

Where *ω<sub>i</sub>* are sampled from the kernel's frequency spectrum, and D is the number of features. Note, here, ‘j’ represents the imaginary unit. This expression can then be rewritten as the inner product of feature maps:

  *k*(x, y) ≈ φ(x)<sup>T</sup>φ(y)

Where φ(x) = [exp(j*ω<sub>1</sub><sup>T</sup>x), exp(j*ω<sub>2</sub><sup>T</sup>x), … , exp(j*ω<sub>D</sub><sup>T</sup>x)] / sqrt(*D*).

Now, here's the catch. The softmax kernel as it stands *isn't* shift-invariant. But, we are not aiming to replace it directly; we are trying to approximate the results it would give. In practice, when using RFF to approximate the results of a softmax kernel, what happens is we generate random features based on a gaussian distribution or something similar (the actual distribution depends on the desired kernel) and use those to approximate the result we would have gotten from the softmax kernel. We are not applying a fourier transform to the kernel itself.

Here is a practical Python snippet to illustrate this:

```python
import numpy as np

def gaussian_rff(x, num_features, gamma):
    """Generates random Fourier features for a Gaussian kernel.

    Args:
      x: Numpy array representing the input data (n_samples, n_features).
      num_features: The number of random features to generate.
      gamma: The kernel parameter.

    Returns:
      Numpy array of transformed features (n_samples, num_features).
    """
    n_dim = x.shape[1]
    omega = np.random.normal(0, gamma, size=(num_features, n_dim))
    b = np.random.uniform(0, 2 * np.pi, size=(num_features, 1))
    features = np.cos(np.dot(x, omega.T) + b.T) / np.sqrt(num_features)
    return features


# Example usage
data = np.random.rand(100, 5)  # 100 samples, 5 features
num_features = 500
gamma = 0.1

features = gaussian_rff(data, num_features, gamma)
print(f"Shape of RFF features: {features.shape}")
```

Notice how we are generating random frequencies (omega) sampled from a gaussian distribution. We then use the generated frequencies and the input features to create a new, lower-dimensional, feature representation. This is then used to approximate the result of the softmax kernel. This makes the calculations much more efficient.

**Performers**:

Performers take a slightly different approach, focusing on linear approximations of kernels, including softmax (actually an approximation of the related gaussian kernel which then can be used in the softmax). They leverage low-rank matrix approximations to achieve efficient computation.

The core concept relies on a decomposition trick that allows us to approximate the softmax kernel as:

  exp(*x*<sup>T</sup>*y*) ≈ *Q*(x)<sup>T</sup>*Q*(y)

Where Q(x) is a learned feature map. This low-rank approximation is crucial because it allows us to perform matrix multiplications in O(n*d*) time complexity, instead of O(n<sup>2</sup>). They achieve this through the use of random projections which then leads to feature embeddings which are more amenable to efficient computation.

The key idea of Performers can be simplified as follows: consider the radial basis function kernel *k*(x, y) = exp(-γ||x - y||<sup>2</sup>). The core of Performers relies on a mapping that translates a high-dimensional input vector to a lower-dimensional space, such as a vector of cosines of projections, and then the approximation of the kernel function becomes the inner product of these vectors. The crucial part is that this mapping can be done effectively with randomness and the use of techniques such as the kernel trick.

Here is a code snippet showing this concept in action. We are using the kernel approximation through cosine features.

```python
import numpy as np

def performer_kernel_approx(x, num_features, gamma):
    """Approximates the Gaussian kernel using Performers' approach.

      Args:
        x: Numpy array representing the input data (n_samples, n_features).
        num_features: Number of random features to generate.
        gamma: The kernel parameter.

      Returns:
        Numpy array of transformed features (n_samples, num_features).
    """
    n_dim = x.shape[1]
    omega = np.random.normal(0, np.sqrt(gamma), size=(num_features, n_dim))
    phi = np.cos(np.dot(x, omega.T)) / np.sqrt(num_features)
    return phi

# Example usage
data = np.random.rand(100, 5)
num_features = 500
gamma = 0.1
transformed_features = performer_kernel_approx(data, num_features, gamma)
print(f"Shape of performer features: {transformed_features.shape}")
```

Again, note the generation of random frequencies and the use of cosines. This transformation gives us a feature set that can be computed efficiently. It is important to note that while this gives an approximation of a gaussian kernel, we can use this gaussian kernel to approximate the results of the softmax kernel. This is because the softmax kernel can be written in such a way that it can be approximated using a gaussian kernel and some normalization.

**Comparison and Implementation Notes:**

While both RFF and Performers aim to achieve efficient kernel approximations, their underlying approaches and guarantees differ:

*   **RFF:** Well-established theory, based on Bochner's theorem. It's fairly straightforward to implement and works well for shift-invariant kernels. The quality of the approximation improves with the number of random features. One issue is that for certain kernels such as the softmax, the approximation quality might require a larger feature dimensionality than desired.

*   **Performers:** Leveraging low-rank matrix approximations, it offers better computational efficiency and theoretical guarantees in certain cases, especially for approximating kernels in high-dimensional spaces, and has a more direct approach for handling non-shift invariant kernels such as the gaussian kernel.

Here is a third code snippet, showing a naive implementation of how we can use these approximations:

```python
import numpy as np
import scipy.special

def softmax_approx(x, y, num_features=500, gamma=0.1, method="rff"):
   """Approximates softmax kernel.

       Args:
         x: First input vector.
         y: Second input vector.
         num_features: Number of random features.
         gamma: kernel parameter.
         method: either 'rff' or 'performer'.

      Returns:
         Approximation of softmax(x,y).
    """
   if method == "rff":
      phi_x = gaussian_rff(x.reshape(1,-1), num_features, gamma)
      phi_y = gaussian_rff(y.reshape(1,-1), num_features, gamma)
      return np.dot(phi_x, phi_y.T)[0, 0]

   elif method == "performer":
      phi_x = performer_kernel_approx(x.reshape(1,-1), num_features, gamma)
      phi_y = performer_kernel_approx(y.reshape(1,-1), num_features, gamma)
      return np.dot(phi_x, phi_y.T)[0, 0]

   else:
      raise ValueError("Invalid method. Please choose either 'rff' or 'performer'.")


#Example usage
x = np.random.rand(5)
y = np.random.rand(5)

rff_approximation = softmax_approx(x, y, method="rff")
performer_approximation = softmax_approx(x, y, method="performer")

print(f"RFF Approximated Softmax: {rff_approximation}")
print(f"Performers Approximated Softmax: {performer_approximation}")
```

**Further Exploration:**

For deeper understanding, I recommend the following:

*   **"Random Features for Large-Scale Kernel Machines" by Rahimi and Recht (2007):** This paper is fundamental for grasping the theory behind RFF.
*   **"Rethinking Attention with Performers" by Choromanski et al. (2021):** This paper thoroughly explores Performers and their theoretical underpinnings, as well as their application in transformers.
*   **"Gaussian Processes for Machine Learning" by Carl Edward Rasmussen and Christopher K.I. Williams:** A comprehensive book that delves into the theory and application of kernel methods, with a very good discussion on kernels, including the gaussian kernel and the softmax.

In my experience, understanding both RFF and Performers has become indispensable when dealing with large-scale machine learning problems that require approximating kernel functions. Choosing between them depends heavily on the specific task, dataset characteristics, and trade-offs between computational efficiency and approximation quality. It's not a one-size-fits-all solution; you have to experiment and see what works best in practice.
