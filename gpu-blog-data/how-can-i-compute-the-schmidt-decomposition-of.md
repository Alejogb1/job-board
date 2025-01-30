---
title: "How can I compute the Schmidt decomposition of a matrix in Python?"
date: "2025-01-30"
id: "how-can-i-compute-the-schmidt-decomposition-of"
---
The Schmidt decomposition, also known as the singular value decomposition (SVD) for bipartite quantum states, offers a powerful tool for analyzing entanglement.  Crucially, it's not a direct matrix decomposition in the same vein as an eigendecomposition, but rather a representation of a matrix as a sum of outer products, revealing the structure of entanglement inherent in the system.  My experience working on quantum information processing algorithms extensively highlighted the subtle distinctions between SVD and the Schmidt decomposition, particularly concerning the interpretation of singular values and the inherent dimensionality constraints.

**1.  Clear Explanation:**

The Schmidt decomposition applies to matrices representing bipartite quantum states, meaning states involving two distinct subsystems, A and B. Let's consider a matrix  `ρ_AB` representing the density matrix of a bipartite system. This matrix acts on the tensor product Hilbert space H_A ⊗ H_B, where H_A and H_B have dimensions N_A and N_B, respectively. The Schmidt decomposition expresses this density matrix as:

`ρ_AB = Σ_i λ_i |i⟩⟨i| ⊗ |i⟩⟨i|`

where:

* `λ_i` are the Schmidt coefficients, non-negative real numbers representing the weight of each entangled state.  These are analogous to singular values in the SVD. They satisfy the normalization condition: Σ_i λ_i = 1.
* `|i⟩` represents an orthonormal basis in H_A.
* `|i⟩` (the same state) represents an orthonormal basis in H_B. The dimensionality of these bases is determined by the rank of the original matrix; the number of non-zero Schmidt coefficients is the Schmidt rank.

The key difference from standard SVD lies in the crucial constraint that the same basis is used for both subsystems.  This reflects the entanglement structure:  the Schmidt decomposition reveals the correlated states in each subsystem contributing to the overall state. A higher Schmidt rank indicates a greater degree of entanglement.  A Schmidt rank of 1 signifies a maximally entangled pure state.

A practical computation hinges on using the standard SVD as a stepping stone.  First, we perform the SVD of the matrix representing the bipartite state. Then, we restructure the results to conform to the Schmidt decomposition form described above.  This typically involves manipulating the singular vectors and singular values to satisfy the equal-basis constraint.

**2. Code Examples with Commentary:**

The following examples utilize NumPy and SciPy, assuming familiarity with these libraries and their linear algebra functions.

**Example 1:  Schmidt Decomposition of a 2x2 Matrix:**

```python
import numpy as np
from scipy.linalg import svd

# Define a 2x2 density matrix (example)
rho_AB = np.array([[0.6, 0.4],
                   [0.4, 0.6]])

# Perform SVD
U, S, Vh = svd(rho_AB)

# Construct Schmidt decomposition
lambda_i = S  # Singular values are Schmidt coefficients
Schmidt_basis_A = U
Schmidt_basis_B = Vh.conj().T  #Note the conjugate transpose

# Verification (optional): reconstruct rho_AB
rho_AB_reconstructed = np.zeros((2, 2), dtype=complex)
for i in range(len(lambda_i)):
    rho_AB_reconstructed += lambda_i[i] * np.outer(Schmidt_basis_A[:, i], Schmidt_basis_B[:, i])

print("Original Matrix:\n", rho_AB)
print("\nReconstructed Matrix:\n", rho_AB_reconstructed)
print("\nSchmidt Coefficients:\n", lambda_i)
print("\nSchmidt Basis A:\n", Schmidt_basis_A)
print("\nSchmidt Basis B:\n", Schmidt_basis_B)

```

This demonstrates the straightforward application of SVD to extract Schmidt coefficients and bases for a small, simple case. The reconstruction verifies the accuracy of the process. Note the use of `Vh.conj().T` to ensure the proper Hermitian conjugate for the basis in system B.


**Example 2: Handling Higher Dimensional Matrices:**

```python
import numpy as np
from scipy.linalg import svd

# Define a larger 4x3 density matrix (example - requires proper normalization)
rho_AB = np.random.rand(4,3)
rho_AB = rho_AB @ rho_AB.conj().T #Ensures a hermitian and positive semi-definite matrix
rho_AB = rho_AB / np.trace(rho_AB) #Normalization step

U, S, Vh = svd(rho_AB)

#For larger matrices, the Schmidt rank needs careful consideration. Truncation may be necessary.
Schmidt_rank = np.sum(S > 1e-10) #Only consider singular values above a threshold

lambda_i = S[:Schmidt_rank]
Schmidt_basis_A = U[:, :Schmidt_rank]
Schmidt_basis_B = Vh.conj().T[:, :Schmidt_rank]

#Verification (truncated):
rho_AB_reconstructed = np.zeros((4, 3), dtype=complex)
for i in range(Schmidt_rank):
    rho_AB_reconstructed += lambda_i[i] * np.outer(Schmidt_basis_A[:, i], Schmidt_basis_B[:, i])

print("Original Matrix (truncated):\n", rho_AB)
print("\nReconstructed Matrix (truncated):\n", rho_AB_reconstructed)
print("\nSchmidt Coefficients:\n", lambda_i)
print("\nSchmidt Rank:", Schmidt_rank)
```

This example highlights the importance of handling higher-dimensional matrices efficiently.  The inclusion of a Schmidt rank determination step, based on a threshold to identify numerically significant singular values, is crucial for practical computation and to avoid including insignificant noise.  Truncation is often necessary to manage computational complexity and noise.


**Example 3:  Pure State Considerations:**

```python
import numpy as np
from scipy.linalg import svd

# Example of a pure state represented as a matrix (vectorized)
psi_AB = np.array([0.707, 0, 0, 0.707]) #Example of a Bell State
rho_AB = np.outer(psi_AB, psi_AB.conj()) #Convert to density matrix

U, S, Vh = svd(rho_AB)

lambda_i = S
Schmidt_basis_A = U
Schmidt_basis_B = Vh.conj().T

print("Original Matrix:\n", rho_AB)
print("\nSchmidt Coefficients:\n", lambda_i)
print("\nSchmidt Basis A:\n", Schmidt_basis_A)
print("\nSchmidt Basis B:\n", Schmidt_basis_B)

```

This case illustrates the Schmidt decomposition for pure states. For a pure state, only one Schmidt coefficient will be non-zero, and the Schmidt rank is one.  This code explicitly demonstrates this property, showcasing the behaviour expected for a maximally entangled state.


**3. Resource Recommendations:**

Nielsen & Chuang's "Quantum Computation and Quantum Information" offers a comprehensive theoretical treatment of quantum information concepts, including a detailed explanation of the Schmidt decomposition and its significance.  Many advanced linear algebra texts cover the theoretical foundations of SVD, providing a strong base for understanding the mathematical underpinnings of the Schmidt decomposition.  Finally,  a text focusing on quantum optics or quantum many-body physics will likely offer insights into applications and interpretations of this decomposition within various physical contexts.
