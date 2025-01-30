---
title: "How can a 2D integral be evaluated on a non-uniform grid?"
date: "2025-01-30"
id: "how-can-a-2d-integral-be-evaluated-on"
---
The challenge in numerically evaluating a 2D integral over a non-uniform grid stems from the fact that standard quadrature rules, like those based on rectangles or trapezoids, are designed for uniformly spaced data. When dealing with irregularly spaced points, we can't simply apply these rules directly without incurring significant inaccuracies. My experience in computational fluid dynamics, where adaptive mesh refinement is common, has made dealing with such non-uniform grids a frequent necessity. The key is to locally approximate the integrand and use those local approximations to perform the integration.

The core concept revolves around tessellating the non-uniform grid into simpler shapes – typically triangles or quadrilaterals. Then, we can utilize techniques appropriate for these shapes to compute the area and integrate the interpolated function. The choice of tessellation and local approximation affects both accuracy and computational cost. I'll detail a common approach using triangulation and linear interpolation, while also briefly covering alternatives.

**Triangulation and Linear Interpolation:**

This approach starts by triangulating the set of scattered points in the 2D plane, creating a mesh of non-overlapping triangles. Algorithms like Delaunay triangulation are often preferred because they maximize the minimum angle of the triangles, which is beneficial for numerical stability. Once a suitable triangulation is obtained, we can perform the integration by approximating the integrand over each triangle.

Within a given triangle, we linearly interpolate the function value based on the function's values at the vertices. Let *f(x, y)* be the function we're integrating, and let *f<sub>1</sub>*, *f<sub>2</sub>*, and *f<sub>3</sub>* be the values of the function at the vertices of a triangle with coordinates (*x<sub>1</sub>*, *y<sub>1</sub>*), (*x<sub>2</sub>*, *y<sub>2</sub>*), and (*x<sub>3</sub>*, *y<sub>3</sub>*), respectively. The linear interpolation of the function at any point *(x, y)* within the triangle is given by:

*f(x, y) ≈ N<sub>1</sub>(x, y)f<sub>1</sub> + N<sub>2</sub>(x, y)f<sub>2</sub> + N<sub>3</sub>(x, y)f<sub>3</sub>*

Where *N<sub>i</sub>(x, y)* are the linear shape functions associated with each node *i*. In barycentric coordinates (*λ<sub>1</sub>*, *λ<sub>2</sub>*, *λ<sub>3</sub>*), where *λ<sub>1</sub> + λ<sub>2</sub> + λ<sub>3</sub> = 1* and *x = λ<sub>1</sub>x<sub>1</sub> + λ<sub>2</sub>x<sub>2</sub> + λ<sub>3</sub>x<sub>3</sub>* and *y = λ<sub>1</sub>y<sub>1</sub> + λ<sub>2</sub>y<sub>2</sub> + λ<sub>3</sub>y<sub>3</sub>*, the shape functions are equivalent to the barycentric coordinates (i.e., *N<sub>i</sub> = λ<sub>i</sub>*).

The integral over a single triangle can then be computed as:

∫∫<sub>triangle</sub> *f(x, y) dx dy ≈ ∫∫<sub>triangle</sub> (N<sub>1</sub>(x, y)f<sub>1</sub> + N<sub>2</sub>(x, y)f<sub>2</sub> + N<sub>3</sub>(x, y)f<sub>3</sub>) dx dy*

The integration of the shape functions multiplied by the constant function values results in area of the triangle times one-third multiplied by the vertex values:

Area(triangle) / 3 * (*f<sub>1</sub> + f<sub>2</sub> + f<sub>3</sub>*)

The final 2D integral is obtained by summing the contributions from all triangles in the mesh.

**Code Example 1: Triangulation and Function Evaluation**

This Python example demonstrates using the `scipy` library to perform the triangulation and evaluate the function on each point.

```python
import numpy as np
from scipy.spatial import Delaunay

# Sample non-uniform grid points (x, y coordinates)
points = np.array([[0, 0], [1, 0], [0.5, 0.8], [0.2, 0.3], [1.2, 0.5], [1.8, 0.1]])

# Create Delaunay triangulation
tri = Delaunay(points)

# Sample function (to be integrated)
def my_function(x, y):
  return x**2 + y**2

# Evaluate the function at the grid points
function_values = np.array([my_function(x, y) for x, y in points])


# Each simplex (triangle) in tri.simplices is a list of indices, we lookup function values by using these indices
function_values_per_triangle = function_values[tri.simplices]

print("Triangulation simplices:", tri.simplices)
print("Function values at each triangle vertex:", function_values_per_triangle)
```

This segment generates a Delaunay triangulation on a sample set of points. It also samples a given function on those points. The `tri.simplices` object stores the triangles in terms of vertex indices. These indices are then used to access the function values at each vertex.

**Code Example 2: Computing Triangle Area**

This Python snippet calculates the area of a given triangle based on its vertices.

```python
def triangle_area(vertices):
  """Calculate area of a triangle given its vertices."""
  x1, y1 = vertices[0]
  x2, y2 = vertices[1]
  x3, y3 = vertices[2]
  return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


triangle_vertices = points[tri.simplices]
triangle_areas = np.array([triangle_area(vertices) for vertices in triangle_vertices])

print("Areas of triangles:", triangle_areas)
```

The `triangle_area` function leverages the determinant formula to determine the area using the coordinates of the vertices of each triangle from the triangulation. The code demonstrates this by calculating all triangle areas based on the triangulation produced in the previous snippet.

**Code Example 3: Integrating over Triangles**

This final Python segment computes the approximated integral using the triangle areas and the function values evaluated earlier.

```python
# Weighted average of function values over triangle vertices (average of function value per triangle)
weighted_func_vals = np.sum(function_values_per_triangle,axis=1)/3
# Area of each triangle times weighted average of function values
triangle_contributions = weighted_func_vals * triangle_areas
# Summation of all triangle contributions to compute the integral
estimated_integral = np.sum(triangle_contributions)

print("Estimated integral:", estimated_integral)
```

This final part sums the contributions from all triangles. For each triangle, it averages the function values and multiplies by the triangle's area, as detailed in the integration approach explained previously. It showcases how the result can be computed.

**Alternative Approaches and Considerations**

Beyond linear interpolation on triangles, several other strategies exist, each with its own trade-offs:

*   **Higher-Order Interpolation:** Using quadratic or cubic interpolation within each triangle can provide higher accuracy, especially if the function is not well-approximated by linear forms. However, this requires additional function evaluations. In practice, I've found linear approximations often strike a balance between performance and accuracy, particularly when the mesh resolution is high.

*   **Quadrilateral Tessellation:** Instead of triangles, we could use quadrilaterals, especially if the input data forms a more structured layout. This might allow more accurate approximations in certain cases, but it is often more complex to generate a mesh of well-behaved quadrilaterals. It is worth exploring this for regular grid modifications.

*   **Voronoi Diagrams:** A less common, but occasionally useful approach, involves using the Voronoi diagram of the points. The area of each Voronoi cell can be used for integration, with a value within each cell selected appropriately.

*   **Adaptive Refinement:** An important consideration when using non-uniform grids is that you may need to apply adaptive refinement to the grid. This involves refining areas of the mesh with high function variation to improve accuracy. Error estimation techniques help to guide this process, ensuring that computational effort is allocated to the parts of the integral that need it.

**Resource Recommendations**

To further explore this topic, I recommend resources covering:

*   **Computational Geometry:** Materials detailing triangulation algorithms, such as Delaunay and constrained Delaunay triangulation.
*   **Numerical Integration:** Textbooks focusing on numerical methods for integration and quadrature, including discussion of adaptive quadrature.
*   **Finite Element Methods:** Resources covering finite element methods often detail how numerical integration is performed over meshes of triangles and quadrilaterals, and how higher-order approximations are used.

These resources can provide a deeper understanding of the mathematical foundations and practical implementations of these numerical integration methods. The choice of the correct method will depend on the specific problem, desired accuracy, and available computational resources.
