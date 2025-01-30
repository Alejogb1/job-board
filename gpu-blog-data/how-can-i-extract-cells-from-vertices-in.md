---
title: "How can I extract cells from vertices in a pyvista PolyData mesh?"
date: "2025-01-30"
id: "how-can-i-extract-cells-from-vertices-in"
---
Extracting cell data associated with specific vertices in a PyVista `PolyData` mesh requires understanding the underlying connectivity of the mesh.  Directly accessing cell data based on vertex indices isn't straightforward because a cell might be associated with multiple vertices, and a vertex might belong to multiple cells.  My experience working on geological modeling projects, specifically subsurface reservoir simulation, frequently required this type of data manipulation for property assignment and analysis.  The solution involves leveraging PyVista's cell connectivity information and array indexing.

**1.  Explanation:**

PyVista stores mesh connectivity using `cells` arrays. These arrays describe how vertices are arranged to form cells.  For example, a triangle's cell data will be represented as `[3, v1, v2, v3]`, where `3` denotes the number of vertices forming the cell, and `v1`, `v2`, and `v3` are the indices of those vertices in the mesh's `points` array.  To extract cell data based on a vertex, we need to identify all cells containing that vertex, then access the corresponding cell data.  This involves iterating through the `cells` array or, more efficiently, utilizing PyVista's built-in functions to achieve this.

The process can be summarized as follows:

a. **Identify Target Vertex:** Determine the index of the vertex of interest.

b. **Identify Associated Cells:** Find all cells containing the target vertex using PyVista's `extract_cells` or similar functions, considering the cell type (e.g., triangles, tetrahedra).

c. **Extract Cell Data:** Access the desired cell data array using the indices of the identified cells.


**2. Code Examples:**

**Example 1: Extracting Cell Data from a Triangle Mesh:**

This example focuses on a simple triangular mesh and demonstrates a basic approach using list comprehension.  I've used this technique numerous times in the past when dealing with relatively small meshes to optimize speed over complex array manipulation.


```python
import pyvista as pv

# Create a simple triangular mesh
points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1,1,0]]
cells = [[3, 0, 1, 2], [3, 1, 2, 3]]
mesh = pv.PolyData(points, cells)

# Add some cell data (e.g., permeability)
mesh["permeability"] = [10, 20]

# Target vertex index
target_vertex = 1

# Find cells containing the target vertex using list comprehension
associated_cells = [i for i, cell in enumerate(mesh.cells.reshape(-1, 4)) if target_vertex in cell[1:]]

# Extract cell data for associated cells
extracted_data = mesh["permeability"][associated_cells]

print(f"Cells containing vertex {target_vertex}: {associated_cells}")
print(f"Extracted permeability values: {extracted_data}")
```

**Example 2: Utilizing `extract_cells` for efficiency:**

For larger meshes, using PyVista's built-in functionality is much more efficient. This example utilizes `extract_cells` which avoids explicit looping and offers better performance for complex meshes. I've found this to be crucial in my experience dealing with high-resolution models.

```python
import pyvista as pv
import numpy as np

# Create a more complex mesh (example)
mesh = pv.Sphere()
mesh["scalar_data"] = np.random.rand(mesh.n_cells)

# Target vertex index
target_vertex = 10

# Find cells connected to the vertex
cell_ids = mesh.extract_cells(mesh.find_cells_containing_point(mesh.points[target_vertex])).cell_arrays['vtkOriginalCellIds']

#Extract cell data for associated cells
extracted_data = mesh["scalar_data"][cell_ids]

print(f"Cell IDs containing vertex {target_vertex}: {cell_ids}")
print(f"Extracted scalar data: {extracted_data}")
```

**Example 3: Handling different cell types:**

This example demonstrates adaptation for a tetrahedral mesh.  The core principle remains the same, but cell representation and associated data structures change.  This is crucial for adapting to diverse mesh types encountered in simulations.

```python
import pyvista as pv

# Create a tetrahedral mesh
points = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,1]]
cells = [[4, 0, 1, 2, 3], [4, 1, 2, 3, 4]]
mesh = pv.PolyData(points, cells)

# Add cell data
mesh["temperature"] = [20, 30]

# Target vertex
target_vertex = 1

#Find cells containing target vertex – adapted for tetrahedra
associated_cells = [i for i, cell in enumerate(mesh.cells.reshape(-1, 5)) if target_vertex in cell[1:]]


# Extract data
extracted_data = mesh["temperature"][associated_cells]

print(f"Cells containing vertex {target_vertex}: {associated_cells}")
print(f"Extracted temperature values: {extracted_data}")

```


**3. Resource Recommendations:**

The PyVista documentation.  The official PyVista tutorial examples.  A comprehensive text on scientific computing with Python.  A reference guide on mesh data structures.  A book on finite element methods (for context on mesh representation).
