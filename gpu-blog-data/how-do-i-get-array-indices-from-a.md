---
title: "How do I get array indices from a Mayavi picker object?"
date: "2025-01-30"
id: "how-do-i-get-array-indices-from-a"
---
The Mayavi picker, while powerful for interactive data selection, doesn't directly provide array indices corresponding to picked points.  Its output is fundamentally a world coordinate, requiring a reverse-mapping process based on your data structure and how it's rendered.  My experience working on high-resolution geological simulations highlighted this limitation;  efficiently identifying the original dataset element linked to a picked point necessitates careful consideration of the underlying data organization and the Mayavi pipeline.

**1.  Understanding the Mayavi Picker Output and Data Structure**

The Mayavi picker's `picked` attribute, upon a successful pick event, returns a tuple containing the picked position in world coordinates, along with other metadata.  This world coordinate is not directly translatable to an array index without knowledge of the data's spatial representation within the scene.  This crucial understanding informed my approach during a project analyzing stress fields in complex fractured rock formations, where I needed to pinpoint individual fracture points selected by the user.

The key is to bridge the gap between the world coordinates and your original data.  This depends entirely on how your data was structured and fed into Mayavi.  Common scenarios include:

* **Structured Grid Data:** Data arranged in a regular grid (e.g., using `numpy` arrays representing a 3D volume).  Here, the world coordinates can be directly mapped to array indices using simple calculations based on grid spacing.
* **Unstructured Grid Data:** Data represented by vertices and cells (e.g., using `vtkUnstructuredGrid`).  This requires a more complex spatial search, often involving algorithms like nearest neighbor searches or KD-trees to find the closest data point to the picked world coordinates.
* **Point Cloud Data:**  Data represented as a collection of points in 3D space. Similar to unstructured data, a nearest neighbor search is needed.


**2. Code Examples with Commentary**

The following examples illustrate how to retrieve array indices for different data types.  I'll assume you're already familiar with the basics of Mayavi's scene creation and picker integration.  Remember that error handling (e.g., checking for valid picks) should be integrated into a production environment.

**Example 1: Structured Grid Data**

This example demonstrates index retrieval for a regularly spaced 3D grid.

```python
import numpy as np
from mayavi import mlab

# Sample data: a 3D scalar field
x, y, z = np.mgrid[0:10:10j, 0:10:10j, 0:10:10j]
data = np.sin(x*y*z)

# Create Mayavi scene
src = mlab.pipeline.scalar_field(data)
iso = mlab.pipeline.iso_surface(src, contours=[0.5])

# Add picker
picker = iso.picker

# ... (Mayavi interaction code to trigger a pick event) ...

if picker.picked():
    # Extract world coordinates
    x_world, y_world, z_world = picker.pick_position

    # Assuming uniform grid spacing (adjust according to your data)
    grid_spacing = 1.0

    # Calculate indices (integer division for grid indexing)
    i = int(x_world // grid_spacing)
    j = int(y_world // grid_spacing)
    k = int(z_world // grid_spacing)

    print(f"Array indices: i={i}, j={j}, k={k}")
    print(f"Original data value: {data[i,j,k]}")

mlab.show()
```

**Example 2: Unstructured Grid Data (using `scipy.spatial`)**

This example uses SciPy's KDTree for efficient nearest-neighbor search within an unstructured grid.

```python
import numpy as np
from mayavi import mlab
from scipy.spatial import KDTree

# Sample unstructured data (replace with your data)
points = np.random.rand(100, 3)
data_values = np.random.rand(100)  # Values associated with each point

# Create Mayavi scene (using points3d for simplicity)
mlab.points3d(points[:,0], points[:,1], points[:,2], data_values, scale_factor=0.1)
picker = mlab.gcf().scene.picker

# ... (Mayavi interaction code to trigger a pick event) ...

if picker.picked():
    picked_point = picker.pick_position

    # Build KDTree for efficient nearest neighbor search
    kdtree = KDTree(points)
    _, index = kdtree.query(picked_point)

    print(f"Nearest neighbor index: {index}")
    print(f"Original data value: {data_values[index]}")

mlab.show()
```

**Example 3: Point Cloud Data (simplified approach)**

This example simplifies the search for point cloud data using a brute-force approach suitable for smaller datasets. For larger datasets, a KD-Tree based approach (as in Example 2) is recommended.

```python
import numpy as np
from mayavi import mlab

# Sample point cloud data
points = np.random.rand(50, 3)
mlab.points3d(points[:, 0], points[:, 1], points[:, 2])
picker = mlab.gcf().scene.picker

# ... (Mayavi interaction code to trigger a pick event) ...

if picker.picked():
    picked_point = picker.pick_position
    distances = np.linalg.norm(points - picked_point, axis=1)
    closest_index = np.argmin(distances)
    print(f"Index of closest point: {closest_index}")

mlab.show()

```

**3. Resource Recommendations**

For advanced techniques involving complex data structures and optimization strategies, consult the Mayavi documentation. The VTK documentation (on which Mayavi is built) provides detailed information on data structures and algorithms for spatial searching.  Explore the SciPy library's spatial functions and algorithms for efficient nearest-neighbor searches in higher dimensions.   Familiarize yourself with the concept of spatial indexing (e.g., KD-trees, Octrees) for efficient handling of large datasets.  Lastly, reviewing examples of point cloud processing and visualization within Mayavi would prove beneficial.
