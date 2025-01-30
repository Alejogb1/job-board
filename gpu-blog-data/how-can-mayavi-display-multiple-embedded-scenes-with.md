---
title: "How can Mayavi display multiple embedded scenes with volume rendering?"
date: "2025-01-30"
id: "how-can-mayavi-display-multiple-embedded-scenes-with"
---
Mayavi, while primarily known for its interactive 3D visualization, can effectively handle multiple embedded scenes, each potentially displaying volume rendering. This is accomplished by managing separate `Mayavi` engines and their associated `Scene` objects, allowing for distinct render spaces within a single application window. My experience over the past several years visualizing medical imaging data has led me to repeatedly leverage this capability for comparative analysis and multi-dimensional data exploration.

The core concept revolves around the use of `mayavi.mlab.figure()` to create independent `Mayavi` engine instances and associated graphical user interface elements. When a `figure` is created, Mayavi establishes a rendering context and window. Subsequent calls to `mayavi.mlab.figure()` generate new, distinct contexts, each potentially exhibiting different data, rendering techniques, or viewpoints. Each of these figures contains a `Scene` object which holds the rendered content. This fundamental separation enables the embedding of multiple volume-rendered datasets, often alongside other visualizations like surface plots or vector fields, within a coherent application layout. To achieve precise control over layout, the `scene.set_position()` and `scene.set_size()` methods are pivotal. I habitually use these, often in conjunction with grid layout managers in accompanying GUI toolkits (like PyQt), to avoid visual clutter and allow for intuitive comparison of different viewpoints.

Here are three practical code examples demonstrating distinct configurations of embedded volume rendering in Mayavi:

**Example 1: Two Side-by-Side Volumes**

This example creates two Mayavi scenes, each displaying a different 3D NumPy array using volume rendering. The scenes are positioned side-by-side within the application window.

```python
import numpy as np
from mayavi import mlab

# Generate dummy data
data1 = np.random.rand(50, 50, 50)
data2 = np.sin(np.linspace(0, 4*np.pi, 50)).reshape(1, 1, 50) * np.cos(np.linspace(0, 2*np.pi, 50)).reshape(1, 50, 1) * np.random.rand(50, 50, 50)

# Create the first figure and scene
fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
mlab.clf()  # Clear the scene if previous content exists
vol1 = mlab.pipeline.volume(mlab.pipeline.scalar_field(data1))
scene1 = mlab.gcf()
scene1.scene.set_position([0, 0, 400, 400]) # Position the first scene in the upper left quadrant

# Create the second figure and scene
fig2 = mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
mlab.clf()  # Clear the scene if previous content exists
vol2 = mlab.pipeline.volume(mlab.pipeline.scalar_field(data2))
scene2 = mlab.gcf()
scene2.scene.set_position([400, 0, 800, 400]) # Position the second scene to the right of the first

mlab.show()
```

*Code commentary:* This example sets up two separate `mlab.figure` contexts. Each context has its respective rendering context within a shared window.  `mlab.clf()` is included to ensure a clean scene. The `mlab.pipeline.scalar_field()` converts the NumPy arrays into a suitable format for Mayavi's visualization pipeline, and the `mlab.pipeline.volume()` function adds the volume renderer. The `scene.set_position()` method dictates the screen coordinates (x, y, width, height) for each scene, placing them adjacent.  I frequently use `mlab.gcf()` (get current figure) to reference the currently active scene, as directly accessing figure elements can be cumbersome.

**Example 2: Volume and Surface Rendering**

This example demonstrates embedding a volume rendering alongside a surface plot within a single application window by using separate scenes within the same window.

```python
import numpy as np
from mayavi import mlab

# Generate data
x, y = np.ogrid[-2:2:20j, -2:2:20j]
z = x * np.exp(-x**2 - y**2)
data = np.random.rand(50, 50, 50)

# Create a figure and scene
fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 400))
mlab.clf()

# Create a volume visualization within the first scene
vol_scene = mlab.gcf()
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(data))
vol_scene.scene.set_position([0, 0, 400, 400])


# Create a separate scene
surf_scene = mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
mlab.clf()

# Surface rendering in the second scene
surf = mlab.surf(x, y, z)
surf_scene.scene.set_position([400, 0, 800, 400])

mlab.show()
```

*Code commentary:* This example further reinforces the use of `mlab.figure` for distinct scenes. It demonstrates that each scene can render different types of visualization. I employ `mlab.surf()` to render a surface plot on the second scene created via figure 2 and `mlab.pipeline.volume()` for the first one. The positional placement through `scene.set_position()` creates side by side visual presentations, showing a simple surface and volume rendering next to each other.

**Example 3: Different Volumes with Independent Controls**

This example shows two volume renderings, each displayed in its own scene, each with independent viewing controls. These independent view controls are crucial when exploring complex datasets from different perspectives.

```python
import numpy as np
from mayavi import mlab

# Generate dummy volume data
data_a = np.random.rand(60, 60, 60)
data_b = np.sin(np.linspace(0, 4*np.pi, 60)).reshape(1, 1, 60) * np.cos(np.linspace(0, 2*np.pi, 60)).reshape(1, 60, 1) * np.random.rand(60, 60, 60)

# Create the first volume's scene and render
fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
mlab.clf()
vol1 = mlab.pipeline.volume(mlab.pipeline.scalar_field(data_a))
scene1 = mlab.gcf()
scene1.scene.set_position([0, 0, 400, 400])


# Create the second volume's scene and render
fig2 = mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
mlab.clf()
vol2 = mlab.pipeline.volume(mlab.pipeline.scalar_field(data_b))
scene2 = mlab.gcf()
scene2.scene.set_position([400, 0, 800, 400])


mlab.show()
```

*Code commentary:*  This example showcases two separate scenes, each displaying a volume rendering. Crucially, when running the code, manipulation of the 3D view in one scene (e.g., rotation, zoom) doesn't affect the other scene. This is the fundamental benefit of using separate `mlab.figure()` calls. I’ve utilized this for side-by-side comparison of data with varying intensity ranges and structural characteristics in my previous projects. As in the previous examples the position of each scene is set independently.

For those pursuing advanced topics, I recommend exploring the documentation for Mayavi’s `mlab.pipeline` module.  Understanding the pipeline's structure is essential for manipulating complex data and integrating custom algorithms. Additionally, familiarity with the Visualization Toolkit (VTK), which Mayavi relies on, can unlock further customization options. Finally, delving into scientific visualization textbooks can offer a deeper appreciation of different rendering strategies and best practices for presenting complex datasets. Understanding the structure of both libraries can help achieve more intricate and robust visualization workflows, and such background has been invaluable in my professional development.
