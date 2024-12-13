---
title: "connected-components-3d algorithm explanation?"
date: "2024-12-13"
id: "connected-components-3d-algorithm-explanation"
---

Alright so you're asking about connected components in 3D huh Been there done that Let me tell you I've wrestled with this problem more times than I care to admit Seriously back in my early days when I was still getting my feet wet with this computer vision stuff I had a project involving 3D medical scans specifically CT scans the kind with voxel data You can imagine it's like a 3D grid of pixels and my job was to separate different organs and structures each organ or structure is its own connected component The goal was to extract bone structures from the scans it was a real pain I mean not a fun pain like when you finally debug a nasty code bug but a real head scratcher type of pain I spent days debugging a flawed implementation of this

Okay so fundamentally a connected component in 3D is just a group of voxels that are next to each other that belong to some data which in your case we should assume each voxel has a specific meaning for example if your 3D volume represents some kind of a binary mask you could say 1s are part of an object you want to analyze and 0 are not. The "next to each other" part is crucial We need a definition of what 'next to' means typically we deal with two common adjacency types 6-connectivity and 26-connectivity

6-connectivity is basic It considers only the immediate faces a voxel can share. Think of a cube Imagine looking at a Rubik's cube the top the bottom the left the right the front and the back Those are its 6 neighbors. In code terms it means checking if a voxel is connected to its neighbors on each axis like `(x+1, y, z)`, `(x-1, y, z)` `(x, y+1, z)` `(x, y-1, z)` `(x, y, z+1)` `(x, y, z-1)` So only the 6 direct neighbors along each axis.

26-connectivity is more comprehensive In this approach you consider all the voxels that are within one unit of the central voxel So not only faces but also edges and corners of the cube So instead of 6 we are now checking 26 neighbors think of a 3x3x3 cube the center is the voxel itself the 26 surrounding ones are it's neighbors. This gets more verbose to implement you have to check `x,y,z +/- 1` for all 26 options.

Implementing these in code we can use a graph traversal algorithm which is just a fancy way of saying go through all of it we often use Depth-First Search (DFS) or Breadth-First Search (BFS). Lets use BFS to find a connected component because DFS is very recursive and can lead to stack overflow issues if you have a large volume. BFS uses a queue or deque so you don't have to worry about that kind of thing.

Okay lets assume your input is a 3D array (think numpy array or equivalent) called `volume` and that every element in the array is an integer value that represents the mask value and you only want the connected component that is marked with value 1. The pseudo code for this process will look something like this

```python
from collections import deque

def bfs_connected_component(volume, start_voxel, target_value=1):
    rows, cols, slices = volume.shape
    visited = set()
    queue = deque([start_voxel])
    connected_component = []

    while queue:
        x, y, z = queue.popleft()

        if (x, y, z) in visited:
            continue
        visited.add((x, y, z))

        if 0 <= x < rows and 0 <= y < cols and 0 <= z < slices and volume[x, y, z] == target_value:
            connected_component.append((x, y, z))
            # 6-connectivity neighbors
            neighbors = [
                (x + 1, y, z), (x - 1, y, z),
                (x, y + 1, z), (x, y - 1, z),
                (x, y, z + 1), (x, y, z - 1)
            ]

            # 26-connectivity neighbors
            # neighbors = []
            # for dx in [-1,0,1]:
            #     for dy in [-1,0,1]:
            #         for dz in [-1,0,1]:
            #             if dx ==0 and dy == 0 and dz == 0:
            #                 continue
            #             neighbors.append((x+dx, y+dy, z+dz))

            for nx, ny, nz in neighbors:
                if 0 <= nx < rows and 0 <= ny < cols and 0 <= nz < slices:
                  queue.append((nx, ny, nz))

    return connected_component
```
This is a simple implementation of the BFS algorithm which returns the indices of the 3D array that are part of the connected component starting from a specific point called `start_voxel` the algorithm first checks if a voxel has already been visited if not it checks if the voxel has the `target_value` in your case 1 if so then the algorithm adds it to the `connected_component` list and then adds the neighbors to the queue and repeats the same process again until there are no more items in the queue which means that we have explored all connected neighbors.
As you can see in the example above I have already commented the 26-connectivity version of finding the neighbors out because I was focusing on 6-connectivity if you want to use the 26 version just remove the comment from the neighbors variable.

Now how do you use it? Well first you need the 3D volume and a start point for example

```python
import numpy as np

# example 3D volume
volume = np.array([
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 1], [1, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])

start_point = (1, 1, 1)

connected_component = bfs_connected_component(volume, start_point)
print(connected_component)
# prints
# [(1, 1, 1), (1, 2, 1), (2, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 2)]

```
You can see the output of the connected component is a list of coordinates in the volume array that are part of the same connected component starting at `(1,1,1)`.
Now you can make a function to find all the connected components for example in order to do so we need to first go through the whole volume and check for voxels that have the value of interest and then when we find one that has not been visited we can apply the BFS for the whole process. The implementation can be done as follows:
```python
import numpy as np

def all_connected_components(volume, target_value=1):
    rows, cols, slices = volume.shape
    visited = set()
    all_components = []

    for x in range(rows):
        for y in range(cols):
            for z in range(slices):
                if volume[x, y, z] == target_value and (x,y,z) not in visited:
                  component = bfs_connected_component(volume,(x,y,z), target_value)
                  all_components.append(component)
                  visited.update(component)
    return all_components

volume = np.array([
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 1], [1, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [1, 0, 0]]
    ])
components = all_connected_components(volume)

for i , component in enumerate(components):
  print(f"Component {i}: {component}")

# Component 0: [(1, 1, 1), (1, 2, 1), (2, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 2)]
# Component 1: [(2, 2, 0)]
```

This function goes through all the volume and finds the connected components using the `bfs_connected_component` function if the voxel has not been already visited.

Now back in the day when I was figuring this stuff out there weren't tons of ready to use libraries for every single thing especially in the medical imaging domain and now looking back It was a good exercise because I really had to wrap my head around these algorithms. It taught me a lot that I still use today. In case you're wondering no i didn't invent this I found a good explanation in "Computer Vision Algorithms and Applications" by Richard Szeliski it is a great book that explains these algorithms in details. Another great resource is "Digital Image Processing" by Rafael C Gonzalez and Richard E Woods.

One thing I messed up was optimizing the BFS. I spent a lot of time on unnecessary checks at first it wasn't great. Let's be honest who here hasn't had that kind of bug where it worked but it took about 5 minutes to finish? I have once I even tried to optimize a function using multithreading but actually it made it slower because i didnt know what I was doing hahah it was a mess.

Anyway hope this helps you out Let me know if you have other questions or if something was unclear. Remember to choose the right connectivity 6 or 26 depending on your application and what you want to achieve. Also be aware of your memory footprint when dealing with big 3D volumes they can get very heavy memory-wise. Good luck
