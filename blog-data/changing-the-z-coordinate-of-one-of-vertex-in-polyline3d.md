---
title: "changing the z coordinate of one of vertex in polyline3d?"
date: "2024-12-13"
id: "changing-the-z-coordinate-of-one-of-vertex-in-polyline3d"
---

 so you're wrestling with the z-coordinate of a vertex in a Polyline3D eh I've been there believe me This isn't some obscure corner case it’s a fundamental thing you deal with all the time when you’re manipulating 3D geometry so let's get this sorted out

First off let's assume you've got some kind of data structure holding this Polyline3D I'm not going to pretend I know exactly how yours is setup everyone rolls their own I’ve seen everything from a simple list of 3D points to more complex objects with associated metadata so I'm going generic but I'll also add some concrete examples later

The crux of the matter is you need to identify the specific vertex and then just alter its z-value nothing more nothing less Don't overthink it I remember the first time I worked with this kind of 3D stuff I was all about matrices and transforms and I made things way more complicated than they needed to be Turns out straightforward is often the best path

Let’s say you have it as a simple list of 3D points something like this python style example

```python
class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

polyline = [
    Point3D(1, 2, 3),
    Point3D(4, 5, 6),
    Point3D(7, 8, 9)
]
```

Now to change the z of the second point index 1 remember we start counting from 0 right

```python
polyline[1].z = 10 # Set the z value of the second vertex to 10
```

That's it seriously no magic This gets you the basic operation you want The other thing I want to say you can do if you want you can modify it with an indexing function if it helps or if it makes your code more maintainable.

```python
def set_z(polyline, vertex_index, new_z):
    if 0 <= vertex_index < len(polyline):
        polyline[vertex_index].z = new_z
    else:
        print("Error invalid index")

set_z(polyline, 1, 10)
```

Now this is all well and good if you're working with a basic list or similar data structure. If you’re dealing with a mesh or some fancier structure in 3D engine this all scales in exactly the same way you still need to locate the data and then change the z-value.

For instance if you were using something in Unity well you would have a `Mesh` object with vertex data stored in `vertices` an array of `Vector3` and its same concept

```csharp
// Unity C# code example
using UnityEngine;

public class PolylineModifier : MonoBehaviour
{
    public MeshFilter meshFilter; // Assign your MeshFilter in the Inspector
    public int vertexIndex = 1; // The index of the vertex you want to change
    public float newZValue = 10f; // The new Z value

    void Update()
    {
        if (meshFilter == null)
        {
           Debug.LogError("MeshFilter not assigned");
           return;
        }

        Mesh mesh = meshFilter.mesh;
        if (mesh == null)
        {
           Debug.LogError("Mesh not found");
           return;
        }

        Vector3[] vertices = mesh.vertices;
        if (vertices.Length <= vertexIndex)
        {
            Debug.LogError("Invalid vertex index");
            return;
        }
         
        Vector3 vertex = vertices[vertexIndex];
        vertex.z = newZValue;
        vertices[vertexIndex] = vertex;
        mesh.vertices = vertices;
    }
}
```

Note the core idea remains the same: get the vertices data modify the specific z value of one of them set it back and you're good to go. If you’re using something like OpenGL, Direct3D, Vulkan the process is also conceptually the same. The details of how you access the vertex buffer and update the data might be different depending on the library you're using. I’m pretty sure I don’t need to mention that you need to rebuild or draw the mesh after that.

Now a very critical point one that I have missed in the past I mean more than once more than twice probably even more than three times because I am not that great at keeping track of things and this is crucial: if you modify vertex data the way I've described here you are directly changing the geometry data of the object, so you have to watch out for side effects because that affects the entire mesh if they are not copies or if they are not managed by some other system you need to keep track of your copies or it is gonna be a debugging nightmare one that can make you hate your life and make you have a terrible week like it happened to me a few weeks ago or last month it’s all blurry now.

I saw many a good developer struggle because they were changing the data of a shared object without knowing it so watch out for that one that’s probably one of the most important lessons I've learned in my time dealing with this kind of stuff.

So let's talk about performance because well I've learned the hard way optimization is crucial. If you're only changing one or two vertices here and there a direct update like I've shown is usually fine but if you're altering a lot of points very very often like you are doing some dynamic changes or some mesh deformation then that can kill performance quickly. You should instead of updating every single frame you can update only the values when they change if they are managed by a reactive system and that is much more efficient. Or what you can do is batch updates and calculate those vertices and update all at once instead of doing it every frame by frame.

And here is the joke part here is where the real optimization happens get it… hahahaha ok. Ok. Moving on…

If you are interested in the mathematical background for this I recommend these resources I have used in the past and they have helped me. "Mathematics for 3D Game Programming and Computer Graphics" by Eric Lengyel it's a classic and has a section on linear algebra with all you need to know about vector manipulation. And for more specific to data structures I would recommend “Data Structures and Algorithms in C++” by Adam Drozdek even if the language it uses is C++ the data structures are language agnostic. And for even more low level stuff like vertex buffers you should probably learn more on the docs of your respective API or library they will have plenty of details on how vertex buffers are created and managed.

So that's basically it in my experience you need to pinpoint your vertex then simply change the z component it’s pretty basic stuff but you need to be extra careful with data structures copies and performance I hope that this is of some help. Remember to not overcomplicate it. Good luck and if you've got any other troubles feel free to ask.
