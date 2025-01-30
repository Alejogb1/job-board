---
title: "How can Xbim geometry be leveraged for GPU acceleration?"
date: "2025-01-30"
id: "how-can-xbim-geometry-be-leveraged-for-gpu"
---
The bottleneck in processing complex building information models (BIM) lies significantly in the visualization stage, specifically when rendering large, geometrically-rich models. Traditional CPU-based rendering struggles with the sheer volume of triangles, especially in interactive applications. Leveraging the GPU, designed for parallel processing, is the key to achieve real-time frame rates and smooth user experience when working with such data. Xbim, as a C# library for handling IFC data, doesn't directly provide GPU rendering capabilities; instead, it requires a pipeline approach involving extracting relevant geometrical data and utilizing dedicated GPU rendering APIs.

My experience developing a BIM viewer for a large-scale infrastructure project highlighted these limitations vividly. Initially, we attempted to render an entire model in our Unity-based application using CPU-based processing. Performance was inadequate, with frame rates dropping below 5 FPS even on high-end workstations. This prompted us to explore GPU acceleration through a custom rendering pipeline.

The fundamental concept is to move triangle data – vertices, normals, and texture coordinates – from system memory to the GPU. The GPU then uses vertex and fragment shaders, specialized programs written in languages like HLSL or GLSL, to transform and render this data into pixels. This process avoids CPU bottlenecks, allowing for significantly faster rendering. Xbim’s `XbimGeometry` class provides the tools to extract these meshes as triangles; these are then converted into a format digestible by GPU rendering APIs.

The most common approach is to retrieve the triangle data, often as a list or an array of `float` or `Vector3` elements, which is then structured for transfer to the GPU. A rendering engine like Unity, Unreal Engine, or raw DirectX or OpenGL facilitates this data transfer and rendering. We adopted a hybrid approach with Unity, using it for the high-level rendering management and UI, while custom code handled the Xbim to GPU data conversion and optimization.

Here’s a basic illustration of how one could begin extracting triangle data from Xbim using its `XbimGeometry` and `XbimMesh` objects.

```csharp
using Xbim.Common.Geometry;
using Xbim.Ifc;
using Xbim.Ifc2x3;
using System.Collections.Generic;
using System.Linq;

public static class XbimMeshExtractor
{
    public static List<TriangleData> ExtractTriangles(IfcStore model)
    {
        var triangleDataList = new List<TriangleData>();
        var geomStore = model.GeometryStore;
        var meshTransform = XbimMatrix3D.Identity;

        foreach (var representationItem in model.Instances.Where<IfcProductRepresentation>(p => p.Representations != null).SelectMany(p => p.Representations).SelectMany(r => r.Items).OfType<IfcShapeRepresentation>())
        {
                foreach (var shapeAspect in representationItem.Items.OfType<IfcShapeAspect>())
                {
                    var product = shapeAspect.ProductPlacement;
                     meshTransform = product.Matrix;

                    if (shapeAspect.ShapeRepresentation is IfcShapeRepresentation shapeRep) {
                        foreach(var styledItem in shapeRep.Items.OfType<IfcStyledItem>()) {
                             if(styledItem.Item is IfcGeometricRepresentationSubContext subContext ){
                                 foreach(var rep in subContext.Representations){
                                    foreach (var geometryItem in rep.Items)
                                    {
                                            if (geometryItem is IfcMappedItem mappedItem) {
                                                var item = mappedItem.MappingSource.MappedRepresentation.Items[0] ;
                                                    if (item is IfcSolidModel solidModel) {
                                                         var x = geomStore.GetShape(solidModel.ToShape());
                                                        var mesh = x.GetMesh(meshTransform);
                                                       if(mesh != null)
                                                        {
                                                            triangleDataList.AddRange(ExtractMeshTriangles(mesh));
                                                        }
                                                    }

                                            }
                                            else if (geometryItem is IfcSolidModel solidModel){
                                                        var x = geomStore.GetShape(solidModel.ToShape());
                                                        var mesh = x.GetMesh(meshTransform);
                                                      if(mesh != null)
                                                        {
                                                             triangleDataList.AddRange(ExtractMeshTriangles(mesh));
                                                        }
                                            }
                                    }
                                }


                            }
                             else if (styledItem.Item is IfcSolidModel solidModel){
                                  var x = geomStore.GetShape(solidModel.ToShape());
                                   var mesh = x.GetMesh(meshTransform);

                                if(mesh != null)
                                    {
                                         triangleDataList.AddRange(ExtractMeshTriangles(mesh));
                                    }
                             }



                        }
                    }




                }
        }

        return triangleDataList;
    }


    private static List<TriangleData> ExtractMeshTriangles(XbimMesh mesh)
    {
          var triangleDataList = new List<TriangleData>();
          for (int i = 0; i < mesh.TriangleCount; i++)
            {
                var triangle = mesh.GetTriangle(i);
                    var triangleData = new TriangleData{
                        Vertices = new [] {mesh.Vertices[triangle.V1].ToPoint3D(), mesh.Vertices[triangle.V2].ToPoint3D(), mesh.Vertices[triangle.V3].ToPoint3D()},
                        Normals = new [] {mesh.Normals[triangle.V1].ToVector3D(), mesh.Normals[triangle.V2].ToVector3D(), mesh.Normals[triangle.V3].ToVector3D()},
                       //TexCoords = new Vector2[] {mesh.TexCoords[triangle.V1], mesh.TexCoords[triangle.V2], mesh.TexCoords[triangle.V3]}
                    };
                triangleDataList.Add(triangleData);


            }
      return triangleDataList;

    }

    public class TriangleData {
        public  XbimPoint3D[] Vertices {get; set; }
        public XbimVector3D[] Normals {get; set;}
        //public Vector2[] TexCoords {get; set;}
    }


}
```

This code iterates through the IFC model, extracts `IfcSolidModel` instances, fetches their associated `XbimMesh`, and then converts the triangles into a custom struct, `TriangleData`, encapsulating vertex and normal information. This data needs to be further processed into a format suitable for the specific rendering API you choose. Note that texture coordinates are commented out for brevity. The `mesh.GetTriangle` method retrieves the indices of vertices, and then those indices are used to access the vertex and normal data. This illustrates the core of the extraction process and is necessary before data can be sent to the GPU.

Next, the extracted triangle data has to be structured in a manner consumable by the GPU. Here's a basic example of how vertex, normal, and texture coordinate data would be transferred to the GPU in Unity using its `Mesh` object:

```csharp
using UnityEngine;
using Xbim.Common.Geometry;
using System.Collections.Generic;
using System.Linq;
public class UnityMeshGenerator : MonoBehaviour
{
    public Material Material;

    public void GenerateMesh(List<XbimMeshExtractor.TriangleData> triangleDataList)
    {
        if (triangleDataList.Count == 0) return;
         var mesh = new Mesh();


       List<Vector3> vertices = new List<Vector3>();
        List<Vector3> normals = new List<Vector3>();


        foreach (var triangle in triangleDataList)
        {
            vertices.Add(ToVector3(triangle.Vertices[0]));
             vertices.Add(ToVector3(triangle.Vertices[1]));
              vertices.Add(ToVector3(triangle.Vertices[2]));

                normals.Add(ToVector3(triangle.Normals[0]));
                normals.Add(ToVector3(triangle.Normals[1]));
                normals.Add(ToVector3(triangle.Normals[2]));
        }




        mesh.vertices = vertices.ToArray();
        mesh.normals = normals.ToArray();

       int[] triangles = Enumerable.Range(0, vertices.Count).ToArray();
        mesh.triangles = triangles;

         var meshFilter = gameObject.AddComponent<MeshFilter>();
        meshFilter.mesh = mesh;
        var meshRenderer = gameObject.AddComponent<MeshRenderer>();
        meshRenderer.material = Material;
    }

     private static Vector3 ToVector3(XbimPoint3D point)
    {
        return new Vector3((float)point.X, (float)point.Y, (float)point.Z);
    }
        private static Vector3 ToVector3(XbimVector3D point)
    {
        return new Vector3((float)point.X, (float)point.Y, (float)point.Z);
    }
}
```

This code converts the `XbimPoint3D` and `XbimVector3D` data to `UnityEngine.Vector3` types and creates a new Unity `Mesh` by pushing the data onto it, as well as creating a game object and rendering it. The triangle array specifies the order in which the vertices need to be connected to form triangles. In the interest of simplicity I did not implement texture coordinates, however, it would follow a similar pattern.

Finally, to maximize GPU throughput, techniques like vertex buffer objects (VBOs) and indexed drawing should be utilized. Instead of re-uploading vertex data to the GPU each frame, VBOs store this data in GPU memory once and allow the GPU to access it directly. Similarly, indexed drawing uses an index buffer to specify which vertices form the triangles, reducing memory usage and vertex buffer bandwidth. Here is a simplified code snippet of how a VBO would be constructed in OpenGL.

```c++
#include <GL/glew.h>
#include <vector>
// Assume TriangleData struct is defined and populated similar to previous C# example

void createVBO(const std::vector<TriangleData>& triangles, GLuint& vbo, GLuint& ibo, int &numTriangles)
{
    numTriangles = triangles.size();
    std::vector<float> vertices;
    std::vector<float> normals;
     std::vector<unsigned int> indices;

        for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& triangle = triangles[i];
        for(int v =0; v<3; ++v){
           vertices.push_back( (float)triangle.Vertices[v].X);
            vertices.push_back( (float)triangle.Vertices[v].Y);
            vertices.push_back( (float)triangle.Vertices[v].Z);

             normals.push_back((float)triangle.Normals[v].X);
            normals.push_back((float)triangle.Normals[v].Y);
            normals.push_back((float)triangle.Normals[v].Z);

              indices.push_back(i*3 + v);
         }
    }

    // Vertex Buffer Object
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    //Index Buffer Object
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Unbind Buffers
     glBindBuffer(GL_ARRAY_BUFFER, 0);
     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
```

This C++ code snippet shows the process of creating VBOs and IBOs. Note the usage of OpenGL functions like `glGenBuffers`, `glBindBuffer` and `glBufferData`.  The index buffer is the `indices` vector and the vertex data consists of interleaved vertex and normal data. This approach is much more efficient than sending the data each frame. These buffers can be bound in the render loop via  `glBindBuffer` for drawing.

Implementing GPU acceleration for Xbim geometry rendering requires a multi-stage process of data extraction using Xbim libraries, format conversion, GPU data uploading via API calls, and efficient data management through techniques like VBOs and IBOs. Optimization also requires careful consideration of data structures, and batching to reduce draw calls.

For further information on this process, explore the documentation for DirectX and OpenGL, and related literature on rendering optimizations and techniques. The book "Real-Time Rendering" by Tomas Akenine-Möller et al., provides a comprehensive overview of modern rendering pipelines. Additionally, examining available sample code for frameworks such as Unity or Unreal can assist in understanding the specifics of GPU data transfer and API integration. The Khronos Group provides extensive documentation on the OpenGL API.
