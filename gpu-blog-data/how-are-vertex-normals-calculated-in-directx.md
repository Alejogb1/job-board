---
title: "How are vertex normals calculated in DirectX?"
date: "2025-01-26"
id: "how-are-vertex-normals-calculated-in-directx"
---

In DirectX, the accurate and efficient calculation of vertex normals is critical for realistic lighting and shading in 3D rendering. Unlike fragment (pixel) normals that can be calculated per-pixel, vertex normals are computed once per vertex and then interpolated across a polygon’s surface, directly impacting the final appearance of the rendered object.

The fundamental method involves analyzing the surrounding geometry to estimate the normal vector at each vertex. I’ve found, over years of working on graphics applications, that a common and robust approach is the averaging of face normals. Here's how I typically approach this:

1.  **Face Normal Calculation:** Initially, for each polygon in a mesh (most often triangles), we need to calculate the face normal. Given three vertices of a triangle, *v1*, *v2*, and *v3*, we can form two vectors, *edge1* = *v2* - *v1* and *edge2* = *v3* - *v1*. The cross product of these two vectors, `edge1 x edge2`, yields a vector perpendicular to the plane of the triangle. This resultant vector, which we can then normalize to unit length, becomes the face normal. The direction of the normal dictates the front face orientation, which is particularly important for backface culling.

2.  **Vertex Normal Accumulation:** Once face normals are available, the vertex normal calculation proceeds by accumulating the face normals of all faces adjacent to a given vertex. For a single vertex, we iterate through every polygon that uses that vertex.  For each polygon, we add its *normalized* face normal to an accumulator vector associated with that vertex. In this phase, I've often discovered the need for some robust algorithms. If not handled correctly, the accumulation process will cause the normal to deviate from the actual average.

3.  **Normalization and Refinement:** After accumulating all relevant face normals for a vertex, we divide the accumulated vector by the number of contributing face normals to get the average direction. Then this averaged vector *must* be normalized to unit length, resulting in the final vertex normal. This normalized vertex normal is then typically stored with the vertex data for use in shader programs. This ensures the normal used for lighting calculations does not affect the intensity of the light.

Here's a breakdown of that process in code using HLSL structures (this is a conceptual code example and not DirectX API specific):

```hlsl
// Example Structs representing a Vector and a Triangle
struct Vector3 {
    float x;
    float y;
    float z;
};

struct Triangle {
    Vector3 v1;
    Vector3 v2;
    Vector3 v3;
};

// Helper function to calculate the face normal of a triangle.
Vector3 CalculateFaceNormal(Triangle triangle)
{
    Vector3 edge1 = { triangle.v2.x - triangle.v1.x, triangle.v2.y - triangle.v1.y, triangle.v2.z - triangle.v1.z };
    Vector3 edge2 = { triangle.v3.x - triangle.v1.x, triangle.v3.y - triangle.v1.y, triangle.v3.z - triangle.v1.z };
	
    Vector3 normal;
    normal.x = (edge1.y * edge2.z) - (edge1.z * edge2.y);
    normal.y = (edge1.z * edge2.x) - (edge1.x * edge2.z);
    normal.z = (edge1.x * edge2.y) - (edge1.y * edge2.x);

	float length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    normal.x /= length;
	normal.y /= length;
	normal.z /= length;
    return normal;
}
```

*Commentary:* The preceding HLSL fragment demonstrates how the cross product of two edges from a triangle creates a perpendicular face normal. The code carefully handles the coordinate differences and subsequent calculations in the cross product and ensures the face normal is normalized. While this code is simplified, it illustrates how face normals are constructed from vertex locations.

The next piece of HLSL pseudo-code exemplifies the accumulation process and normalization:

```hlsl
// Example Structs
struct Vertex {
    Vector3 position;
    Vector3 normal;
};
struct Mesh {
    Triangle triangles[NUM_TRIANGLES];
    Vertex vertices[NUM_VERTICES];
};


// Accumulate face normals for each vertex and compute average normal
void CalculateVertexNormals(inout Mesh mesh)
{
    for (uint vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
    {
        Vector3 accumulatedNormal = { 0.0f, 0.0f, 0.0f };
        int numAdjacentFaces = 0;

        for (uint triangleIndex = 0; triangleIndex < NUM_TRIANGLES; triangleIndex++)
        {
            Triangle triangle = mesh.triangles[triangleIndex];
			// Check if the vertex is part of the triangle
            if(vertexIndex == GetVertexIndexInTriangle(triangle, mesh.vertices)
			    ||  vertexIndex == GetVertexIndexInTriangle(triangle, mesh.vertices) +1
				|| vertexIndex == GetVertexIndexInTriangle(triangle, mesh.vertices)+2 )
            {
                Vector3 faceNormal = CalculateFaceNormal(triangle);
				accumulatedNormal.x += faceNormal.x;
				accumulatedNormal.y += faceNormal.y;
				accumulatedNormal.z += faceNormal.z;
                numAdjacentFaces++;
            }
        }

        // Average accumulated normals and normalize
        if (numAdjacentFaces > 0)
        {
           accumulatedNormal.x /= (float)numAdjacentFaces;
           accumulatedNormal.y /= (float)numAdjacentFaces;
           accumulatedNormal.z /= (float)numAdjacentFaces;

		    float length = sqrt(accumulatedNormal.x * accumulatedNormal.x + accumulatedNormal.y * accumulatedNormal.y + accumulatedNormal.z * accumulatedNormal.z);
            if(length > 0)
            {
               accumulatedNormal.x /= length;
               accumulatedNormal.y /= length;
               accumulatedNormal.z /= length;

               mesh.vertices[vertexIndex].normal = accumulatedNormal;
            }
           
        }
    }
}
```

*Commentary:* This snippet shows the crucial part of looping through all triangles to find the faces that use a given vertex. The accumulated normal is finally averaged and normalized, and assigned back to the vertex normal. Note that for the sake of simplicity, `GetVertexIndexInTriangle` is not implemented but implies the logic for checking the triangle’s vertex indices and finding if a vertex in the vertex array is used in the current triangle under evaluation.  I find that this is commonly one of the most expensive steps in mesh processing. Optimizations at this point are vital.

Below we have a simple illustration of the vertex structure before and after the calculation of the normals. The example uses the HLSL declaration of the struct in the two code snippets above.

```hlsl
// Sample use of the code
void ExampleUsage()
{
    Mesh myMesh; // In practice would be populated with real mesh data
    // Here I assume that the Mesh was populated from a 3D model 
    // before calling this function

    // Initialize the mesh, in particular populate vertices and triangles
    // ...

    // Calculate the vertex normals
    CalculateVertexNormals(myMesh);

    // At this point, each vertex in myMesh.vertices will
	// have a valid normal vector populated in myMesh.vertices.normal

	// The mesh data with its vertex normals can now be used in rendering.
	//...
}
```

*Commentary:* The last code snippet is conceptual in nature and shows how the previous functions would be invoked. In a real application, one needs to parse the mesh and store vertex position and connectivity information before calling the vertex calculation function. After the call, the vertex’s normal attribute is now populated for use in the rendering pipeline.

While the averaging method is common, there are other methods. Some involve weighting face normals based on the angle of the face at that vertex, which can provide smoother results on curved surfaces. Also, when creating procedural meshes or handling dynamically changing geometry, I use code structures that can operate on the fly, sometimes with a trade-off in performance or with the use of compute shaders that are designed for this type of parallel processing on the GPU. However, the foundation remains as illustrated here; calculate face normals then average for a vertex.

For further study, I’d suggest investigating resources on Computer Graphics and 3D Modeling. Texts covering topics like ‘Surface Normals,’ ‘Mesh Processing,’ and ‘Shading Algorithms’ can provide a deeper understanding. Additionally, documentation on specific 3D modeling packages often includes information on how they compute these vectors. Further, textbooks on DirectX and rendering pipelines will go into specific details of these types of calculations. These resources usually detail the mathematical concepts and the implementations using DirectX or other graphics APIs.
