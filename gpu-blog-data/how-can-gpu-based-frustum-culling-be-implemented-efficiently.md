---
title: "How can GPU-based frustum culling be implemented efficiently?"
date: "2025-01-30"
id: "how-can-gpu-based-frustum-culling-be-implemented-efficiently"
---
Implementing frustum culling efficiently on the GPU requires a careful understanding of the graphics pipeline and leveraging its parallel processing capabilities. I’ve spent considerable time optimizing real-time rendering engines, and I’ve found that relying primarily on the GPU’s architecture for this task, rather than pre-filtering on the CPU, leads to superior performance, especially with scenes exhibiting high complexity.

The central idea behind frustum culling is identifying which objects, represented by their bounding volumes, are within the camera’s view frustum and discarding the rest. This process reduces the number of primitives that need to be processed by later pipeline stages, resulting in significant performance improvements. The challenge lies in performing this test quickly and concurrently across a vast number of objects. Moving this task to the GPU allows for massive parallelization since it can execute the same culling logic against each object simultaneously using thousands of processing cores.

The process can be broken down into these essential steps: first, define the frustum using the camera’s parameters; second, transform each object’s bounding volume into clip space; third, perform a simple test against the clip space representation of the frustum to determine visibility. Finally, generate an index buffer indicating which objects should be rendered. This last step is critical, as it avoids drawing potentially large batches of invisible geometry.

The critical consideration here is transforming the bounding volumes into clip space. This is achieved by applying the same model, view, and projection matrices to the object's bounding volume that are used to position it within the scene. Once these transformations are performed on the object's bounding volume, we can apply specific clip space tests.

Let's illustrate this with shader code examples. While a compute shader could be used, we'll focus on a vertex shader approach here, since it is more broadly accessible across different graphics APIs and can be used within conventional rendering passes for simplicity and educational purposes. I typically start using a vertex shader for culling in the beginning of a project. Later, optimization may involve a compute shader, when necessary.

**Example 1: Bounding Sphere Culling using a Vertex Shader**

The first example implements bounding sphere culling, which is relatively straightforward. The idea is to project the sphere’s center into clip space and then use its radius and position to decide if it's completely outside the view frustum. Note, I am leaving out setting up the appropriate uniforms, since that is API specific. I am also assuming the bounding sphere is a simple uniform, not based on a specific model.

```glsl
#version 450 core

layout (location = 0) in vec3 inPosition; // Object position in world space, this is not needed for culling but it is used to be able to render if visible.
layout (location = 1) in int inObjectID;  // unique object ID for this instance

layout (location = 0) out int outObjectID; // Pass through the object ID for the fragment shader
flat out int visible;

uniform mat4 modelViewProjectionMatrix;
uniform vec3 sphereCenter;  // Sphere center world space
uniform float sphereRadius; // Sphere radius

void main() {

    vec4 centerClip = modelViewProjectionMatrix * vec4(sphereCenter, 1.0);
    float radiusClip = sphereRadius * length(vec4(modelViewProjectionMatrix[0].xyz, 0.0)); //Estimate radius in clip space.
    // Note: A more correct approach would transform the bounding sphere by the inverse of the camera matrix and do the tests in eye space, for better accuracy.
    // Note: You could also transform the bounds to clip space with an approach that does not use scaling of the sphere and the position, and rather, you can consider if the sphere overlaps with all 6 planes.
    // Note: As an optimization, we are not testing all the cases, for example, we are ignoring if the sphere overlaps.

    // Clip space is (-w, w) where w is the homogeneous component. We can use this range to detect if the sphere is out of bounds on each axis.

     // Check if the sphere is completely outside any of the 6 clip space planes. We are doing simple clipping for this demonstration.

    if(centerClip.x + radiusClip < -centerClip.w || centerClip.x - radiusClip > centerClip.w ||
       centerClip.y + radiusClip < -centerClip.w || centerClip.y - radiusClip > centerClip.w ||
       centerClip.z + radiusClip < -centerClip.w || centerClip.z - radiusClip > centerClip.w) {
        visible = 0; // Not visible
    } else {
        visible = 1; // Visible, at least partially
        gl_Position = modelViewProjectionMatrix * vec4(inPosition, 1.0); // Pass through object's final position
    }

    outObjectID = inObjectID; // Pass through object ID, needed to render this object in the fragment shader
}
```

This code performs the necessary transformations and tests in the vertex shader. The `visible` output variable will be used by the geometry shader (or a render pass) to decide whether to draw the object based on its visibility flag. Note that a more accurate sphere clipping involves more computation, but here we use a simplified version.

**Example 2: Bounding Box Culling using a Geometry Shader**

Although frustum culling is often done within the vertex stage, using a geometry shader, offers more control. The following example illustrates box culling, where the corners of the bounding box are transformed into clip space and tested. This example assumes we have the box defined by a minimum and maximum point in world space. Also, note that this method is not optimal in some edge cases where the bounding box is small but at the very edge of the screen. More complex methods involve testing against all 6 planes of the view frustum. This example is provided to showcase the use of a geometry shader.

```glsl
#version 450 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

layout (location = 0) in vec3 inPosition[]; // Object position in world space, this is not needed for culling but it is used to be able to render if visible.
layout (location = 1) in int inObjectID[];  // unique object ID for this instance

layout (location = 0) out int outObjectID;
flat out int visible;
uniform mat4 modelViewProjectionMatrix;
uniform vec3 minBoundingBox;
uniform vec3 maxBoundingBox;


void main() {

    vec4 clipVertices[8]; // store the 8 vertices of the transformed bounding box

     // transform the box vertices into clip space
     clipVertices[0] = modelViewProjectionMatrix * vec4(minBoundingBox.x, minBoundingBox.y, minBoundingBox.z, 1.0);
     clipVertices[1] = modelViewProjectionMatrix * vec4(maxBoundingBox.x, minBoundingBox.y, minBoundingBox.z, 1.0);
     clipVertices[2] = modelViewProjectionMatrix * vec4(maxBoundingBox.x, maxBoundingBox.y, minBoundingBox.z, 1.0);
     clipVertices[3] = modelViewProjectionMatrix * vec4(minBoundingBox.x, maxBoundingBox.y, minBoundingBox.z, 1.0);
     clipVertices[4] = modelViewProjectionMatrix * vec4(minBoundingBox.x, minBoundingBox.y, maxBoundingBox.z, 1.0);
     clipVertices[5] = modelViewProjectionMatrix * vec4(maxBoundingBox.x, minBoundingBox.y, maxBoundingBox.z, 1.0);
     clipVertices[6] = modelViewProjectionMatrix * vec4(maxBoundingBox.x, maxBoundingBox.y, maxBoundingBox.z, 1.0);
     clipVertices[7] = modelViewProjectionMatrix * vec4(minBoundingBox.x, maxBoundingBox.y, maxBoundingBox.z, 1.0);

      // Check if all vertices are outside of any clip space plane
     bool outsideX = true, outsideY = true, outsideZ = true;
     for (int i = 0; i < 8; i++){
        if(clipVertices[i].x > -clipVertices[i].w && clipVertices[i].x < clipVertices[i].w){
            outsideX = false;
        }
        if(clipVertices[i].y > -clipVertices[i].w && clipVertices[i].y < clipVertices[i].w){
            outsideY = false;
        }
          if(clipVertices[i].z > -clipVertices[i].w && clipVertices[i].z < clipVertices[i].w){
            outsideZ = false;
        }
     }


    if(outsideX || outsideY || outsideZ) {
         visible = 0; // outside
    } else {
        visible = 1;  // visible, or partially visible
        outObjectID = inObjectID[0];
        gl_Position = modelViewProjectionMatrix * vec4(inPosition[0], 1.0);
        EmitVertex();
         gl_Position = modelViewProjectionMatrix * vec4(inPosition[0], 1.0);
         EmitVertex();
         gl_Position = modelViewProjectionMatrix * vec4(inPosition[0], 1.0);
         EmitVertex();
        gl_Position = modelViewProjectionMatrix * vec4(inPosition[0], 1.0);
         EmitVertex();
        EndPrimitive();
    }


}
```

In this example, the geometry shader evaluates visibility after receiving the bounding box data and the world position of a vertex (which is not needed for culling). The geometry shader then outputs vertices based on the visibility for the subsequent fragment shader to execute on these visible primitives.

**Example 3: Index Buffer Generation (conceptual code)**

Generating an index buffer that contains only the IDs of the visible objects is a very powerful technique that avoids drawing any invisible object during the actual rendering stage. While this step does not live within any of the shader stages, conceptually, the previous shaders (vertex or geometry) output a `visible` flat variable. When `visible` is set to `1`, it indicates that this object should be rendered. A follow-up process, for example, a compute shader, uses an atomic operation to accumulate all the `objectID` into a buffer, that will then be used to draw only the visible objects. This logic can also be implemented as part of the geometry shader outputting each `outObjectID` with an extra vertex, that will be later interpreted by the host program using the stream-output feature in OpenGL.

```glsl
// Conceptual fragment shader or stream output stage logic
// ... from the previous vertex or geometry shader
if (visible == 1)
{
  // This would be done on the CPU or by an atomic operation on the GPU.
  add_to_index_buffer(outObjectID);
}

```
This conceptual code represents the process of building the index buffer. The `add_to_index_buffer` function would append the object ID to a buffer that’s used in the rendering phase to only draw visible objects. The actual implementation is API dependent.

These techniques can be further enhanced using hierarchical culling, and by performing culling on multiple levels of detail, to maximize the savings in geometry processing. Moreover, more efficient culling methods exist, that can be tailored to specific game scenarios, and that involve testing against the frustum planes, or other advanced techniques to detect overlap. Finally, note that it’s important to measure performance in the real-world as certain optimizations might not provide significant performance boosts if not done carefully.

For further reading and resources, I would recommend the following texts on real-time computer graphics: *Real-Time Rendering* by Moller, Haines, and Hoffman; and *GPU Pro* books series which cover different GPU related topics. The official documentation of graphics APIs, like OpenGL and DirectX, also provides valuable information on the graphics pipeline and optimization strategies. Also, the online documentation by major engine vendors often have relevant information related to this topic.
