---
title: "Wavefront OBJ Files: Why Are Faces Still Stuck with 4 Vertices? 🤔"
date: '2024-11-08'
id: 'wavefront-obj-files-why-are-faces-still-stuck-with-4-vertices'
---

```c++
#include <vector>

// ... your OpenGL initialization ...

// Assuming you have a vector of vertices and a vector of faces
// where each face is a vector of vertex indices

std::vector<glm::vec3> vertices;
std::vector<std::vector<int>> faces;

// ... load your OBJ file and populate vertices and faces ...

// Create a vector to store the triangles
std::vector<glm::vec3> triangles;

// Iterate through each face
for (const auto& face : faces) {
  // If the face has more than 3 vertices, triangulate it
  if (face.size() > 3) {
    for (size_t i = 1; i < face.size() - 1; ++i) {
      triangles.push_back(vertices[face[0]]);
      triangles.push_back(vertices[face[i]]);
      triangles.push_back(vertices[face[i + 1]]);
    }
  } else {
    // If the face has 3 vertices, simply add it to the triangles vector
    triangles.push_back(vertices[face[0]]);
    triangles.push_back(vertices[face[1]]);
    triangles.push_back(vertices[face[2]]);
  }
}

// Bind your vertex array object and vertex buffer object
// ...

// Upload the triangles data to the vertex buffer object
// ...

// Draw the triangles using GL_TRIANGLES
glDrawArrays(GL_TRIANGLES, 0, triangles.size());
```
