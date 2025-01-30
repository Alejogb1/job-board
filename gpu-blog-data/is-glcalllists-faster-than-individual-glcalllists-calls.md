---
title: "Is glCallLists faster than individual glCallLists calls?"
date: "2025-01-30"
id: "is-glcalllists-faster-than-individual-glcalllists-calls"
---
The performance difference between a single `glCallLists` with multiple display lists and individual `glCallList` calls hinges critically on the underlying OpenGL implementation and hardware. My experience profiling 3D rendering engines over the past decade, particularly across disparate mobile GPUs and embedded systems, consistently highlights the overhead associated with frequent state changes and API calls. This overhead can often dwarf the actual rendering time.

The fundamental operation of `glCallLists` is to execute a sequence of pre-compiled OpenGL commands stored within display lists. Display lists are compiled once and can be reused multiple times, saving the CPU from re-sending geometry, transformations, and other rendering commands to the GPU. `glCallList`, on the other hand, calls a single display list. The question of whether one `glCallLists` with a sequence of display lists is faster than multiple calls to `glCallList`, each with its individual list, requires careful examination of the driver-level optimization and data transmission.

When using `glCallList` repeatedly, each call involves the following sequence: (1) transfer the display list ID, (2) potentially switch state (if the display list involves changes to texture, color, matrix, etc.), and (3) command the GPU to execute the precompiled sequence. The process can incur significant overhead, especially when many lists are involved. Each call to `glCallList` can act as a boundary where the driver must potentially flush the graphics pipeline and make resource management decisions. These frequent flushes can interrupt data flow and reduce overall performance.

In contrast, `glCallLists` takes an array (or a pointer) of display list IDs. The driver can analyze this sequence of list IDs and, where possible, consolidate the execution and state transitions. For instance, if consecutive display lists use the same texture, the driver may only need to set the texture once instead of once per `glCallList` invocation. Similarly, if transformations are identical or related, the driver may perform some of the calculations beforehand or optimize memory access. This potential for optimization at the driver level gives `glCallLists` an inherent advantage, and it may group more work into a single command batch on the GPU.

However, the benefit of `glCallLists` is not universal. If each display list in your sequence involves significant state changes, the driver may be unable to optimize the batching as effectively. The internal driver implementation plays a crucial role; poorly written drivers or limitations of the hardware might negate any potential gains. Also, very long `glCallLists` sequences can consume excessive memory, leading to cache misses and performance bottlenecks.

Below are three code examples illustrating the practical differences and showcasing how performance can be affected depending on list composition and usage. These examples are based on my observations while building several cross-platform rendering systems for industrial simulation. The first example demonstrates an ideal case where `glCallLists` shows a performance advantage. The second example shows how unnecessary state changes can neutralize these gains. The final example focuses on data organization and efficient memory usage using `glCallLists`.

**Example 1: Efficient Batching with Common State**

```c++
// Assume displayLists is an array of display list IDs
void drawUsingCallLists(GLuint* displayLists, int numLists) {
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, myTexture); // Single texture binding
  glCallLists(numLists, GL_UNSIGNED_INT, displayLists); // Batch render all
  glDisable(GL_TEXTURE_2D);
}

void drawUsingCallList(GLuint* displayLists, int numLists) {
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, myTexture); // Single texture binding
  for (int i = 0; i < numLists; ++i) {
    glCallList(displayLists[i]); // Individual render per list
  }
  glDisable(GL_TEXTURE_2D);
}
```

In this example, multiple display lists, each probably rendering a relatively small object with the same texture, are displayed. Using a single `glBindTexture` before the `glCallLists` block shows how grouping display list calls with common states allows for the driver to avoid redundant calls, resulting in potentially faster rendering compared to the individual `glCallList` version.

**Example 2: Neutralizing Optimization with State Changes**

```c++
// Assume displayLists is an array of display list IDs
void drawUsingCallListsWithStateChanges(GLuint* displayLists, int numLists) {
    for (int i = 0; i < numLists; ++i) {
      if (i % 2 == 0) {
        glBindTexture(GL_TEXTURE_2D, textureA);
      } else {
        glBindTexture(GL_TEXTURE_2D, textureB);
      }
      glCallList(displayLists[i]);
    }
}

void drawUsingCallListWithStateChanges(GLuint* displayLists, int numLists) {
  for (int i = 0; i < numLists; ++i) {
      if (i % 2 == 0) {
        glBindTexture(GL_TEXTURE_2D, textureA);
      } else {
        glBindTexture(GL_TEXTURE_2D, textureB);
      }
        glCallList(displayLists[i]);
    }
}
```

Here, we intentionally introduce state changes (texture changes) within the render loop. In this contrived example, switching between two textures within the loops, regardless of whether they are called within `glCallLists` or with individual calls, severely hinders any performance gains. The driver has to account for the state change for each list. In this particular instance, the performance is unlikely to show a significant benefit when utilizing `glCallLists`. It highlights that even the most basic state change, like a texture bind, can nullify batching advantages when improperly used.

**Example 3: Memory Organization for Improved Batching**

```c++
//Assume geometry is grouped by material and display lists correspond to one material each.
void drawBatchedCallLists(std::vector<std::vector<GLuint>> displayListsByMaterial){
  for(const auto& lists: displayListsByMaterial){
      glBindTexture(GL_TEXTURE_2D, getTextureForMaterial(lists[0]));//assumes first list defines material for this sequence
      glCallLists(lists.size(), GL_UNSIGNED_INT, lists.data());
  }
}

//Assume geometry is grouped by object
void drawIndividualCallList(std::vector<GLuint> displayLists){
  for(const auto& list : displayLists){
    glBindTexture(GL_TEXTURE_2D, getTextureForList(list));
    glCallList(list);
  }
}
```

This example focuses on data organization. The `drawBatchedCallLists` function groups geometry (represented by display lists) by material before calling `glCallLists`, reducing texture switches. `drawIndividualCallList` assumes a simpler organization with object lists individually, forcing more texture changes. It illustrates that a better organization in the application's data structure can have substantial impact when used with batched lists.

Recommendations for further study include the following resources: the OpenGL Programming Guide, a detailed text for understanding the overall API; the OpenGL Superbible for code examples; and vendor documentation specific to the hardware you’re targeting (e.g., ARM, NVIDIA, or AMD) as these documents will cover specific driver implementations. Also profiling tools like NVIDIA Nsight or AMD RenderDoc can prove invaluable for performance analysis when trying to pinpoint the exact bottlenecks for a specific project. Always use hardware specific debugging tools.

In summary, `glCallLists` *can* be faster than individual `glCallList` calls but requires careful planning. It leverages driver-level optimization to reduce overhead by batching commands and minimizing state changes. However, the specific performance gain depends heavily on the nature of the display lists, their state requirements, the underlying driver implementation, and proper data structure organization. If the display list sequence requires frequent state changes, the benefits might be minimal, and an incorrect data structure can lead to worse performance than multiple calls to `glCallList`. For optimal results, always measure the execution time with appropriate tools and profile the application on the intended target hardware.
