---
title: "How can an octree/BVH be constructed in GPU memory without using pointers?"
date: "2025-01-30"
id: "how-can-an-octreebvh-be-constructed-in-gpu"
---
The challenge of constructing an octree or Bounding Volume Hierarchy (BVH) on the GPU, specifically without resorting to pointer-based data structures, stems from the inherently parallel nature of GPU computation and the limitations of its memory model. GPUs excel at data-parallel operations, where the same instruction is executed across large datasets concurrently. Pointers, however, introduce dependencies that hinder such parallelism. The key is to utilize an implicit, array-based representation where spatial relationships are encoded using array indices instead of direct memory addresses. I've successfully employed this technique in a custom ray tracer I built for interactive scene rendering, facing similar constraints in maintaining performance on highly parallel graphics hardware.

The core idea revolves around representing the tree structure using flat arrays. Instead of each node containing a pointer to its children, we store the children’s data contiguously in other arrays. This eliminates the need for pointer dereferencing, thereby allowing for predictable memory access patterns more suitable for GPUs. Consider a simple octree. Each node has potentially eight children. Instead of storing eight child pointers, we can deduce a child's index within a dedicated array from its parent’s index.

To elaborate on this, we use several arrays in parallel. The first, which I'll call the `nodes` array, holds the data for each node. This often includes bounding box information (min and max coordinates) or a bounding sphere representation. The second, the `child_indices` array, stores the indices within the `nodes` array where a given node's children are located. Critically, the structure of the `child_indices` array implies the relative spatial locations of the children, usually ordered consistently: bottom-left-near, bottom-right-near, bottom-left-far, and so on. This ordering allows the implicit indexing. A third array, the `leaf_data` array, stores actual leaf data associated with the terminal nodes (e.g., triangle indices or object IDs).

Now, the construction process involves multiple parallel passes on the GPU. Initially, we can compute bounding boxes for our primitives (triangles, instances, etc.) using a parallel reduction operation. This gives us the data for the `leaf_data` array and a set of initial bounds. The first pass typically builds the root node and its children, calculating their bounds from the total scene bounds or a designated world space bounding volume. Subsequently, a workgroup can determine whether a child node satisfies a termination criterion – for example, a minimum number of primitives inside the node. If it does not, the workgroup further subdivides the node creating new entries in the `nodes` and `child_indices` arrays. The indices of these children are calculated from the parent’s index based on the predetermined ordering.

The key operation here is the efficient parallel index calculation. Since child nodes are laid out consecutively within the `nodes` array, the index of a child is a function of the parent's index and the child's position (e.g., first, second, third, etc.). For a full octree, this would often take the form of adding a constant offset based on the parent's index and the child’s index.

Below are three simplified code examples demonstrating different aspects of this approach using a C-like pseudocode, conceptually representing GPU kernel operations. Note, this uses a conceptual 'global' array representing data held in GPU memory.

**Code Example 1: Parallel Node Creation**

```c
// Global arrays (GPU memory)
global struct Node nodes[];
global int child_indices[];
global int next_node_index; // Atomic counter for unique index generation

struct Node {
  float min_x, min_y, min_z;
  float max_x, max_y, max_z;
  // Other node data
};

// GPU kernel (executing in parallel)
void create_child_nodes(int parent_index) {
  if (nodes[parent_index].is_leaf) return; // skip leafs

  // Determine child bounding box coordinates from parent
  for(int child_id = 0; child_id < 8; child_id++) {

    int current_child_index = atomic_add(&next_node_index, 1);
    // Example: calculate child min/max coordinates based on child_id and parent bounds
    nodes[current_child_index].min_x = ...; // calculate from parent node
    nodes[current_child_index].max_x = ...;

    // Similar assignments for y and z
    // Store the node index to the child_indices array. Children are stored sequentially from the base index
    child_indices[parent_index * 8 + child_id] = current_child_index;
  }
}
```

This first example demonstrates the parallel construction of child nodes. The `atomic_add` function ensures each thread gets a unique index into the `nodes` array while preventing race conditions when modifying the counter. The inner loop computes the bounding box for each child and stores its index within the `child_indices` array using an implicit offset based on `parent_index` and `child_id`.

**Code Example 2: Parallel Termination Check and Subdivision**

```c
// GPU kernel (executing in parallel)
global int num_primitives_in_node[];
global int num_primitives; // The total number of primitives

void check_and_subdivide_node(int node_index) {

    if(num_primitives_in_node[node_index] < MIN_PRIMITIVES_PER_LEAF){
        nodes[node_index].is_leaf = true; // Node becomes a leaf
        return;
    }

    create_child_nodes(node_index); //Subdivide if it has too many primitives

    // Example logic to distribute primitives to the children.
    for(int primitive_id=0; primitive_id < num_primitives; primitive_id++){
          int child_node_index = determine_child_node_from_primitive(primitive_id, nodes[node_index]);
          if(child_node_index != -1){ // -1 implies a node doesn't contain the primitive
               num_primitives_in_node[child_node_index]++;
          }
    }

}

int determine_child_node_from_primitive(int primitive_index, struct Node parent_node){
  // Perform spatial check for the primitive against each child to see in which child it should be contained
  // returns the child_node_index if the primitive lies inside the child. otherwise returns -1.
}
```

Here, each thread works on a specific node. It first checks the number of primitives in the node. If less than a defined threshold, it becomes a leaf. Otherwise, the `create_child_nodes` kernel is called and the primitives are distributed into child nodes based on their spatial location relative to the children's bounding boxes. The `num_primitives_in_node` array tracks the distribution.

**Code Example 3: Traversal for Ray Intersection**

```c
// Global arrays (GPU memory)
global struct Node nodes[];
global int child_indices[];
global struct LeafData leaf_data[];
global int leaf_indices[]; // index of leaf data

// GPU kernel (executing in parallel)
void ray_intersect(float ray_origin[3], float ray_direction[3], int ray_id) {
    int current_node_index = 0; // start at root node

    while(true){
       if(nodes[current_node_index].is_leaf){
           // intersect ray with primitives contained within the leaf
            int num_leaf_primitives = num_primitives_in_node[current_node_index];
           for(int i=0; i<num_leaf_primitives; i++){
                int primitive_index = leaf_indices[current_node_index * MAX_PRIMITIVES_PER_NODE + i];
                // intersection test with primitive_index against ray_origin and ray_direction
           }
          break;
       }
        float closest_intersection_distance = INFINITY;
        int next_node_index = -1;
        for(int child_id = 0; child_id < 8; child_id++){
            int child_node_index = child_indices[current_node_index * 8 + child_id];
            if(intersects_bounding_box(ray_origin, ray_direction, nodes[child_node_index])){
                 float distance = compute_ray_bounding_box_intersection_distance(ray_origin, ray_direction, nodes[child_node_index]);
                 if(distance < closest_intersection_distance){
                    closest_intersection_distance = distance;
                    next_node_index = child_node_index;
                 }
            }
        }
         if(next_node_index == -1){
            break;
         }
        current_node_index = next_node_index;
    }
}
```

This final example shows how to traverse the octree. Starting at the root (index 0), the kernel checks if a node is a leaf. If so, ray intersections with primitives are performed within that leaf. If not, the code iterates through all the children of the current node. If any of those children's bounding boxes are intersected, the ray proceeds to the closest intersected node. This is a depth-first traversal using an implicit representation of the octree structure based on array indices.

Several resources offer a thorough understanding of GPU-based spatial data structures. Literature on parallel algorithms, specifically those targeting GPUs, will provide a theoretical underpinning. Publications focusing on ray tracing or computational geometry often include practical implementations of these techniques. Furthermore, graphics programming textbooks often dedicate significant sections to the practical considerations of BVHs and octrees when used with parallel computation. Finally, public repositories containing GPU code (typically CUDA or OpenCL based) may offer useful insights. These can provide valuable insight into the practical implementation details. These resources, when used in conjunction, will provide comprehensive understanding of building spatial acceleration structures on the GPU.

This approach, devoid of traditional pointers, is crucial for leveraging the highly parallel processing power of GPUs when constructing and using hierarchical spatial data structures.
