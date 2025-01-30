---
title: "How can a quadtree be implemented using arrays?"
date: "2025-01-30"
id: "how-can-a-quadtree-be-implemented-using-arrays"
---
Implementing a quadtree with arrays necessitates a departure from the typical recursive, node-based structure often associated with tree data structures. My experience, derived from developing spatial indexing solutions for geographic information systems, has shown that a linear, array-based representation can be significantly more memory-efficient, particularly when dealing with large, relatively sparse datasets. This approach leverages implicit hierarchical relationships derived from array indices, effectively simulating the traditional quadtree’s branching structure without explicit node objects.

The core concept revolves around a single, potentially large, one-dimensional array. Each element of this array conceptually corresponds to a region within the spatial domain being represented by the quadtree. We achieve this by imposing a specific ordering on these regions, such that a parent region's data can be located based on simple index calculations from its four child regions. A critical aspect of this ordering is the ability to rapidly derive a region's level and parent index, which are key to querying and data insertion. This eliminates the overhead of managing pointers and allocation, though at the cost of less flexibility than linked node structures.

To understand the process, consider a complete quadtree: a quadtree where all levels are fully populated. A level-order traversal would provide the basis for this array structure. The root node occupies index 0. Its four children then occupy indices 1 through 4. These are followed by the children of node 1 (indices 5-8), then the children of node 2 (indices 9-12), and so on, processing the tree level by level. Importantly, not all array slots are necessarily occupied by user data; many may be null or a designated "empty" value, especially when dealing with sparse datasets. This makes this scheme most effective when the data has clusters of high density.

The spatial region associated with an array index is defined implicitly by the total spatial domain, the size of the array, and the index’s position. Imagine dividing a square spatial domain in half along both its horizontal and vertical axes. This creates four equal quadrants, with each quadrant representing a child of the root. If we subdivide each quadrant in the same fashion, the next level of the tree is created. The implicit association between an array index and its spatial region hinges on the number of subdivisions that have occurred to arrive at that point, the array's size, and where in the traversal the current element is located.

Let's illustrate this with code examples in a language similar to Java, using integer arrays. The spatial domain is assumed to be a square region whose side length is a power of 2. The values of the array indicate a presence (e.g., 1) or absence (e.g., 0) of an object within the associated region.

**Code Example 1: Calculating the parent index**

```java
public class QuadtreeArray {

    private int[] data;
    private int levelSize;
    private int maxLevel;
    private int domainSize;

    public QuadtreeArray(int maxLevel, int domainSize) {
        this.maxLevel = maxLevel;
        this.domainSize = domainSize;
        this.levelSize = (int)Math.pow(4,maxLevel+1) - 1;
        this.data = new int[levelSize];
    }

    public int getParentIndex(int index) {
        if (index == 0) return -1; // Root has no parent
        return (index - 1) / 4;
    }

    // ... other methods
}
```

This function `getParentIndex` takes the index of a node and returns the index of its parent within the array. The `(index-1) / 4` equation leverages the level order representation. For instance, nodes 1-4 are the children of index 0, therefore (1-1)/4 to (4-1)/4 evaluates to 0. Note the handling of the root node's parent, returning -1 to denote a lack of parent. The `QuadtreeArray` constructor calculates the total size needed for an array to store an entire complete quadtree. The constructor takes the `maxLevel` and the `domainSize` and will populate a large enough array to store the quadtree. The `levelSize` will store the total size of the array required for a full quadtree.

**Code Example 2: Retrieving a node's level**

```java
 public int getLevel(int index) {
        if(index < 0) return -1;
        int level = 0;
        int nodeCount = 1;
        while(index >= nodeCount){
          index -= nodeCount;
          nodeCount *= 4;
          level++;
        }
        return level;
    }
```

The `getLevel` function determines the level of a node given its array index. This function leverages the fact that each level's nodes start directly after the last node on the previous level. A node counter keeps track of the beginning of a new level.

**Code Example 3: Inserting a value at a specific location**

```java
 public void insert(int x, int y, int value) {

        int currentLevel = 0;
        int currentSize = this.domainSize;
        int currentIndex = 0; //Start at root.
        
        while(currentSize > 1 && currentLevel <= this.maxLevel){
            
            currentSize = currentSize / 2;
             int quadrant = 0;
             if(x >= currentSize){
                 x -= currentSize;
                 quadrant |= 0b01;
             }

            if(y >= currentSize){
                y -= currentSize;
                quadrant |= 0b10;
            }
           
            currentIndex = (currentIndex * 4) + quadrant + 1;
             
             currentLevel++;
        }
            
        data[currentIndex] = value;
    }
```

The `insert` function demonstrates how to map a coordinate to an array index, and set the appropriate data. The function recursively divides the domain and navigates through the tree's implicitly defined structure. The `currentSize` tracks the size of the domain. This is used to determine which quadrant the point will be part of. The quadrant is encoded into a binary value (0-3). The current index is advanced using the binary value. The loop stops when the desired level is achieved, and the value is written to the appropriate array position.

This array implementation does have limitations. It is not very memory efficient when a majority of the quadtree's nodes are empty. However, it is memory efficient when the data is clustered together. Also, unlike linked node implementations, modifying the depth of the quadtree is far more costly, often requiring a full reallocation of the underlying array. Its advantage is simplicity, performance (due to low overhead), and a high degree of predictability in memory usage.

In terms of resource recommendations, I'd suggest studying literature on space-filling curves, such as Hilbert and Morton curves. These techniques provide alternative linear orderings that can improve spatial locality, resulting in even more efficient implementations. Also, researching sparse matrix representations can yield alternative ideas about representing the data in memory, particularly in how to store quadtree cells that are empty. Furthermore, analyzing specific use cases with varying data densities is critical to selecting the most appropriate representation of quadtree data, so practical experimentation with various techniques is advised.

My experience has shown that there is no single "best" implementation. When the primary concerns are low overhead and straightforward spatial representation, an array-based quadtree can provide a solid foundation. Choosing this technique, like all engineering problems, comes down to trade-offs and a clear understanding of the problem being addressed.
