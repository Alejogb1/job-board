---
title: "How do I send an object to the back of the display list?"
date: "2025-01-30"
id: "how-do-i-send-an-object-to-the"
---
In many 2D rendering engines, the order in which objects are drawn to the screen directly influences their visual layering; objects drawn later appear “on top” of those drawn earlier. Manipulating this draw order to send an object to the back of the display list involves managing the list’s structure and typically requires direct access to the underlying rendering pipeline. Over my years developing a custom 2D game engine in C++ with OpenGL, I've frequently encountered this task. Therefore, I will address how an object can be placed at the rear of the display stack, often represented as an array, vector or similar data structure.

The foundational principle is understanding that the display list isn't just a collection of objects; it's an ordered sequence used by the rendering loop to iterate through and draw each item. To move an object to the back, one essentially needs to remove it from its current position and reinsert it at the beginning of the sequence, resulting in it being drawn first. This is often referred to as "reordering the display list."

This seemingly simple operation requires care. A poorly managed reordering could result in a noticeable performance drop, especially in complex scenes with many objects. It's critical to optimize the process to avoid unnecessary iterations or memory allocations. Consider using a data structure that allows for efficient insertion at the beginning or efficient movement of items to the first position.

To illustrate, consider the following scenarios with specific code examples, assuming we have a basic display list represented as a `std::vector` of renderable objects. For the sake of simplicity, let's assume each object is represented by a simple struct:

```cpp
struct RenderableObject {
    int id;
    // ... other rendering data ...
};
```

**Example 1: Direct Reordering Using `std::vector::erase` and `std::vector::insert`**

This method directly manipulates the `std::vector` by removing an object by ID and inserting it at the beginning. This is straightforward but can become inefficient with large lists because insertion at the beginning forces subsequent elements to be shifted.

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

void sendObjectToBack(std::vector<RenderableObject>& displayList, int objectId) {
    auto it = std::find_if(displayList.begin(), displayList.end(),
                         [objectId](const RenderableObject& obj){ return obj.id == objectId; });

    if (it != displayList.end()) {
        RenderableObject objectToMove = *it; // Copy the object to insert
        displayList.erase(it); // Remove from original position.
        displayList.insert(displayList.begin(), objectToMove); // insert at the beginning
    }
    else {
        std::cerr << "Object with ID " << objectId << " not found." << std::endl;
    }
}

int main() {
  std::vector<RenderableObject> displayList = {{1}, {2}, {3}, {4}, {5}};
  
  std::cout << "Original order: ";
  for(const auto& obj : displayList) {
     std::cout << obj.id << " ";
  }
  std::cout << std::endl;

  sendObjectToBack(displayList, 3);

   std::cout << "New order: ";
    for(const auto& obj : displayList) {
     std::cout << obj.id << " ";
  }
    std::cout << std::endl;


    return 0;
}
```

**Commentary:** The code uses `std::find_if` to locate the target object by its ID. If found, it copies the object's data, removes it using `erase` (invalidating iterators beyond the erased element), and then inserts the copy at the very beginning via `insert`. The cost of insert at the beginning could be a performance issue for large display lists, but it is easy to understand and implement, making it reasonable for smaller projects. The `main` function demonstrates basic usage and output for verification.

**Example 2: Optimized Reordering Using `std::remove` and `std::vector::insert` with move semantics.**

This example avoids unnecessary copying by employing `std::remove` and move semantics. `std::remove` does not actually remove elements, but moves all elements other than the removal target toward the beginning of the vector. We then use `std::move` in the insert to avoid copying the removed object.

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

void sendObjectToBackOptimized(std::vector<RenderableObject>& displayList, int objectId) {
    auto it = std::remove_if(displayList.begin(), displayList.end(),
                        [objectId](const RenderableObject& obj){ return obj.id == objectId; });

    if (it != displayList.end()) {
       RenderableObject objectToMove = std::move(*it); // Move the object
       displayList.erase(it);
       displayList.insert(displayList.begin(), std::move(objectToMove)); // Move insert at the beginning
    }
      else {
        std::cerr << "Object with ID " << objectId << " not found." << std::endl;
    }
}

int main() {
  std::vector<RenderableObject> displayList = {{1}, {2}, {3}, {4}, {5}};
  
  std::cout << "Original order: ";
  for(const auto& obj : displayList) {
     std::cout << obj.id << " ";
  }
  std::cout << std::endl;

  sendObjectToBackOptimized(displayList, 3);

  std::cout << "New order: ";
    for(const auto& obj : displayList) {
     std::cout << obj.id << " ";
  }
    std::cout << std::endl;

    return 0;
}
```

**Commentary:** Here `std::remove_if` is used to 'move' the object to the back of the 'valid' portion of the vector. We then use the iterator that `remove_if` returned to create a copy via move semantics and remove the original position of the object from the vector, followed by insertion at the beginning using move semantics. This avoids the overhead of a copy if the `RenderableObject` is complex or has many members. For large lists, this should perform better than the first method. The `main` function's output confirms the reordering.

**Example 3: Using a List and Move Semantics**

This approach uses `std::list`, which provides efficient insertion at the beginning. We still use `remove_if` which, for lists, is much more performant.

```cpp
#include <list>
#include <algorithm>
#include <iostream>

void sendObjectToBackList(std::list<RenderableObject>& displayList, int objectId) {
   auto it = std::remove_if(displayList.begin(), displayList.end(),
                        [objectId](const RenderableObject& obj){ return obj.id == objectId; });

    if (it != displayList.end()) {
        RenderableObject objectToMove = std::move(*it); // Move the object
        displayList.erase(it);
        displayList.push_front(std::move(objectToMove));
    }
        else {
        std::cerr << "Object with ID " << objectId << " not found." << std::endl;
    }
}

int main() {
    std::list<RenderableObject> displayList = {{1}, {2}, {3}, {4}, {5}};

      std::cout << "Original order: ";
    for(const auto& obj : displayList) {
     std::cout << obj.id << " ";
  }
    std::cout << std::endl;

  sendObjectToBackList(displayList, 3);

  std::cout << "New order: ";
    for(const auto& obj : displayList) {
     std::cout << obj.id << " ";
  }
    std::cout << std::endl;

    return 0;
}
```

**Commentary:** Here, `std::list` is used instead of `std::vector`. The logic is similar, but the `push_front` operation has better performance than insert at the beginning of a vector. This is due to the nature of `std::list` which uses nodes in memory and thus is able to quickly move objects to the beginning without the costly memory movements of a vector. Again move semantics are used to avoid copying objects. The `main` function again provides a basic use and verification.

Choosing between the approaches discussed largely depends on the number of objects in your display list and how often you need to move objects. For smaller lists, the first method is sufficient. For larger lists or frequent reordering, using a `std::list` or the optimized `std::remove` approach is recommended for higher performance. Additionally, the cost of object copying should be carefully considered, making move semantics desirable.

I'd strongly advise consulting resources that deeply cover data structures and algorithms. Publications on effective modern C++ should also be examined to gain a thorough understanding of memory management and efficiency techniques. Additionally, examining the documentation for the specific rendering API being used will reveal details on how the display list operates internally and any available methods for reordering. The principles remain the same regardless, the implementation often requires specialized knowledge of the underlying rendering system.
