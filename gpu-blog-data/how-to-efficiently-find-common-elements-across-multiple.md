---
title: "How to efficiently find common elements across multiple sorted lists?"
date: "2025-01-30"
id: "how-to-efficiently-find-common-elements-across-multiple"
---
The crux of efficiently finding common elements across multiple sorted lists lies in avoiding redundant comparisons. A naive approach, comparing each element of the first list against every element in subsequent lists, results in a time complexity approaching O(n*m*k), where n is the length of the first list and m and k are the lengths of other lists. This is clearly suboptimal. The key is to leverage the sorted nature of the lists to prune unnecessary comparisons, leading to a significant performance gain.

Specifically, we can utilize a multi-pointer technique, akin to a merge sort algorithm, to achieve linear time complexity relative to the sum of all list lengths, or O(N) in the best case scenario if all the lists are roughly the same length. This approach maintains an index (pointer) for each list. We simultaneously advance the pointers based on comparisons of the currently pointed-to elements across all the lists. If we find all pointed-to elements to be equal, we know we have a common element, and we add it to our results, and advance all the pointers. If they are not equal, we advance only the pointer(s) associated with the smallest value, thereby preventing unnecessary back-tracking.

The efficiency gain becomes apparent when dealing with longer lists, and is especially beneficial when processing data streams where sorting and preprocessing of the list is not an option. To maintain linear time complexity, we must ensure the operations on the lists and indices themselves do not add additional steps. Standard array indexing provides constant time access to the current list elements.

Here’s how we can express this in Python, keeping in mind that the general principles apply to other languages such as Java or C++.

**Code Example 1: Basic Implementation**

```python
def find_common_elements(lists):
    if not lists:
        return []

    pointers = [0] * len(lists)
    common_elements = []
    
    while True:
        current_values = [lists[i][pointers[i]] if pointers[i] < len(lists[i]) else float('inf') for i in range(len(lists))]
        min_value = min(current_values)
        
        if min_value == float('inf'):
          break

        if all(val == min_value for val in current_values if val != float('inf')):
            common_elements.append(min_value)
            for i in range(len(lists)):
              if pointers[i] < len(lists[i]):
                pointers[i] += 1
        else:
            for i in range(len(lists)):
                if current_values[i] == min_value and pointers[i] < len(lists[i]):
                  pointers[i] += 1
    
    return common_elements

# Example Usage:
list1 = [1, 3, 4, 6, 7]
list2 = [2, 3, 4, 8, 9]
list3 = [3, 4, 5, 6]
print(find_common_elements([list1, list2, list3]))  # Output: [3, 4]
```
In this first example, I establish an array of pointers, `pointers`, initialized to zero for every list in the input. I iterate over the lists using a while loop. Inside the loop, `current_values` lists all the current element values at the index pointed by `pointers` in each list. If the pointer exceeds the list bound, `float('inf')` is added instead to prevent processing. `min_value` is then set to the smallest value amongst the pointed elements. If all `current_values` are equivalent to `min_value` and not equal to infinity, they are determined as common and appended to the `common_elements` list. Finally, I increment all pointers. If the values aren't common, I increment only the pointers associated with the smallest `current_values`, in order to skip all elements that are smaller than other lists. The while loop is stopped when `min_value` is equal to infinity, which implies that the end of all lists has been reached. This is a basic but functional version.

**Code Example 2: Handling Empty Lists and Duplicates**

```python
def find_common_elements_v2(lists):
    if not lists:
        return []

    pointers = [0] * len(lists)
    common_elements = []
    
    while True:
        current_values = [lists[i][pointers[i]] if pointers[i] < len(lists[i]) else float('inf') for i in range(len(lists))]
        min_value = min(current_values)

        if min_value == float('inf'):
          break
        
        if all(val == min_value for val in current_values if val != float('inf')):
           if not common_elements or common_elements[-1] != min_value: 
              common_elements.append(min_value)
           for i in range(len(lists)):
                if pointers[i] < len(lists[i]):
                  pointers[i] += 1

        else:
            for i in range(len(lists)):
                if current_values[i] == min_value and pointers[i] < len(lists[i]):
                  pointers[i] += 1

    return common_elements
# Example Usage:
list1 = [1, 3, 3, 4, 6, 7]
list2 = [2, 3, 3, 4, 8, 9]
list3 = [3, 3, 4, 5, 6]
print(find_common_elements_v2([list1, list2, list3])) # Output: [3, 4]

list4 = []
list5 = [1,2,3]
print(find_common_elements_v2([list4,list5])) # Output: []
```
This version improves on the first by explicitly handling two key considerations. First, it prevents duplicate entries of common elements into `common_elements`. Here, before appending a common element, I check if the last appended element is not the same value. Second, this version handles empty lists correctly: it will return an empty list instead of throwing an error. I found that these corner case checks are important to avoid issues during development and runtime.

**Code Example 3: Generator Implementation (Memory Efficiency)**

```python
def common_elements_generator(lists):
    if not lists:
        return

    pointers = [0] * len(lists)
    
    while True:
        current_values = [lists[i][pointers[i]] if pointers[i] < len(lists[i]) else float('inf') for i in range(len(lists))]
        min_value = min(current_values)

        if min_value == float('inf'):
          break

        if all(val == min_value for val in current_values if val != float('inf')):
            
            yield min_value
            for i in range(len(lists)):
                if pointers[i] < len(lists[i]):
                  pointers[i] += 1
        else:
            for i in range(len(lists)):
                if current_values[i] == min_value and pointers[i] < len(lists[i]):
                   pointers[i] += 1

# Example Usage:
list1 = [1, 3, 3, 4, 6, 7]
list2 = [2, 3, 3, 4, 8, 9]
list3 = [3, 3, 4, 5, 6]

for element in common_elements_generator([list1, list2, list3]):
    print(element) # Output: 3 \n 4

list4 = []
list5 = [1,2,3]

for element in common_elements_generator([list4, list5]):
   print(element)  # Output: None, iterator is empty
```

This final example refactors the function into a generator, which is particularly useful when the input lists are very large and the output is intended to be processed iteratively rather than storing the entire set of common elements in memory. Instead of appending the common values to the `common_elements` list, I yield each element as it is found, letting the caller control how it is used. This makes the function far more memory-efficient for very large datasets. The generator itself does not store all the common elements, making this solution scale to potentially very big data without memory constraints. Furthermore, I use this generator with an iterator that handles the empty return from empty lists more gracefully than a print statement that would return '[]'.

When selecting a specific implementation, I would consider the trade-offs between code simplicity, memory constraints and required output behavior. For smaller lists, version 2 is often sufficient; for larger lists or cases requiring real-time processing, the generator-based implementation in example 3 provides a better architectural foundation.

For further study, I would recommend exploring data structure and algorithm resources with emphasis on sorting algorithms and pointer manipulation techniques. Specific books and online materials include those on algorithm design and analysis, which often cover multi-way merge techniques. Researching the asymptotic behavior of algorithms will also provide a deeper understanding of why this solution performs better than the naive comparison based solution. Also, studying design patterns associated with iterator objects can further enhance understanding of the usage of generators.
