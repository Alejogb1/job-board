---
title: "How do I retrieve values from a CUDA Thrust map given their keys?"
date: "2025-01-30"
id: "how-do-i-retrieve-values-from-a-cuda"
---
Retrieving values from a Thrust `map` by key, while seemingly straightforward, requires a specific approach due to the nature of GPU data structures and Thrust’s parallel algorithms. Thrust's `map` is not a direct equivalent to a CPU-based associative container offering efficient key-based lookups, such as C++'s `std::map`. Instead, a Thrust `map` is primarily intended for parallel data transformations and represents a sequence of key-value pairs held in device memory. Consequently, direct random access by key is not its strength. The primary consideration is therefore how to efficiently search or filter through this parallel key-value data to extract desired values.

The core problem stems from the fact that Thrust's `map` does not provide a built-in, optimized function to retrieve values based on a query key. Thrust is designed for parallel processing of data, making the data layout fundamental to efficiency. A `thrust::device_vector<std::pair<KeyType, ValueType>>` is generally used to represent the `map`, often without internal index structures that allow direct, fast key lookups. Linear scans or more specialized algorithms are necessary to locate the desired key and retrieve the associated value, making algorithmic selection crucial for optimized performance on the GPU. The method chosen depends on the scale of the map, number of queries and the underlying key-value data characteristics.

I have frequently encountered this challenge during my time developing high-performance numerical simulations for fluid dynamics. One particular project required me to store and rapidly access material properties associated with grid points on a computational mesh. The properties were calculated once then needed to be repeatedly accessed based on grid point indices (acting as keys), hence the need for efficient key-value retrieval. The simplest approach, suitable for smaller data sets, is to use Thrust's algorithms to search the entire `map` for a matching key. This technique, despite its linear complexity, is often the most readily implementable if the size of the map and number of queries remain modest.

The following code demonstrates this method:

```cpp
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/find_if.h>
#include <thrust/execution_policy.h>
#include <iostream>

// Helper function to create a map
thrust::device_vector<thrust::pair<int, float>> createMap() {
    thrust::device_vector<thrust::pair<int, float>> map(5);
    map[0] = thrust::make_pair(1, 10.0f);
    map[1] = thrust::make_pair(2, 20.0f);
    map[2] = thrust::make_pair(3, 30.0f);
    map[3] = thrust::make_pair(4, 40.0f);
    map[4] = thrust::make_pair(5, 50.0f);
    return map;
}

// Searches the map for a key and returns the associated value.
thrust::optional<float> lookupValue(const thrust::device_vector<thrust::pair<int, float>>& map, int key) {
    auto found_it = thrust::find_if(
        thrust::execution::par,
        map.begin(), map.end(),
        [key] (const thrust::pair<int, float>& pair) { return pair.first == key; }
    );
    if (found_it != map.end()) {
        return found_it->second;
    }
    return {}; // return an empty optional if not found
}


int main() {
    thrust::device_vector<thrust::pair<int, float>> myMap = createMap();

    int searchKey = 3;
    auto value = lookupValue(myMap, searchKey);

    if (value) {
        std::cout << "Value associated with key " << searchKey << " is: " << *value << std::endl;
    } else {
        std::cout << "Key " << searchKey << " not found in map" << std::endl;
    }

    searchKey = 6; // key not present in map
    value = lookupValue(myMap, searchKey);

    if (value) {
        std::cout << "Value associated with key " << searchKey << " is: " << *value << std::endl;
    } else {
        std::cout << "Key " << searchKey << " not found in map" << std::endl;
    }

    return 0;
}
```

In this example, `createMap` sets up a sample map in device memory. The `lookupValue` function uses `thrust::find_if` with a lambda predicate to perform a parallel linear search for the given key. The result is an iterator which, if not at the end, points to the key-value pair. The associated value is extracted and returned as an `std::optional<float>`. This approach is straightforward to implement and leverages Thrust's parallelism, but its performance degrades as the size of the map increases. For very large datasets, a single linear scan for each query will quickly become a bottleneck.

When dealing with large datasets and many key lookup operations, creating a structure specifically designed for efficient lookups is necessary. A suitable technique, which I often employ, is to transform the initial key-value pair vector into a sorted structure based on the keys. Subsequently, a binary search can be conducted to find the desired key. This method significantly improves search performance from O(N) to O(log N).

The following code illustrates this:

```cpp
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <iostream>

// Helper function to create and sort the map
thrust::device_vector<thrust::pair<int, float>> createAndSortMap() {
    thrust::device_vector<thrust::pair<int, float>> map(5);
    map[0] = thrust::make_pair(3, 30.0f); // out of order
    map[1] = thrust::make_pair(1, 10.0f);
    map[2] = thrust::make_pair(5, 50.0f);
    map[3] = thrust::make_pair(2, 20.0f);
    map[4] = thrust::make_pair(4, 40.0f);
    thrust::sort(
        thrust::execution::par,
        map.begin(), map.end(),
        [](const thrust::pair<int, float>& a, const thrust::pair<int, float>& b) {
        return a.first < b.first;
        }
    );

    return map;
}


// Looks up a value by key using binary search. Returns an empty optional if not found
thrust::optional<float> lookupValueBinarySearch(const thrust::device_vector<thrust::pair<int, float>>& map, int key) {
    auto found_it = thrust::lower_bound(
        thrust::execution::par,
        map.begin(), map.end(),
        thrust::make_pair(key, 0.0f),
        [](const thrust::pair<int, float>& a, const thrust::pair<int, float>& b) {
        return a.first < b.first;
        }
    );

    if(found_it != map.end() && found_it->first == key)
        return found_it->second;
    return {};
}

int main() {
    thrust::device_vector<thrust::pair<int, float>> sortedMap = createAndSortMap();

    int searchKey = 3;
    auto value = lookupValueBinarySearch(sortedMap, searchKey);

    if (value) {
        std::cout << "Value associated with key " << searchKey << " is: " << *value << std::endl;
    } else {
        std::cout << "Key " << searchKey << " not found in map" << std::endl;
    }

    searchKey = 6; // key not present in map
    value = lookupValueBinarySearch(sortedMap, searchKey);

    if (value) {
        std::cout << "Value associated with key " << searchKey << " is: " << *value << std::endl;
    } else {
        std::cout << "Key " << searchKey << " not found in map" << std::endl;
    }
    return 0;
}
```
Here, `createAndSortMap` creates the key-value pairs and then uses `thrust::sort` to order the pairs by key. `lookupValueBinarySearch` utilizes `thrust::lower_bound` to find the position of the key using binary search. The result is checked if the key was actually found at the returned position before extracting the value. While sorting adds an initial preprocessing step, the significant improvement in lookup performance is often worthwhile when numerous lookups are performed.

When the keys are known to have a limited range, a more efficient technique can be employed: a lookup table. If the keys are integers from 0 to N-1, or can be easily mapped to that range, a vector of values can be directly indexed using the key. This approach, having constant time complexity, provides the fastest possible lookups but adds significant memory requirements if the key range is substantially large and the map itself is sparsely populated.

Here's an example:
```cpp
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>


// Helper function to create a map with a known key range
thrust::device_vector<thrust::pair<int, float>> createMapWithRange() {
    thrust::device_vector<thrust::pair<int, float>> map(5);
    map[0] = thrust::make_pair(1, 10.0f);
    map[1] = thrust::make_pair(3, 30.0f);
    map[2] = thrust::make_pair(0, 5.0f);
    map[3] = thrust::make_pair(2, 20.0f);
    map[4] = thrust::make_pair(4, 40.0f);
    return map;
}

// Creates a lookup table from the key-value pairs. Returns a thrust::device_vector.
thrust::device_vector<thrust::optional<float>> createLookupTable(const thrust::device_vector<thrust::pair<int, float>>& map, int tableSize){
    thrust::device_vector<thrust::optional<float>> lookupTable(tableSize);
    for(auto const& pair: map){
      lookupTable[pair.first] = pair.second;
    }
    return lookupTable;
}


// Looks up a value by key in lookup table
thrust::optional<float> lookupValueLookupTable(const thrust::device_vector<thrust::optional<float>>& lookupTable, int key){
  if(key < lookupTable.size())
    return lookupTable[key];
  return {};
}


int main() {
  thrust::device_vector<thrust::pair<int, float>> myMap = createMapWithRange();
  const int tableSize = 5; //Size is equal to largest key + 1 since keys are from 0 to 4 in this example

  thrust::device_vector<thrust::optional<float>> lookupTable = createLookupTable(myMap, tableSize);


    int searchKey = 3;
    auto value = lookupValueLookupTable(lookupTable, searchKey);

    if (value) {
        std::cout << "Value associated with key " << searchKey << " is: " << *value << std::endl;
    } else {
        std::cout << "Key " << searchKey << " not found in map" << std::endl;
    }

    searchKey = 6; // key not present in the table's range
    value = lookupValueLookupTable(lookupTable, searchKey);

    if (value) {
        std::cout << "Value associated with key " << searchKey << " is: " << *value << std::endl;
    } else {
        std::cout << "Key " << searchKey << " not found in map" << std::endl;
    }

    return 0;
}
```
Here `createMapWithRange` creates a sample map which is subsequently transformed into `lookupTable`. `createLookupTable` allocates a `device_vector` sized by the maximal key + 1 and fills it using the provided key-value pairs. `lookupValueLookupTable` then performs the lookup by using the key as a direct index into the `device_vector`. This provides an O(1) lookup time, but requires extra memory for the full lookup table.

In conclusion, the optimal approach to retrieving values from a Thrust map based on keys depends greatly on factors including the map size, number of lookups, key ranges, and available memory. For small-scale maps or infrequent queries, a linear search can be sufficient.  When dealing with extensive maps and repeated lookups, sorting the data and employing binary search or a lookup table becomes necessary for optimal performance. The choice depends on the particular needs of the application and the trade-off between memory usage and processing time. For further understanding, reviewing resources on parallel data structures and algorithms specifically pertaining to CUDA and GPU programming is essential. Focus on materials covering binary search, hash maps, and other relevant search algorithms implemented in a parallel environment. Studying advanced CUDA coding techniques would also enhance understanding of memory management and optimal data layout for GPU. Furthermore, reading up on different use cases and examples from various scientific and engineering application areas will be beneficial in choosing the correct strategy for performance.
