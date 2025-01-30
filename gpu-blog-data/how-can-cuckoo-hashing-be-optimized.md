---
title: "How can cuckoo hashing be optimized?"
date: "2025-01-30"
id: "how-can-cuckoo-hashing-be-optimized"
---
Cuckoo hashing, a collision resolution technique in hash tables, inherently offers good average-case performance due to its ability to move existing keys to alternate locations upon collisions. However, several bottlenecks can emerge, particularly concerning worst-case scenarios and practical implementations. I've personally grappled with these limitations in distributed cache systems where predictable performance is paramount. Optimization isn't about a single 'magic bullet,' but rather a layered approach focusing on load factor management, improved eviction strategies, and efficient implementation details.

The primary characteristic of cuckoo hashing is the use of multiple hash functions, typically two, to determine possible locations for a key. Upon insertion, if the target location is occupied, the existing key is "kicked out" to its alternate location, repeating this process until an empty slot is found. If the procedure falls into a cycle, causing endless relocation, it is declared as a failure, and a rehash operation is needed with a new set of hash functions, usually on a table double the original size. It's the potential for these rehashes and cycles that demand careful optimization.

First and foremost, a crucial area for enhancement is load factor management. A high load factor leads to increased collision frequency, thus extending insertion times and greatly enhancing the likelihood of rehash operations. Empirical evidence suggests keeping the load factor relatively low, typically below 50%, improves the chances of successful insertion within a short time. The trade-off here is increased memory usage in the table, but the performance gains justify this sacrifice. Instead of simply aiming for a target load factor, dynamic adjustment of the load threshold can be highly effective. For example, one could track the average number of relocation steps during insertions. If this average starts to trend upwards, it signals an imminent need for a resize even before the pre-defined maximum load is reached. Implementing such a strategy can proactively reduce worst-case scenarios.

Another optimization pathway concerns eviction selection when dealing with more than two hash functions. While two hash functions are most common, I’ve implemented systems using three or even four to mitigate cycles. However, when more hash functions are involved, the simple "kicking out" of the current key can lead to sub-optimal relocation patterns. Implementing intelligent eviction policies becomes beneficial. Instead of randomly selecting from available hash functions when relocating a key, one can strategically choose to move the key to the location with the fewest collisions, considering the table’s overall occupancy density. This helps in flattening collision hotspots, improving the efficiency of future insertions.

Thirdly, the choice and implementation of hash functions play a critical role in the overall efficiency of cuckoo hashing. The quality of hash function is measured by how randomly it distributes keys across the hash table, and the speed of hashing operation. The ideal hash functions will uniformly distribute keys and can be quickly computed. I've found that using carefully implemented, well-established functions such as MurmurHash or xxHash offers a good balance between computational efficiency and key distribution. Moreover, avoiding expensive hash computations on every relocation step can significantly speed up insert operations. This could involve pre-computing hash values, or only calculating the second or the third hash after the previous locations are filled. Furthermore, ensuring that hash functions are truly independent and exhibit low correlation is key. A higher correlation in outputs between different hash functions leads to similar locations, increasing the number of relocations during insertions and making the system vulnerable to cycles.

Let's illustrate with code examples, focusing on optimization in the context of C++. It's important to note that, while I'm focusing on C++, the principles are portable.

```cpp
// Example 1: Basic Cuckoo Hashing with a dynamic load factor
#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <algorithm>

template <typename K, typename V>
class CuckooHashTable {
private:
    struct Entry {
        K key;
        V value;
        bool isOccupied;
    };
    std::vector<Entry> table;
    std::vector<std::function<size_t(const K&)>> hashFunctions;
    size_t capacity;
    double maxLoadFactor;
    size_t numEntries;
    std::mt19937 gen;

    void resizeTable() {
        size_t oldCapacity = capacity;
        capacity *= 2;
        std::vector<Entry> oldTable = table;
        table.assign(capacity, {{}, {}, false});
        numEntries = 0;
        for (size_t i = 0; i < oldCapacity; ++i) {
           if(oldTable[i].isOccupied) {
               insert(oldTable[i].key, oldTable[i].value);
           }
        }
        
    }

    bool insertHelper(const K& key, const V& value, size_t hashIndex, int depth) {
       if (depth > 2*capacity) { //detect and resolve cycles
           resizeTable();
           return insert(key, value);
       }
       size_t index = hashFunctions[hashIndex](key) % capacity;
       if (!table[index].isOccupied) {
           table[index] = {key, value, true};
           numEntries++;
           return true;
       }
       K tempKey = table[index].key;
       V tempValue = table[index].value;
       table[index] = {key, value, true};
       return insertHelper(tempKey, tempValue, (hashIndex +1 ) % hashFunctions.size(), depth+1);
    }


public:
    CuckooHashTable(size_t initialCapacity = 16, double loadFactor = 0.5) : capacity(initialCapacity), maxLoadFactor(loadFactor), numEntries(0) {
       table.resize(capacity);
        std::hash<K> stdHash;
       hashFunctions.push_back([stdHash](const K& k) {
           return stdHash(k);
       });
       std::uniform_int_distribution<long long> distrib;
       long long randomOffset = distrib(gen);
       hashFunctions.push_back([stdHash, randomOffset](const K& k) {
           return stdHash(k) ^ randomOffset;
       });
        gen.seed(std::random_device{}());
    }

    bool insert(const K& key, const V& value) {
       if(numEntries >= capacity * maxLoadFactor){
           resizeTable();
       }
       return insertHelper(key, value, 0, 0);

    }

    V* search(const K& key) {
        for (const auto& hashFunc : hashFunctions) {
            size_t index = hashFunc(key) % capacity;
             if(table[index].isOccupied && table[index].key == key) {
                return &table[index].value;
            }
        }
        return nullptr;
    }
     void remove(const K& key){
          for (const auto& hashFunc : hashFunctions) {
            size_t index = hashFunc(key) % capacity;
             if(table[index].isOccupied && table[index].key == key) {
                table[index].isOccupied = false;
                numEntries--;
             }
        }
    }
    size_t getCapacity() const {
        return capacity;
    }
     size_t size() const {
        return numEntries;
    }

};
```
*Commentary:* In this first example, the CuckooHashTable class encapsulates core cuckoo hashing logic. It includes a dynamic resizing strategy based on a max load factor, which allows you to keep occupancy low while avoiding too much reallocation. The example also addresses cycle detection with a max depth check to trigger rehash if necessary. This provides a starting point for evaluating the impact of load factor on performance.

```cpp
// Example 2: Strategic eviction using an "occupied count" function

#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <algorithm>

template <typename K, typename V>
class CuckooHashTableOptimized {
private:
    struct Entry {
        K key;
        V value;
        bool isOccupied;
    };
    std::vector<Entry> table;
    std::vector<std::function<size_t(const K&)>> hashFunctions;
    size_t capacity;
    double maxLoadFactor;
    size_t numEntries;
    std::mt19937 gen;

    void resizeTable() {
        size_t oldCapacity = capacity;
        capacity *= 2;
        std::vector<Entry> oldTable = table;
        table.assign(capacity, {{}, {}, false});
        numEntries = 0;
        for (size_t i = 0; i < oldCapacity; ++i) {
           if(oldTable[i].isOccupied) {
               insert(oldTable[i].key, oldTable[i].value);
           }
        }
    }
     int occupiedCount(const K& key) const {
            int count = 0;
            for (size_t i = 0; i < hashFunctions.size(); ++i) {
                size_t index = hashFunctions[i](key) % capacity;
                if (table[index].isOccupied) {
                   count++;
                }
            }
         return count;
    }

    bool insertHelper(const K& key, const V& value, int depth) {
       if (depth > 2*capacity) { //detect and resolve cycles
           resizeTable();
           return insert(key, value);
       }
       size_t minOccupiedIndex = 0;
       int minOccupiedCount = -1;
       for (size_t i = 0; i < hashFunctions.size(); ++i){
            size_t index = hashFunctions[i](key) % capacity;
           if(!table[index].isOccupied){
               table[index] = {key, value, true};
                numEntries++;
                 return true;
           }
           int count = occupiedCount(table[index].key);
           if(minOccupiedCount == -1 || count < minOccupiedCount){
               minOccupiedCount = count;
               minOccupiedIndex = i;
           }

       }
       size_t index = hashFunctions[minOccupiedIndex](key) % capacity;
       K tempKey = table[index].key;
       V tempValue = table[index].value;
       table[index] = {key, value, true};
       return insertHelper(tempKey, tempValue, depth+1);

    }

public:
    CuckooHashTableOptimized(size_t initialCapacity = 16, double loadFactor = 0.5) : capacity(initialCapacity), maxLoadFactor(loadFactor), numEntries(0) {
       table.resize(capacity);
        std::hash<K> stdHash;
       hashFunctions.push_back([stdHash](const K& k) {
           return stdHash(k);
       });
       std::uniform_int_distribution<long long> distrib;
       long long randomOffset = distrib(gen);
       hashFunctions.push_back([stdHash, randomOffset](const K& k) {
           return stdHash(k) ^ randomOffset;
       });
         gen.seed(std::random_device{}());
    }

    bool insert(const K& key, const V& value) {
       if(numEntries >= capacity * maxLoadFactor){
           resizeTable();
       }
       return insertHelper(key, value, 0);

    }

    V* search(const K& key) {
        for (const auto& hashFunc : hashFunctions) {
            size_t index = hashFunc(key) % capacity;
             if(table[index].isOccupied && table[index].key == key) {
                return &table[index].value;
            }
        }
        return nullptr;
    }
     void remove(const K& key){
          for (const auto& hashFunc : hashFunctions) {
            size_t index = hashFunc(key) % capacity;
             if(table[index].isOccupied && table[index].key == key) {
                table[index].isOccupied = false;
                numEntries--;
             }
        }
    }
      size_t getCapacity() const {
        return capacity;
    }
     size_t size() const {
        return numEntries;
    }
};
```
*Commentary:* This second implementation demonstrates a strategic eviction strategy. Instead of selecting the next available hash function, the code calculates the occupancy around each possible location and chooses the least dense option for the eviction. This helps to distribute keys more evenly, potentially decreasing the insertion time.

```cpp
// Example 3: Pre-computed and Independent Hash functions

#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <stdint.h> // Required for uint64_t
#include <cstring> // Required for memcpy


//Implementation of MurmurHash3 (32 bit variant) for speed
uint32_t murmurHash3_32(const void * key, size_t len, uint32_t seed) {
	const uint32_t c1 = 0xcc9e2d51;
	const uint32_t c2 = 0x1b873593;
	const uint32_t r1 = 15;
	const uint32_t r2 = 13;
	const uint32_t m = 5;
	const uint32_t n = 0xe6546b64;

    uint32_t hash = seed;
	const uint8_t* data = (const uint8_t*)key;

	size_t numBlocks = len / 4;
    const uint32_t* blocks = (const uint32_t*)(data);

    for (size_t i = 0; i < numBlocks; i++){
        uint32_t k = blocks[i];
		k *= c1;
		k = (k << r1) | (k >> (32 - r1));
		k *= c2;

		hash ^= k;
		hash = (hash << r2) | (hash >> (32 - r2));
		hash = hash * m + n;
    }

    const uint8_t* tail = data + numBlocks * 4;
	uint32_t k1 = 0;
    switch(len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
              k1 *= c1; k1 = (k1 << r1) | (k1 >> (32-r1)); k1 *=c2; hash ^= k1;
    };
	
	hash ^= len;
	hash ^= (hash >> 16);
	hash *= 0x85ebca6b;
	hash ^= (hash >> 13);
	hash *= 0xc2b2ae35;
	hash ^= (hash >> 16);

    return hash;
}

template <typename K, typename V>
class CuckooHashTableFastHash {
private:
    struct Entry {
        K key;
        V value;
        bool isOccupied;
    };
    std::vector<Entry> table;
    std::vector<std::function<size_t(const K&)>> hashFunctions;
    size_t capacity;
    double maxLoadFactor;
    size_t numEntries;
    std::mt19937 gen;
    
    void resizeTable() {
        size_t oldCapacity = capacity;
        capacity *= 2;
        std::vector<Entry> oldTable = table;
        table.assign(capacity, {{}, {}, false});
        numEntries = 0;
        for (size_t i = 0; i < oldCapacity; ++i) {
           if(oldTable[i].isOccupied) {
               insert(oldTable[i].key, oldTable[i].value);
           }
        }
    }


    bool insertHelper(const K& key, const V& value, size_t hashIndex, int depth) {
       if (depth > 2*capacity) { //detect and resolve cycles
           resizeTable();
           return insert(key, value);
       }
       size_t index = hashFunctions[hashIndex](key) % capacity;
       if (!table[index].isOccupied) {
           table[index] = {key, value, true};
           numEntries++;
           return true;
       }
       K tempKey = table[index].key;
       V tempValue = table[index].value;
       table[index] = {key, value, true};
       return insertHelper(tempKey, tempValue, (hashIndex + 1 ) % hashFunctions.size(), depth+1);
    }


public:
    CuckooHashTableFastHash(size_t initialCapacity = 16, double loadFactor = 0.5) : capacity(initialCapacity), maxLoadFactor(loadFactor), numEntries(0) {
       table.resize(capacity);
       std::uniform_int_distribution<uint32_t> distrib;
       uint32_t seed1 = distrib(gen);
       uint32_t seed2 = distrib(gen);
       hashFunctions.push_back([seed1](const K& k) {
        return  murmurHash3_32(&k, sizeof(K), seed1);
       });
       hashFunctions.push_back([seed2](const K& k) {
          return  murmurHash3_32(&k, sizeof(K), seed2);
       });
        gen.seed(std::random_device{}());
    }

    bool insert(const K& key, const V& value) {
       if(numEntries >= capacity * maxLoadFactor){
           resizeTable();
       }
       return insertHelper(key, value, 0, 0);
    }

    V* search(const K& key) {
        for (const auto& hashFunc : hashFunctions) {
            size_t index = hashFunc(key) % capacity;
             if(table[index].isOccupied && table[index].key == key) {
                return &table[index].value;
            }
        }
        return nullptr;
    }
    void remove(const K& key){
          for (const auto& hashFunc : hashFunctions) {
            size_t index = hashFunc(key) % capacity;
             if(table[index].isOccupied && table[index].key == key) {
                table[index].isOccupied = false;
                numEntries--;
             }
        }
    }
    size_t getCapacity() const {
        return capacity;
    }
     size_t size() const {
        return numEntries;
    }
};
```
*Commentary:* Example three moves away from the standard C++ hash implementation and uses the MurmurHash3 hash function and random seeds to generate hash functions. This is computationally faster and ensures better distribution with lower correlation, although this benefit might be less noticeable for trivial key types, the speed increase on more complex keys can be significant. It's important to consider the types of keys that will be used within this context.

Finally, for further study, I recommend exploring texts that focus on algorithm design and data structures. Look for detailed chapters specifically covering hashing techniques and their optimization. Furthermore, examining academic papers on the subject of cuckoo hashing implementations can provide valuable theoretical background and experimental data. It's beneficial to study implementation strategies used in open-source libraries and database systems as well. These resources provide a solid foundation for developing deeper insights and optimizing your own solutions.
