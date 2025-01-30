---
title: "Why are map lookups slow in this Java application?"
date: "2025-01-30"
id: "why-are-map-lookups-slow-in-this-java"
---
The observed performance bottleneck in map lookups within this Java application primarily stems from suboptimal hash function implementation and subsequent collision handling within the `HashMap` structure. My experience working on high-throughput data processing applications has demonstrated that seemingly insignificant choices in data structure usage, specifically hash function design, can lead to exponential increases in lookup times, drastically impacting overall application performance.

A `HashMap` relies on hashing to distribute key-value pairs across internal buckets. When a lookup occurs, the key's hash code is computed, determining the target bucket. An ideally performing hash function distributes keys evenly across buckets, leading to near O(1) retrieval times. However, poorly designed hash functions produce clusters of keys within specific buckets, transforming lookups into linear searches within those buckets, effectively negating the performance benefits of a hash map. When many keys hash to the same bucket, a ‘hash collision’ occurs, and `HashMap` uses a linked list (or a tree for newer Java versions in specific cases) to manage those entries. A high collision rate degrades performance from near-constant time to O(n), where 'n' represents the number of colliding keys in the bucket. This degradation becomes especially pronounced when the map is heavily populated and frequent lookups are required.

Furthermore, the default `hashCode()` implementation of custom classes often contributes to the problem. If `hashCode()` is not properly overridden for a class used as a key in the `HashMap`, it typically resorts to the object's memory address. This leads to distinct objects, even if logically equivalent, being hashed to different buckets, which might seem to be a good distribution at first. The problem arises when the application creates many new equivalent objects, each with its own unique memory address, resulting in a new unique hash value. The `HashMap` then has to manage a growing number of entries, effectively becoming a list and causing the same performance degradation associated with hash collisions, but now for every logically same object. Similarly, if a custom `hashCode()` function generates many of the same hash codes, collision rates will rise significantly, degrading lookup performance.

Let’s illustrate with examples. Assume we are processing customer data using a custom `Customer` class as keys.

**Example 1: Incorrect `hashCode()` implementation:**

```java
public class Customer {
    private String customerId;
    private String name;

    public Customer(String customerId, String name) {
        this.customerId = customerId;
        this.name = name;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Customer customer = (Customer) obj;
        return customerId.equals(customer.customerId);
    }
    // Notice the missing hashCode() method.

    public String getCustomerId() { return customerId; }
    public String getName() { return name; }
}

// Usage
public void processCustomerData(){
        Map<Customer, String> customerMap = new HashMap<>();
        // Creating new Customer objects instead of using same ones or implementing proper hashCode() causes performance bottleneck.
        for(int i=0; i < 100000; i++) {
           Customer c = new Customer(String.valueOf(i), "Customer_"+i);
            customerMap.put(c, "Data_"+i);
        }

        for(int i=0; i < 100000; i++){
           Customer lookupC = new Customer(String.valueOf(i), "Customer_"+i);
            customerMap.get(lookupC);  // Slow lookup because each Customer is a new object resulting in different hashcodes
        }

}
```
In this scenario, the `Customer` class only overrides the `equals()` method, but not `hashCode()`. This causes `HashMap` to utilize the default `hashCode()` inherited from the `Object` class which is based on object identity (memory address).  While `equals()` correctly identifies objects based on the `customerId`, each `new Customer(...)` invocation creates a new object with a different hash code, making lookups exceedingly slow despite the intended logical equality.  The application effectively does not use the HashMap efficiently, causing it to devolve into essentially a costly `List` lookup.

**Example 2: A poor custom `hashCode()` Implementation:**

```java
public class Customer {
    private String customerId;
    private String name;

    public Customer(String customerId, String name) {
        this.customerId = customerId;
        this.name = name;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Customer customer = (Customer) obj;
        return customerId.equals(customer.customerId);
    }

    @Override
    public int hashCode() {
        return 7; // All Customer objects have the same hashcode
    }

    public String getCustomerId() { return customerId; }
    public String getName() { return name; }
}
// Usage in the similar way as before
```
Here, while `hashCode()` is overridden, the implementation always returns a constant. This is an extremely poor hashing strategy. The `HashMap` effectively reduces to a single bucket and each lookup entails traversing all previously added objects which is extremely inefficient. This leads to severe performance degradation due to massive hash collisions and linear search within a single bucket.

**Example 3: Improved `hashCode()` implementation:**

```java
import java.util.Objects;

public class Customer {
    private String customerId;
    private String name;

    public Customer(String customerId, String name) {
        this.customerId = customerId;
        this.name = name;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Customer customer = (Customer) obj;
        return customerId.equals(customer.customerId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(customerId);
    }

    public String getCustomerId() { return customerId; }
    public String getName() { return name; }
}
// Usage in the similar way as before
```
This implementation uses `Objects.hash()` to calculate a hash code based on the `customerId`, the key that defines equality for `Customer` objects. This results in a more appropriate distribution across `HashMap` buckets and reduces collisions. The critical aspect here is that for logically equal `Customer` objects (same `customerId`), `hashCode()` will return the same value. This is crucial for `HashMap`'s efficient lookup capabilities. Furthermore, using the provided `Objects.hash()` allows for more robust hashcode generation, which in turn leads to less collisions.

In addition to correcting the hash function implementation, it is also important to monitor the load factor of the `HashMap`. The load factor is a threshold that, when exceeded, triggers the rehashing of the map. Rehashing involves reallocating and redistributing existing key-value pairs into the new buckets. Although a well-distributed hash function results in a low collision rate, even those can cause issues when the number of entries in the `HashMap` reaches near the limit of the capacity * loadFactor. The default load factor is 0.75, which provides a good trade-off between space and time costs. However, depending on the dataset, it may be necessary to increase it for a more memory-efficient map at the cost of some collision increase, or to lower it for a less memory-efficient map that has fewer collisions.
Several other techniques can be used for optimizing `HashMap` usage, which include: pre-sizing maps when the size is known in advance, using `EnumMap` for enum-based keys, or using concurrent collections such as `ConcurrentHashMap` where concurrent modifications are required, which is beyond the scope of current request.

To further enhance understanding of hashing and its application in data structure performance, I would recommend consulting resources that cover general data structure and algorithm concepts, specifically delving into chapters on hash functions, collision resolution techniques, and the analysis of time complexity within data structures. Also, it is beneficial to consult resources focusing on Java data structures, particularly those detailing the internal workings and performance implications of various `java.util` classes, including `HashMap`. Additionally, reviewing articles focusing on effective `hashCode()` and `equals()` method implementation for custom classes will also be beneficial. These resources will deepen the understanding of map operation performance and facilitate the application of optimized techniques to future development projects.
