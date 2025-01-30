---
title: "Why does this code produce the output using maps and sets?"
date: "2025-01-30"
id: "why-does-this-code-produce-the-output-using"
---
The behavior observed when using maps and sets in code, specifically when dealing with complex objects as keys or elements, stems directly from how JavaScript handles object equality. Unlike primitive types which are compared by value, objects are compared by reference. This means that two object literals, even if they possess identical properties and values, are not considered equal unless they refer to the *same* memory location. Therefore, if we intend for unique objects to act as distinct keys in a map or distinct elements in a set, we must ensure they are indeed different references in memory.

My experience troubleshooting several data management pipelines has repeatedly underscored this concept. I once spent a considerable amount of time debugging a seemingly simple user profile update process where identical-looking user objects kept overriding each other within a Map. It turned out that the user objects were being freshly constructed each time, which, while structurally identical, resulted in new memory references. Correcting this required an understanding of the underlying object comparison.

Let’s first examine how maps use object keys. A map, as a data structure, relies on the hash of a key to quickly locate the corresponding value. In JavaScript, object keys utilize their reference (memory location) to compute this hash. If two objects, although bearing identical properties, are created independently, they will be assigned different memory locations, resulting in distinct hash values. Therefore, a Map will treat them as distinct keys.

Consider the following code snippet:

```javascript
const map = new Map();

const obj1 = { id: 1, name: 'Alice' };
const obj2 = { id: 1, name: 'Alice' };

map.set(obj1, 'Value 1');
map.set(obj2, 'Value 2');

console.log(map.size);  // Output: 2
console.log(map.get(obj1));  // Output: Value 1
console.log(map.get(obj2));  // Output: Value 2
```

In this example, even though `obj1` and `obj2` have the same properties and values, they are separate objects located in different memory locations. When they are used as keys in the Map, they are treated as two independent keys. Therefore, the map ends up holding two entries, each associated with a different key object. The retrieval using `map.get()` will correctly return the value associated with the exact key reference provided. If we were to attempt retrieval with a third object, identical to `obj1` or `obj2`, but distinct in terms of reference, it would return `undefined`.

The same fundamental principle applies to Sets. A set, by definition, only stores unique values. Uniqueness is also determined by object reference, not structural equivalence. This means that two objects with identical content, if they are separate objects in memory, will both be included within a Set.

Here’s an example to illustrate set behavior:

```javascript
const set = new Set();

const obj3 = { id: 2, name: 'Bob' };
const obj4 = { id: 2, name: 'Bob' };

set.add(obj3);
set.add(obj4);

console.log(set.size); // Output: 2
console.log(set.has(obj3)); // Output: true
console.log(set.has(obj4)); // Output: true
```

Similarly to the Map example, here, `obj3` and `obj4`, despite being identical in their properties, are distinct objects in memory, thus both being considered unique within the Set. Therefore, both are included and `set.size` reflects this fact.

In a scenario where we *do* require objects with identical structures to be treated as equal, we must implement a strategy that does not rely solely on reference comparison. One common approach is to convert the object into a string representation and then use that string as a key in the map or as an element in a set. However, this approach requires considering the stability of object serialization: the order of the properties in a stringified object might be inconsistent across executions, even when the underlying object properties are the same, leading to inconsistencies. One solution is to sort the object properties alphabetically before stringification.

Here is an example demonstrating this behavior:

```javascript
const mapWithSerializedKeys = new Map();

const obj5 = { id: 3, name: 'Charlie' };
const obj6 = { name: 'Charlie', id: 3 }; // Properties in different order
const obj7 = { id: 3, name: 'David'};

function serializeObject(obj) {
    const sortedKeys = Object.keys(obj).sort();
    const sortedObj = {};
    for (const key of sortedKeys) {
        sortedObj[key] = obj[key];
    }
    return JSON.stringify(sortedObj);
}


const key5 = serializeObject(obj5);
const key6 = serializeObject(obj6);
const key7 = serializeObject(obj7);

mapWithSerializedKeys.set(key5, "Value for Charlie 1");
mapWithSerializedKeys.set(key6, "Value for Charlie 2");
mapWithSerializedKeys.set(key7, "Value for David");


console.log(mapWithSerializedKeys.size); // Output: 2 (key5 and key6 are serialized the same)
console.log(mapWithSerializedKeys.get(serializeObject({ id:3, name: "Charlie"}))); //Outputs: Value for Charlie 2, because the second setting overwrote the first with the serialized form of the object.
```

In this example, `obj5` and `obj6`, although initially having different property order, become identical string keys after serialization. As a result, only a single entry for them exists in the map, and the last key/value pair assigned will be stored. This demonstrates that by serializing object to a string using `JSON.stringify` and sorting keys, we can treat structurally identical objects as the same key in Maps.

To summarize, the core reason for the observed behavior stems from how JavaScript treats objects: by reference. Maps and Sets rely on this reference to establish the uniqueness of keys or elements. If you need to use objects based on structural or semantic equality, you must implement custom logic for object comparison, for example, by serializing objects to JSON strings with sorted keys, or using a custom equality check function during lookups.

For further exploration, I recommend studying the documentation on `Map` and `Set` objects within the ECMAScript specification. Additionally, reviewing materials focusing on object equality and hash table implementation in JavaScript will enhance one's understanding. The concept of object references is foundational, and having a firm grasp of this will facilitate advanced manipulations of collections.  Books on data structures and algorithms often cover hash table implementations and object equality. Furthermore, resources that detail JavaScript’s object model provide invaluable background for dealing with these concepts.
