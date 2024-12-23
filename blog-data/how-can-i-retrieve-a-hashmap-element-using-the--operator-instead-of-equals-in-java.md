---
title: "How can I retrieve a hashmap element using the `==` operator instead of `.equals()` in Java?"
date: "2024-12-23"
id: "how-can-i-retrieve-a-hashmap-element-using-the--operator-instead-of-equals-in-java"
---

Alright, let’s unpack this one. I recall a particular project back in the early 2010s where we inadvertently stumbled upon a similar issue with `HashMap` comparisons, though not precisely this scenario. It involved a custom key class, and that experience taught me a lot about how `HashMap` works under the hood, and how crucial it is to understand the difference between `==` and `.equals()`. The short answer, of course, is that you generally *cannot* reliably retrieve a `HashMap` element using `==` to compare keys. The intended, and correct, mechanism is always through the `equals()` method, coupled with a properly implemented `hashCode()`. Let's delve into why, and then I’ll show you some practical examples.

The fundamental issue boils down to how `HashMap` is structured. Internally, it uses a hashing algorithm to determine where to store key-value pairs. This means when you put an element into the `HashMap`, it doesn’t just store it in some arbitrary place; it computes a hash code based on your key's `hashCode()` method, which determines the bucket location for this key-value pair. To *retrieve* an element, this hash code computation happens again for the key you're searching for. Then, inside the bucket, the `equals()` method is used to compare keys. This multi-stage process ensures that only objects that are considered 'equal' according to the `equals()` definition will resolve to the same element.

The `==` operator, in contrast, checks for *reference equality*. That is, it will only return true if the two references point to the exact same object in memory. In most cases when you’re dealing with keys in a hashmap, you'll have different object instances that, while representing the same *value*, are not the same object *reference*. This disparity makes the `==` operator essentially useless for retrieving from a `HashMap` using anything but identical object instances.

Let's illustrate with some code examples. First, a common scenario where relying on `==` fails:

```java
public class Key {
    private String value;

    public Key(String value) {
        this.value = value;
    }

    // Note: Missing hashCode() and equals() implementations.
    // Intentionally left out to demonstrate the problem.
}

public class Main {
    public static void main(String[] args) {
        HashMap<Key, String> map = new HashMap<>();
        Key key1 = new Key("test");
        map.put(key1, "value1");

        Key key2 = new Key("test"); // Different object, same value
        System.out.println("Retrieving with equals(): " + map.get(key2)); // Returns null

        System.out.println("Retrieving with ==" + (key1 == key2)); // prints false;

        //The following is not an appropriate way to retrieve
        //from a hashmap, you can't rely on the object being the same instance.
        //HashMap retrieval with == will almost certainly fail in non-trivial use cases.
    }
}
```

In this snippet, even though `key1` and `key2` conceptually represent the same string value “test”, they are two distinct objects. The `HashMap.get()` method internally calls `.equals()` – in this case, the inherited `Object.equals()` which just performs `==`. Therefore `map.get(key2)` returns null because there is no match within the Hashmap. The comparison `key1==key2` is of course false. This demonstrates how using `==` (or a default `equals()` implementation) fails even in the presence of objects that represent the same concept.

Now, let's fix this with a proper `equals()` and `hashCode()` implementation:

```java
public class Key {
    private String value;

    public Key(String value) {
        this.value = value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Key key = (Key) o;
        return value.equals(key.value);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }
}

public class Main {
    public static void main(String[] args) {
        HashMap<Key, String> map = new HashMap<>();
        Key key1 = new Key("test");
        map.put(key1, "value1");

        Key key2 = new Key("test"); // Different object, same value
        System.out.println("Retrieving with equals(): " + map.get(key2)); // Now returns "value1"
        System.out.println("Retrieving with ==" + (key1 == key2)); // prints false; still.
    }
}
```

Here, we’ve overridden the `equals()` method to compare the underlying string values of `Key` instances and overridden `hashCode()` to return the `hashCode` of the String values. The critical piece here is that `key1.equals(key2)` now evaluates to true. When `HashMap.get(key2)` is called, it hashes `key2`, finds the bucket, then, crucially, it compares `key2` to the existing key in that bucket using `.equals()` and finds the match. So even though `key1 == key2` evaluates to false, `key1.equals(key2)` will be true due to the correct overriding, making retrieval possible. `==` is still irrelevant to the behavior of `HashMap` retrieval; the comparison is still done using `equals()`.

Finally, consider a scenario where you could, in principle, *make* `==` work by reusing the same object instance, but this is generally an anti-pattern and highly problematic:

```java
public class Key {
    private String value;

    public Key(String value) {
        this.value = value;
    }

    // Still using the hashCode and equals implemented in the example above.
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Key key = (Key) o;
        return value.equals(key.value);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }
}


public class Main {
    public static void main(String[] args) {
       HashMap<Key, String> map = new HashMap<>();
       Key key = new Key("test");
       map.put(key, "value1");


        System.out.println("Retrieving with equals(): " + map.get(key)); // Correctly returns "value1"
       System.out.println("Retrieving with ==" + (map.containsKey(key))); // Using the same key instance for containsKey works, just as it does with get.
       System.out.println("Retrieving with ==" + (map.get(key))); //Retrieving with `get()` still requires a key instance. This line would also work

       Key keySameReference = key;

        System.out.println("Retrieving with ==" + (key == keySameReference)); // prints true
        System.out.println("Retrieving with equals using second reference" + map.get(keySameReference)); // also works, because it's the same reference

    }
}

```
In this last example, I have two different references, `key`, and `keySameReference` that refer to the same object instance. This does mean `key==keySameReference` will return true, as they reference the same object in memory. We can use this single key reference when calling `map.containsKey(key)`, `map.get(key)` and it will work without issues. However, it's vital to note that relying on *reference* equality like this is usually a bad approach and would break the fundamental paradigm of `HashMap`, which expects to determine equality via the `equals()` method not through comparison of object *references*. You absolutely cannot rely on this kind of comparison, the behavior is a side-effect of using the same reference.

For a deeper dive, I strongly recommend studying "Effective Java" by Joshua Bloch, particularly the sections on `equals` and `hashCode`. Additionally, for a more theoretical understanding of hashing and data structures, "Introduction to Algorithms" by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein will be incredibly beneficial.

To summarize: using `==` to retrieve elements from a `HashMap` is generally incorrect and unreliable, because hashmap retrieval requires `equals()`. The correct solution involves properly overriding `equals()` and `hashCode()` in your key class. Reference equality as in the last example is not the intended nor correct use of a `HashMap`. It is important to always keep the core design and intended use of data structures in mind for code that is reliable and maintainable. I hope this clears things up.
