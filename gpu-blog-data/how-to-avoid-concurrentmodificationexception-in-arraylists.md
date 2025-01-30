---
title: "How to avoid ConcurrentModificationException in ArrayLists?"
date: "2025-01-30"
id: "how-to-avoid-concurrentmodificationexception-in-arraylists"
---
The core issue with `ConcurrentModificationException` in Java's `ArrayList` stems from the inherent design of the `ArrayList` class: its iterators maintain a version number that's checked against the internal modification count of the list.  Any structural modification—addition, removal, or clearing—increments this count. If an iterator detects a mismatch during iteration, the exception is thrown to signal an invalidated iteration state. This is a crucial detail often overlooked; it's not simply about multiple threads; single-threaded code can also trigger this if modifications are made during iteration.  My experience debugging multithreaded applications has underscored this repeatedly.

This behavior is designed to protect data integrity.  Attempting to iterate and modify concurrently leads to unpredictable results and potential data corruption. Ignoring the exception is dangerous, masking potentially severe bugs. The solution is not to suppress the exception, but to refactor your code to prevent the problematic concurrent modification.

**1.  Using `Iterator.remove()`:**

The `ArrayList`'s `Iterator` provides a safe method for removing elements during iteration.  Crucially, this method directly interacts with the `ArrayList`'s internal mechanisms, updating the modification count correctly and avoiding the exception.  Using any other removal method within the iteration loop will be problematic.  Consider this example:

```java
ArrayList<String> myList = new ArrayList<>(Arrays.asList("apple", "banana", "cherry", "date"));

Iterator<String> iterator = myList.iterator();
while (iterator.hasNext()) {
    String fruit = iterator.next();
    if (fruit.equals("banana")) {
        iterator.remove(); // Safe removal
    }
}

System.out.println(myList); // Output: [apple, cherry, date]
```

This code iterates through the list.  If "banana" is encountered, `iterator.remove()` is called. This directly interacts with the list’s internal state and updates the modification count accordingly, preventing the exception.  Trying to use `myList.remove("banana")` within this loop would result in a `ConcurrentModificationException`.

**2.  Creating a copy for modification:**

If you need to modify the list extensively while iterating, creating a copy is a straightforward, albeit potentially less efficient, solution. This decouples the iteration process from the modification, thus avoiding any conflicts. This approach is particularly valuable when dealing with complex filtering or transformation operations.

```java
ArrayList<String> myList = new ArrayList<>(Arrays.asList("apple", "banana", "cherry", "date"));
ArrayList<String> newList = new ArrayList<>(myList); // Create a copy

for (String fruit : myList) { // Iterate through the original
    if (fruit.length() > 5) {
        newList.remove(fruit); // Modify the copy
    }
}

myList = newList; // Replace original with modified copy (if necessary)
System.out.println(myList); // Output: [apple, banana, cherry]
```

In this example, we create a copy `newList`.  Modifications are performed on this copy.  Finally,  the original list can be replaced, depending on requirements. Note the iteration happens on the original list; it does not get modified.


**3.  Using Concurrent Collections:**

For multithreaded scenarios, the use of concurrent collections is strongly recommended.  `CopyOnWriteArrayList` is a prime example.  This class uses a copy-on-write strategy: modifications create a new internal array, leaving existing iterators unaffected.  While this approach has a performance overhead for frequent modifications, it's the preferred solution for thread safety.

```java
CopyOnWriteArrayList<String> myList = new CopyOnWriteArrayList<>(Arrays.asList("apple", "banana", "cherry", "date"));

for (String fruit : myList) {
    if (fruit.equals("banana")) {
        myList.remove(fruit); // Safe removal in concurrent environment
    }
}
System.out.println(myList); // Output: [apple, cherry, date]

// Example with multiple threads (Illustrative only; proper thread management needed in a real application)
ExecutorService executor = Executors.newFixedThreadPool(2);
executor.submit(() -> {
    for (String fruit : myList) {
        System.out.println("Thread 1: " + fruit);
    }
});
executor.submit(() -> myList.add("fig"));
executor.shutdown();
```

Here, `CopyOnWriteArrayList` handles concurrent access without throwing `ConcurrentModificationException`.  The example also demonstrates a simple (though incomplete) multithreaded scenario to highlight its capability.  In real-world multithreaded applications, comprehensive synchronization and thread management is crucial and often requires more sophisticated techniques beyond the scope of this example.


**Resource Recommendations:**

The Java Concurrency in Practice book.
The Java Collections Framework documentation.
Effective Java by Joshua Bloch.


In conclusion,  handling `ConcurrentModificationException` effectively requires understanding its root cause: the mismatch between iterator version and list modification count.  The choice of solution—using `Iterator.remove()`, creating copies, or employing concurrent collections—depends on the specific needs of your application and the nature of the modifications. Choosing the wrong solution will lead to hidden bugs and unreliable results. Remember, suppressing the exception should be your last resort and only after thoroughly understanding the implications and risks associated with doing so.  Prioritize correctness and maintainability over short-term gains in efficiency.
