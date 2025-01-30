---
title: "What is the difference between a `List<Integer>` and a raw `var` array?"
date: "2025-01-30"
id: "what-is-the-difference-between-a-listinteger-and"
---
In my experience optimizing Java applications over the past decade, the subtle distinctions between parameterized `List<Integer>` and raw `var` arrays often dictate performance and maintainability. The primary difference lies in type safety and dynamic sizing capabilities versus static sizing and potentially unchecked data access.

A `List<Integer>`, specifically when parameterized using generics, represents a *typed* collection. The `<Integer>` designation explicitly informs the Java Virtual Machine (JVM) and compiler that this `List` is designed to hold objects of the `Integer` class. This enforced type constraint prevents accidental insertion of non-`Integer` objects, leading to compile-time errors rather than runtime `ClassCastException`s. Furthermore, `List` is an interface, typically implemented by classes like `ArrayList` or `LinkedList`, offering dynamic resizing. This means the underlying data structure automatically adjusts its capacity as elements are added or removed, avoiding the common pitfall of buffer overflows or wasted memory with pre-allocated array sizes.

Conversely, a raw `var` array, once defined with a specific size, is *statically sized*. The `var` keyword in Java, introduced in Java 10, performs type inference. When initializing an array directly like `var myArray = new int[5];`, `myArray` will become an array of type `int[]`. This array can only hold primitive `int` values, and its size (5 in this example) is fixed at instantiation. While `var` simplifies initialization syntax, it doesn’t introduce any type-safe collection-like behavior inherent in `List<Integer>`. Modification of the size is not a direct operation; you would need to create a new array with the desired size and copy the old data over, resulting in more complex logic and potential performance overhead. Also, `int[]` and `Integer[]` are handled differently. The former is a primitive array while the later is an object array. This subtle distinction has implication when doing things like sorting, since primitive types have their own optimized sorting logic.

The difference in memory management should also be considered. An `ArrayList`, for instance, doesn't allocate its entire maximum capacity at the outset; rather, it increments allocation as necessary and will reallocate the underlying storage if that becomes insufficient. On the other hand, an `int` array allocates memory for the number of elements defined at its instantiation, which consumes a larger block upfront.

Regarding usability, the `List` interface provides a rich set of methods, including `.add()`, `.remove()`, `.get()`, `.size()`, and others, facilitating seamless operations with collections. Raw arrays only offer direct element access via index, requiring more manual handling for adding, removing, or managing size. The lack of these built-in functions increases the complexity of code when working directly with arrays. The absence of explicit type information with raw arrays also demands more vigilance when iterating, adding or retrieving from the data structure; this is because no automatic type-safety check exists at compile-time.

Let’s illustrate these points with examples:

**Example 1: Using `List<Integer>`:**

```java
import java.util.ArrayList;
import java.util.List;

public class ListExample {
    public static void main(String[] args) {
        List<Integer> integerList = new ArrayList<>();
        integerList.add(10);
        integerList.add(20);
        integerList.add(30);

        // Compile-time error if you try to add a String:
        // integerList.add("This is a string");

        int element = integerList.get(1);
        System.out.println("Element at index 1: " + element);
        System.out.println("Size of the list: " + integerList.size());

        integerList.remove(Integer.valueOf(20));
        System.out.println("List after removing 20: " + integerList);
     }
}

```

This example demonstrates the type safety of the `List<Integer>`. Attempting to add a `String` will result in a compile-time error, preventing runtime issues. Dynamic resizing is also implicitly handled by the `ArrayList` implementation. The ease with which one can add, remove and get items is evident using its built-in methods.

**Example 2: Using a raw `var` array:**

```java
public class ArrayExample {
   public static void main(String[] args) {
        var intArray = new int[3];
        intArray[0] = 10;
        intArray[1] = 20;
        intArray[2] = 30;

        int element = intArray[1];
        System.out.println("Element at index 1: " + element);
        System.out.println("Array length: " + intArray.length);

       // Cannot dynamically add elements; must create a new array or use helper methods

       //Incorrect usage: this is illegal and cannot grow a primitive array
       // intArray = { 10, 20, 30, 40};

        for(int num : intArray)
        {
          System.out.println(num);
        }


    }
}
```

In this case, `intArray` is inferred as an `int[]`. Access is done using index only. Attempting to add an element beyond the initial size will result in an `ArrayIndexOutOfBoundsException`, while no compile-time checks occur for incompatible types since it is a primitive array. Also, note the lack of methods to remove an element or to dynamically expand the array.

**Example 3: Illustrating resizing behavior:**

```java
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public class ResizingExample {
    public static void main(String[] args) {
        List<Integer> integerList = new ArrayList<>();

        for (int i = 0; i < 100; i++) {
            integerList.add(i);
        }
       System.out.println("List size after adding 100 elements: " + integerList.size());

       var intArray = new int[5];

        for(int i=0; i<5; i++)
        {
          intArray[i] = i;
        }

       // Cannot add beyond the allocated 5 elements
       // This will cause ArrayIndexOutOfBoundsException
       // intArray[5] = 5;

       // Workaround to resize a primitive array:
       intArray = Arrays.copyOf(intArray, 10);

       for(int i=5; i<10; i++)
        {
            intArray[i] = i;
        }


       System.out.println("Array length after copying to a larger array: " + intArray.length);
       for(int num : intArray)
       {
         System.out.println(num);
       }
    }
}
```
This example highlights the dynamic resizing of the `List`. We can easily add 100 elements without worrying about the size, and the list will adjust the storage transparently. However, for primitive arrays, expanding their size involves creating a completely new array and copying all elements, necessitating extra operations and can be costly. The above example show the use of the `Arrays.copyOf` method.

In conclusion, while `var` arrays might seem simpler for certain use cases involving statically-sized data, `List<Integer>` offers superior type safety, dynamic resizing, and a more user-friendly interface. The choice between the two depends on specific requirements. When type safety, dynamic resizing, and operations are paramount, `List<Integer>` is the preferred option. Raw arrays are applicable when you have known sizes or require raw primitive array performance, but this should come after understanding the increased risk in runtime errors and the necessary manual handling.

For further exploration of the subject, several excellent resources on Java collections and arrays exist. Texts focusing on Java best practices, data structures and algorithms would also assist in understanding the implications and appropriate selection of these data structures. Furthermore, the official Oracle documentation for the Java Collection Framework provides exhaustive information on available APIs and their usage patterns. Understanding the principles behind these collections will enable developers to make informed choices, optimize application performance, and prevent common pitfalls.
