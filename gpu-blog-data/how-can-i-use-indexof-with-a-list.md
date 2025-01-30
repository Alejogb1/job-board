---
title: "How can I use `indexOf` with a `List` containing elements from an `int''''`?"
date: "2025-01-30"
id: "how-can-i-use-indexof-with-a-list"
---
Accessing elements within a `List` derived from a multi-dimensional `int[][]` using `indexOf` presents a challenge because `indexOf` relies on object equality, not structural equivalence for non-primitive types. This distinction is crucial for understanding why a naive approach fails and how to properly achieve the desired functionality. My experience in various data processing projects has highlighted this very common point of confusion.

The core issue stems from how Java handles object comparisons. `indexOf` on a `List` of objects (which an `int[][]` becomes when added to a `List`) compares memory addresses using the `.equals()` method. Two arrays holding identical integer values will be distinct objects with different memory addresses. Consequently, a new `int[]` with the same values as an element within the `List` will not be found using `indexOf`. Instead, the `indexOf` method will return -1 because the objects are not identical in memory.

To illustrate, consider a scenario where a grid of sensor readings, represented as an `int[][]`, is loaded and then converted into a `List` of `int[]` for processing. We might desire to locate a specific row of readings within the `List`. A direct `indexOf` will not work. The solution requires us to implement a custom comparison mechanism, typically by iterating over the list and performing an element-by-element value check within each nested array.

Here's a breakdown with code examples:

**Example 1: Demonstrating the Failure of a Direct `indexOf`**

```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IndexOfArrayFailure {

    public static void main(String[] args) {
        int[][] sensorReadings = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        List<int[]> readingList = new ArrayList<>(Arrays.asList(sensorReadings));

        int[] searchRow = {4, 5, 6};
        int index = readingList.indexOf(searchRow);

        System.out.println("Index found using indexOf: " + index); // Output: -1
    }
}
```

In this example, `sensorReadings` represents sample sensor data. It’s converted into a `List<int[]>` called `readingList`. When searching for `searchRow` (which holds the same values as the second row in the initial array) using `indexOf`, we get -1. This clearly illustrates that even with the same numerical data, `indexOf` fails due to the difference in object identities. The search array and the corresponding array inside the `readingList` are distinct objects residing at different memory locations, even if they contain the same values.

**Example 2: Implementing a Custom Search using a Loop**

```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CustomSearch {
    public static void main(String[] args) {
        int[][] sensorReadings = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        List<int[]> readingList = new ArrayList<>(Arrays.asList(sensorReadings));

        int[] searchRow = {4, 5, 6};
        int index = findIndexOfArray(readingList, searchRow);

        System.out.println("Index found with custom search: " + index); // Output: 1
    }

    public static int findIndexOfArray(List<int[]> list, int[] target) {
         for (int i = 0; i < list.size(); i++) {
              if (Arrays.equals(list.get(i), target)) {
                    return i;
              }
         }
        return -1;
    }
}
```

Here, I introduce a static method `findIndexOfArray`. This method iterates through the `List` and uses `Arrays.equals` to compare each `int[]` with the `target` array. `Arrays.equals` performs a deep comparison, checking the equality of each element at each index. This approach accurately identifies the location of the desired array. This custom method is needed because the standard `indexOf` operates at object level identity, not value comparison.

**Example 3: Using Java Streams for a Functional Approach**

```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;

public class StreamSearch {
    public static void main(String[] args) {
        int[][] sensorReadings = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

         List<int[]> readingList = new ArrayList<>(Arrays.asList(sensorReadings));

        int[] searchRow = {4, 5, 6};

        OptionalInt index = findIndexOfArrayStream(readingList,searchRow);


        System.out.println("Index found with stream search: " + index.orElse(-1)); // Output: 1

    }

    public static OptionalInt findIndexOfArrayStream(List<int[]> list, int[] target) {
        return java.util.stream.IntStream.range(0,list.size())
                    .filter(i -> Arrays.equals(list.get(i), target))
                    .findFirst();
        }

}
```

This third example leverages Java Streams for a more concise solution.  `IntStream.range(0, list.size())` generates a sequence of indices for the list.  The `.filter` method then keeps the index `i` only if the array at that index matches the `target` according to `Arrays.equals`.  Finally, `findFirst()` returns the first index if a match exists. If no match is found an empty `OptionalInt` is returned, so I used `.orElse(-1)` to emulate the behavior of a standard `indexOf`, and to output a simple `-1` in the absence of an index. Although seemingly shorter, I find that in practice, this stream implementation can sometimes be slightly less performant than a basic for loop on larger lists. But its use of functional style may be preferred in many cases.

When dealing with lists of arrays, specifically from multi-dimensional primitive type arrays, it is essential to recognize that standard library functions like `indexOf` perform object identity comparisons by default.  When value-based comparisons are needed for arrays, custom mechanisms that use `Arrays.equals` to compare the array contents element by element, are necessary. I have found that choosing between iterative or stream approaches often depends on the project's coding style and whether performance of a loop is required, or if stream based code offers sufficient performance while improving readability.

For further exploration, consider reviewing documentation related to Java collections and array handling. Specifically, focus on the `List` interface, the `ArrayList` implementation, and the `Arrays` utility class. Understanding object identity and the nuances of `.equals()` is fundamental to working with collections of objects in Java. Resources focusing on functional programming concepts within Java will also help to further use stream processing effectively.
