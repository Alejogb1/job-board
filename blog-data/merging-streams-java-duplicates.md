---
title: "merging streams java duplicates?"
date: "2024-12-13"
id: "merging-streams-java-duplicates"
---

Alright so you're wrestling with merging streams in Java and getting duplicate entries thats a classic I've been there trust me I've coded through enough of these to fill a small library

Okay lets dive in straight into the mess and the fixes I am going to assume you are already familiar with streams and basic lambda concepts if not then I would suggest you first check out "Java 8 in Action" by Raoul-Gabriel Urma Mario Fusco and Alan Mycroft that book covers the basics of Java Streams pretty well go grab it now seriously I will wait

I've been doing this Java thing for a while now back in 2014 when Java 8 first dropped I actually spent a solid week trying to debug a particularly nasty stream merge issue at a startup I was at We were building this real time analytics dashboard and we were getting duplicate data on the frontend it was a nightmare It turned out we were merging multiple event streams that had overlapping data its always something right

Anyways back to the problem you are having so what we are dealing with is a basic merging issue with potentially duplicated items when combining multiple streams in Java Lets see what options we have

**The Problem**

The root of the issue when you merge streams is that the base `Stream.concat()` operation does not care about duplicates It just smashes streams end to end like a train wreck on the rails You basically have two streams say `streamA` and `streamB` that each have possibly duplicate elements within and between them When you concat them using `Stream.concat(streamA streamB)` you get a new stream that has every element from both streams irrespective of if these elements are duplicates

**Simple Cases**

Now lets explore how we can handle this with various different needs that arise

**Case 1 Distinct elements using HashSet**

The most straightforward approach is to use a `HashSet` to keep track of elements we've already encountered as we stream then only let distinct elements through HashSet efficiently checks for existing elements We can apply this by utilizing `filter` with the `HashSet` check like below

```java
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class StreamMergeDuplicates {

    public static void main(String[] args) {
        List<Integer> listA = Arrays.asList(1 2 3 2 4 5);
        List<Integer> listB = Arrays.asList(4 5 6 7 1);

        Stream<Integer> streamA = listA.stream();
        Stream<Integer> streamB = listB.stream();

        Set<Integer> seen = new HashSet<>();
        List<Integer> mergedDistinct = Stream.concat(streamA streamB)
                .filter(n -> seen.add(n))
                .collect(Collectors.toList());

        System.out.println("Merged Distinct using HashSet " + mergedDistinct);  // Output: [1, 2, 3, 4, 5, 6, 7]
    }
}
```

This is pretty simple but has a minor gotcha The stateful `HashSet seen` variable is captured in lambda that modifies it It works okay for single threaded streams but is not threadsafe in parallel streams This is important thing to keep in mind if you decide to use parallel streams In this case it should work great but be mindful of it

**Case 2 Distinct elements using `distinct()` Method**

Java Streams has a built in method for removing duplicates `distinct()` this method use the `equals()` and `hashCode()` methods to identify duplicates You will have to override these methods if you are dealing with a custom Object The `distinct()` method is stateless and can work well with parallel streams unlike our `HashSet` approach

```java
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class StreamMergeDuplicates {

    public static void main(String[] args) {
      List<String> listA = Arrays.asList("apple" "banana" "apple" "cherry");
      List<String> listB = Arrays.asList("cherry" "date" "fig" "banana");

      Stream<String> streamA = listA.stream();
      Stream<String> streamB = listB.stream();

      List<String> mergedDistinct = Stream.concat(streamA streamB)
                                      .distinct()
                                      .collect(Collectors.toList());

       System.out.println("Merged Distinct using distinct() " + mergedDistinct); // Output: [apple, banana, cherry, date, fig]
    }
}
```

This approach is often simpler and cleaner but it has a performance consideration It uses a `HashSet` internally to perform the deduplication The performance will depend on the size of streams being merged and the hashcode implementation of the objects you are working with For smaller streams its generally fine but for big data processing there are more optimal options we will not cover

**Case 3 Merging custom objects with deduplication using `equals` and `hashCode`**

So you want to merge objects not just primitives okay got it In this case overriding `equals()` and `hashCode()` is key for deduplication Lets make an example with a simple `Product` class

```java
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class StreamMergeDuplicates {

    static class Product {
        String name;
        double price;

        public Product(String name double price) {
            this.name = name;
            this.price = price;
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Product product = (Product) o;
            return Double.compare(product.price this.price) == 0 && Objects.equals(name product.name);
        }

        @Override
        public int hashCode() {
            return Objects.hash(name price);
        }
        @Override
        public String toString() {
            return "Product{" +
                    "name='" + name + '\'' +
                    " price=" + price +
                    '}';
        }
    }
    
    public static void main(String[] args) {
        List<Product> listA = Arrays.asList(new Product("Laptop" 1200.00) new Product("Mouse" 25.00) new Product("Keyboard" 75.00) new Product("Laptop" 1200.00));
        List<Product> listB = Arrays.asList(new Product("Keyboard" 75.00) new Product("Monitor" 300.00) new Product("Headphones" 100.00) new Product("Mouse" 25.00));
    
        Stream<Product> streamA = listA.stream();
        Stream<Product> streamB = listB.stream();
        
        List<Product> mergedDistinct = Stream.concat(streamA streamB)
                                              .distinct()
                                              .collect(Collectors.toList());
    
        System.out.println("Merged Distinct Custom objects " + mergedDistinct);
    }
}

// Output: Merged Distinct Custom objects [Product{name='Laptop' price=1200.0} Product{name='Mouse' price=25.0} Product{name='Keyboard' price=75.0} Product{name='Monitor' price=300.0} Product{name='Headphones' price=100.0}]
```

I added the `toString` method for easy printing You see how important `equals` and `hashCode` are right? Without these methods the `distinct` call would just return you all the elements since object equality is reference based by default in Java

**Considerations**

*   **Performance** For very large streams consider parallel processing using `parallelStream()` and `unordered()` to speed up the operation but bear in mind it does bring in the complexity we talked about with the `HashSet` state capture earlier on Also remember to benchmark first because it might not help in your specific use case If anything this might make the process slower due to the overhead of managing parallel processes
*   **Order** Stream operations maintain the encounter order except when `unordered()` operation is used `distinct()` does preserve the encounter order However if you dont care about the ordering then you can use `unordered` for better performance
*   **Object Identity** If your objects rely on object identity instead of field values for equality then you will have to implement a more specific `Comparator` and use other approaches to remove duplicates This is a more niche case that you will probably not need unless you are working with some very complex custom objects.
*   **Null Values** Keep in mind null values could be passed through stream depending on your input data and what you are doing with the stream after that In this case you need to handle possible `NullPointerExceptions`
*   **Memory** Keep in mind all these operations use memory as we store the stream in the memory for further processing if the streams are too big you may have to find another way. This is one of the most common issues with using streams improperly in Java as they can blow up your memory very quickly if you do not handle this part properly.

**In summary**

You have several options you can use the `distinct` method for most cases as it's simple and works for basic deduplication and its also threadsafe If you are dealing with custom objects make sure to implement `equals()` and `hashCode()` correctly If you have performance issues in mind think about `parallelStream()` and `unordered()` for large streams while being mindful of the trade offs

Okay I think that covers most of it you've got the tools go forth and deduplicate your streams I've wasted enough time on this in the past I hope it helps you out but if you get stuck again do not hesitate to ask here we all have been there
Oh and one more thing Why was the Java developer always unhappy? Because he never got arrayed!

Anyways back to the streams and if you feel like you need more in-depth explanation I would suggest you read "Effective Java" by Joshua Bloch he is a well-known and respected expert in Java and he also covers collections and streams really well
