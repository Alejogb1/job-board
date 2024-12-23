---
title: "What memory types are causing Spring Boot's OOM errors?"
date: "2024-12-23"
id: "what-memory-types-are-causing-spring-boots-oom-errors"
---

Okay, let's talk about those pesky OutOfMemoryErrors in Spring Boot applications. It's a beast I've grappled with countless times across various projects, from microservices humming in the background to monoliths handling critical user transactions. These errors, often cryptic, usually point to a problem with how the Java Virtual Machine (JVM) manages memory. Specifically, in the context of Spring Boot, we're typically dealing with a handful of memory regions that become battlegrounds for these OOM errors. Understanding them deeply, from first principles, is key to preventing these situations.

First, let’s address the heap. The heap is where all objects are allocated; this includes all your application's instances, like the beans managed by Spring, any lists, maps, and basically anything created with `new`. When this area fills up, you hit a `java.lang.OutOfMemoryError: Java heap space`. This is, in my experience, the most common OOM error you'll encounter. It usually means there's either a memory leak, where objects aren't being released by the garbage collector (GC) because there's a strong reference to them somewhere, or that the heap itself is simply undersized for the application's demands.

Another significant area is the method area, or metaspace (introduced in Java 8, replacing the permgen space). The metaspace stores class definitions, static variables, method information, and the constant pool. While a `java.lang.OutOfMemoryError: Metaspace` error isn't as frequent as a heap space error, it's a serious concern when it does happen. A continually growing metaspace typically suggests that classes are being loaded but never unloaded, perhaps due to dynamic class generation gone rogue, excessively deep class hierarchies, or certain classloaders failing to release their resources correctly. This is the one that often leads to head-scratching sessions for days, trust me.

Lastly, although often overlooked in initial diagnosis, there’s the native memory area. While the JVM uses the heap, metaspace, and thread stack for most object allocation, it also uses native memory for tasks such as direct byte buffers (used for I/O operations), the JIT compiler, and internal JVM processes. A `java.lang.OutOfMemoryError: Direct buffer memory` usually stems from the program not releasing these buffers properly. This issue can also lead to a more generic `OutOfMemoryError` if the native OS runs out of allocatable memory due to a Java process that is holding on to a large quantity of non-heap memory. Native memory related OOMs are tricky, often requiring a dive into the native memory tracking or tools like JProfiler to diagnose accurately.

Let me illustrate each of these with examples based on things I've encountered in past projects.

**Example 1: Heap Space Exhaustion (Leak Scenario)**

Imagine an application where we cache user data, using a static `HashMap`. I once saw code like this:

```java
import java.util.HashMap;
import java.util.Map;

public class UserCache {

    private static final Map<String, User> userCache = new HashMap<>();

    public static void cacheUser(String userId, User user) {
        userCache.put(userId, user);
        //No explicit removal from cache based on any criteria
    }

    public static User getUser(String userId){
        return userCache.get(userId);
    }

    //Assume User is a simple class representing user information
    static class User {
        String name;
        int age;

        User(String name, int age){
            this.name = name;
            this.age = age;
        }
    }

    public static void main(String[] args) throws InterruptedException {

        for(int i=0; i< 1000000; i++) {
            cacheUser(String.valueOf(i), new User("test" + i, i % 99));
        }

        Thread.sleep(600000); // Sleep for 10 minutes to keep the process running.
    }
}

```

Here, as more and more users are cached, the `userCache` `HashMap` keeps growing without bound, eventually leading to a `java.lang.OutOfMemoryError: Java heap space`. The fix? Implement a cache eviction strategy (e.g., using a least-recently-used cache) or set a size limit.

**Example 2: Metaspace Exhaustion (Dynamic Class Loading)**

I once encountered a complex data processing application that heavily used dynamic class generation and loading, something like the following (this is an extremely simplified version for demonstration purposes, naturally the real code was much more involved):

```java
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;

public class ClassGenerator {

    public static void main(String[] args) throws MalformedURLException, ClassNotFoundException, InvocationTargetException, IllegalAccessException, InterruptedException {
        for (int i=0; i< 10000; i++) {
           generateAndLoadClass("DynamicClass" + i);
        }

        Thread.sleep(600000); // Sleep for 10 minutes to keep the process running.
    }


   static void generateAndLoadClass(String className) throws MalformedURLException, ClassNotFoundException, InvocationTargetException, IllegalAccessException {
        String classContent = String.format("public class %s { public void display() { System.out.println(\"Hello from %s\"); } }", className, className);
        try {
            byte[] byteCode =  compile(className, classContent);

            // Create a new class loader for each dynamic class.
            URLClassLoader classLoader = new URLClassLoader(new URL[]{}, ClassLoader.getSystemClassLoader());
            Class<?> dynamicClass = classLoader.defineClass(className, byteCode, 0, byteCode.length);
            Object instance = dynamicClass.getDeclaredConstructor().newInstance();

            Method displayMethod = dynamicClass.getMethod("display");
            displayMethod.invoke(instance);


            // classLoader = null; // Unloading the class loader will not automatically unload the class.
         }
         catch(Exception ex){
            ex.printStackTrace();
         }
   }

   //Simulated compilation. Not a real compiler
   static byte[] compile(String className, String classContent) {
        // Very Simplified simulation of class compilation. In reality this is very complex
        return classContent.getBytes();
    }
}
```

This loop creates new class loaders for each class. The issue is each time we do this, the generated classes accumulate in metaspace, since there is no garbage collection on the classes. In real systems, if classes are loaded through many custom classloaders, these can lead to memory leaks. While the `classLoader` variable goes out of scope the loaded class still references the classloader and the classes in it, therefore they are not garbage collected leading to `java.lang.OutOfMemoryError: Metaspace`

The solution often involves using reusable classloaders (if feasible) or reducing the number of classes generated, sometimes using techniques like bytecode manipulation at build time instead of at runtime. Proper resource management with custom class loaders is crucial and can be extremely tricky to get right.

**Example 3: Native Memory Exhaustion (Direct Byte Buffers)**

Consider a scenario where a microservice processes large files using NIO. Here’s a simplified look at code that can lead to a direct memory leak:

```java
import java.nio.ByteBuffer;

public class ByteBufferExample {

    public static void main(String[] args) throws InterruptedException {

      for(int i=0; i< 10000; i++){
         processFile(1024 * 1024 * 10); // 10 MB
      }
        Thread.sleep(600000); // Sleep for 10 minutes to keep the process running.
    }

    static void processFile(int size) {
          ByteBuffer buffer = ByteBuffer.allocateDirect(size); // Allocate Direct Memory
          // Simulate filling the buffer with data
          for (int i = 0; i < buffer.capacity(); i++) {
              buffer.put((byte) (i % 256));
          }
         //Note: No Buffer releasing implemented, this is the cause of the problem

         // buffer = null; // Setting the reference to null does not release the native memory
    }
}
```

Here, each call to `processFile` allocates 10MB of direct byte buffer memory but never releases it. Even if the `buffer` variable goes out of scope and is set to null, the underlying memory allocated remains claimed since there is no explicit cleanup. This eventually exhausts the native memory, resulting in a `java.lang.OutOfMemoryError: Direct buffer memory`. In real code, the native memory usually accumulates gradually depending on how frequently it is used, making it especially hard to diagnose.

The solution here requires ensuring that direct byte buffers are released correctly after use. Ideally this should be done inside a `finally` block or by calling `buffer.clear()` and letting the garbage collector release the buffer.

To further solidify your understanding, I highly recommend exploring “Java Performance: The Definitive Guide” by Scott Oaks. It provides an in-depth explanation of JVM internals, including memory management. Another essential reference is “Understanding the JVM” by Bill Venners, which delves into the subtleties of class loading and memory areas. Also, remember to consult the official JVM documentation from Oracle for detailed information about memory management and garbage collection. Finally, familiarize yourself with tools like jconsole, jvisualvm, or commercial profilers such as JProfiler or YourKit, which are indispensable for diagnosing these types of errors in real-world settings.

In summary, OOM errors in Spring Boot typically arise from issues with heap space, metaspace, or direct buffer memory. Understanding each area and implementing appropriate memory management techniques is key to building robust and scalable applications. Keep a sharp eye on these areas, employ the right diagnostic tools, and don't underestimate the value of good coding practices, and you will eventually be able to avoid running into memory errors in your code.
