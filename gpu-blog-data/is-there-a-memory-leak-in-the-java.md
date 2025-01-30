---
title: "Is there a memory leak in the Java application related to the number of loaded classes?"
date: "2025-01-30"
id: "is-there-a-memory-leak-in-the-java"
---
The assertion that a Java application exhibits a memory leak directly correlated with the number of loaded classes is nuanced. While classes themselves are metadata and don't inherently consume large amounts of heap memory, the underlying mechanisms of class loading, particularly when combined with classloaders, can indeed contribute to a form of memory accumulation if not handled correctly. The most common scenario isn't a leak in the traditional sense of failing to release allocated memory, but rather an increasing occupancy of PermGen/Metaspace, due to the repeated creation and isolation of custom classloaders.

I've encountered this issue several times across projects, most notably within a heavily plugin-based architecture where individual plugins required their own isolated runtime environments. The fundamental challenge arises because classloaders, once created, typically remain in memory, along with the classes they have loaded. This structure is deliberately designed to prevent class conflicts and to support the dynamic loading and unloading of code, but mishandled classloaders can easily lead to problems.

Let's break down the critical aspects. A classloader’s primary responsibility is to locate and load class files from various sources, such as the file system or network. When a class is loaded, its Class object representation is placed within the PermGen space (prior to Java 8) or Metaspace (Java 8 and later). Critically, a classloader keeps a reference to all of the classes it loads, and those classes, in turn, often hold static references to other objects. Consequently, unloading a class essentially requires unloading the classloader. The garbage collector cannot reclaim a classloader and its loaded classes if there are any active references to them. These references might come from other classloaders, running threads, static fields, or the application itself.

The problematic pattern generally arises when custom classloaders are dynamically created, used, and then discarded without proper cleanup or when they are repeatedly instantiated to load similar classes. Consider a system where each incoming web request leads to the creation of a new classloader. Each classloader will load its versions of commonly used classes, such as those in the java.lang package. These classes remain in memory until both the classloader and any objects that reference these classes are eligible for garbage collection. Since the classloaders are rarely, if ever, explicitly unreferenced, a slow accumulation of classloader and class definitions occurs, eventually filling the PermGen/Metaspace and resulting in a `OutOfMemoryError`. While technically not a heap leak, this behavior certainly manifests as a form of memory exhaustion directly related to the number of loaded classes through custom classloaders.

The issue escalates further if each classloader loads different versions of the same class name. If a parent-first delegation strategy isn't followed properly and classloaders inadvertently load redundant classes, then this amplifies the memory consumption. Parent-first means the classloader first asks its parent to load the class. In scenarios involving multiple custom classloaders, this behavior must be strictly adhered to. Neglecting these guidelines generates an uncontrolled expansion of loaded classes and, therefore, potential memory exhaustion.

Now, let's explore some code examples to clarify these concepts.

**Example 1: Simple Classloader Creation Without Proper Unloading**

This example demonstrates the basic creation and usage of a custom classloader, highlighting the issue of unchecked instantiation. This version lacks any attempts to dispose of the classloader or references to the classes it loads.

```java
import java.net.URL;
import java.net.URLClassLoader;

public class ClassLoaderLeakExample1 {
    public static void main(String[] args) throws Exception {
        for (int i = 0; i < 1000; i++) {
            URLClassLoader cl = new URLClassLoader(
                new URL[] {new URL("file:./target/classes/")},
                ClassLoader.getSystemClassLoader());
                
            Class<?> clazz = cl.loadClass("com.example.TestClass");
            Object obj = clazz.getDeclaredConstructor().newInstance();
            
            System.out.println("Instance created: " + obj.getClass().getName() +  ", Iteration: " + i);
            // The classloader 'cl' will be garbage-collected but classes still associated
            // with the classloader may not be, since they're still referenced by the
            // classloader.
        }
        System.out.println("All iterations complete. Check your Metaspace consumption.");
    }
}
```

Here, the code iterates 1000 times, creating a new `URLClassLoader` in each loop, and loads `com.example.TestClass`. While the `cl` variable goes out of scope, the classloader itself and its loaded classes, including `com.example.TestClass`, remain in memory, linked together, contributing to a slow but persistent accumulation. Note that this example presumes that 'com.example.TestClass' exists in 'target/classes/' directory.

**Example 2: Incorrect Unloading Attempt**

This example shows an attempt at unloading by setting the classloader to null. However, this is insufficient to prevent the memory from accumulating.

```java
import java.net.URL;
import java.net.URLClassLoader;

public class ClassLoaderLeakExample2 {
    public static void main(String[] args) throws Exception {
        for (int i = 0; i < 1000; i++) {
             URLClassLoader cl = new URLClassLoader(
                new URL[] {new URL("file:./target/classes/")},
                ClassLoader.getSystemClassLoader());
             Class<?> clazz = cl.loadClass("com.example.TestClass");
            Object obj = clazz.getDeclaredConstructor().newInstance();
            System.out.println("Instance created: " + obj.getClass().getName() +  ", Iteration: " + i);
            cl = null; //Attempt to unload
            // The classloader 'cl' is no longer referenced locally, but references from
            // other objects to loaded classes prevent garbage-collection.
        }
       System.out.println("All iterations complete. Check your Metaspace consumption.");
    }
}
```

The `cl = null` statement merely removes the local reference to the classloader. The classes loaded by the classloader still hold references to their classloader, so the classloader isn't eligible for garbage collection. There might also exist static references to classes which were loaded by these classloaders, preventing the classloader and loaded classes from being reclaimed.

**Example 3: Proper Classloader Unloading with WeakReferences and ContextClassLoader Replacement**

This example illustrates a more sophisticated method using a custom ClassLoader, weak references, and explicitly setting the thread context classloader to null before it goes out of scope. In this setup the classloader can be garbage collected as long as there are no strong references to it or the loaded classes.

```java
import java.lang.ref.WeakReference;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ClassLoaderLeakExample3 {
   public static void main(String[] args) throws Exception {

       ExecutorService executor = Executors.newFixedThreadPool(1);
        for (int i = 0; i < 1000; i++) {
           final int iter = i;
           executor.submit(() -> {
               try {
                  URLClassLoader cl = new URLClassLoader(
                        new URL[] {new URL("file:./target/classes/")},
                       ClassLoader.getSystemClassLoader());

                  Class<?> clazz = cl.loadClass("com.example.TestClass");
                  Object obj = clazz.getDeclaredConstructor().newInstance();
                   System.out.println("Instance created: " + obj.getClass().getName() +  ", Iteration: " + iter);

                  Thread.currentThread().setContextClassLoader(null); // Remove reference

                  // Create a weak reference to the classloader
                  WeakReference<URLClassLoader> weakCl = new WeakReference<>(cl);

                  // Allow classloader to be garbage collected
                  cl = null;
                  while(weakCl.get() != null) {
                      System.gc();
                       Thread.sleep(50);
                   }
                   System.out.println("Classloader for iteration " + iter + " has been garbage collected.");
               } catch(Exception e){
                    e.printStackTrace();
               }
            });
             Thread.sleep(10);
       }
         executor.shutdown();
       System.out.println("All iterations submitted. Monitoring Metaspace.");
   }
}
```

This example makes use of weak references which will be eligible for collection once there are no strong references to the URLClassLoader or objects instantiated with it. The main thread spawns a thread to create and load a class, and then sets its own context class loader to null. This breaks a key linkage point. The garbage collection is then invoked in a loop to help accelerate the reclamation. This illustrates a much more diligent approach to releasing resources associated with class loading. Note that in production code one would need a more reliable approach, which might involve explicit lifecycle management of classloaders in a container.

To delve deeper into this area, I recommend focusing on resources addressing:

1.  **Java Classloader Architecture**: Understanding the parent delegation model, class visibility, and different types of classloaders is essential.
2.  **Garbage Collection in Java**: Specifically, focus on how the garbage collector works and interacts with classloaders and loaded classes. Research generational GC.
3.  **Memory Management and Monitoring**: Study how to monitor PermGen/Metaspace utilization and related performance issues. Techniques include using JConsole, VisualVM, JProfiler or other profiling tools.
4.  **Advanced Classloading Patterns**: Explore techniques like OSGI or frameworks that implement more efficient and reliable strategies for classloading and dynamic class unloading, if needed. Consider how contexts can be used to scope resources and avoid long-lived references.

In conclusion, while it isn't accurate to call it a classic heap leak, the repetitive loading of classes via custom classloaders can indeed cause a memory exhaustion issue, primarily in the PermGen/Metaspace, due to the associated classloader structures accumulating if the classloaders are not properly released. Understanding the role of the classloader in managing classes and being mindful of its lifecycle is critical when working with plugin systems or dynamically generated code that relies on classloading, and helps to prevent these issues.
