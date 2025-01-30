---
title: "What is the definition of perm space?"
date: "2025-01-30"
id: "what-is-the-definition-of-perm-space"
---
PermGen, short for Permanent Generation, was a specific region within the heap of the Java Virtual Machine (JVM) used in earlier versions of Java, primarily before Java 8. Its primary purpose was to store class metadata – information about the structure and behavior of loaded classes, like method bytecode, constant pool data, and static variables. This is a critical distinction from the rest of the heap, which is primarily dedicated to object instantiation and dynamic data storage.

The need for a distinct, permanent area arose from the JVM's architecture. Initially, the JVM needed a predictable, finite region to store class information that was assumed to have a relatively fixed size during the application’s lifecycle. This design was based on the understanding that classes are usually loaded once and their metadata remains constant throughout the application's execution, except for some specific cases involving dynamically generated classes. Allocating this data to the standard heap, along with live objects, would have complicated garbage collection efforts and potentially impacted performance negatively, making PermGen seem like a reasonable solution at the time.

I spent years working on large-scale Java EE applications, and I encountered PermGen issues frequently. In many instances, particularly with server restart sequences during deployments or classloader leaks, the PermGen would become full, leading to the dreaded `java.lang.OutOfMemoryError: PermGen space`. This error occurred because the class metadata allocated in PermGen never undergoes the same kind of garbage collection as normal objects in the heap. If a large number of new classes were generated dynamically (e.g., using reflection, frameworks like Spring, or proxies), or if older classloaders weren't properly cleaned, PermGen would quickly exhaust its finite space, even with a relatively small heap.

PermGen's limitation was the static nature of its sizing and garbage collection behavior. It was sized at startup by command-line parameters such as `-XX:MaxPermSize`, meaning once allocated, it couldn't be resized during runtime. The garbage collection strategy was limited primarily to full garbage collection cycles, which were infrequent and could result in application pauses. This inflexible approach often resulted in a resource bottleneck, making it difficult to diagnose the true nature of memory usage problems. Furthermore, as modern application architectures evolved, with increased use of frameworks and dynamic bytecode generation, the static allocation of PermGen became increasingly problematic, leading to increased out-of-memory errors.

Here is an illustration using code examples which highlights how class metadata is crucial, even when dealing with simple constructs:

**Code Example 1: Simple class loading.**

```java
public class MyClass {
    private static int staticVar = 10;
    private int instanceVar;

    public void myMethod() {
       //Method body
    }
    
     public static void main(String[] args) {
        MyClass obj1 = new MyClass(); //Instance object of MyClass
        MyClass obj2 = new MyClass(); //Another instance object of MyClass
        System.out.println(obj1.instanceVar);
        System.out.println(MyClass.staticVar);
     }
}
```

**Commentary:** In this example, when `MyClass` is loaded by the JVM, all the metadata associated with it—its class name, the signature of `myMethod`, the static variable `staticVar` (and its initial value), the instance variable `instanceVar` — is placed in the PermGen (or MetaSpace in Java 8+) . The objects `obj1` and `obj2`, allocated by using the `new` operator, however, are placed in the normal heap. The class metadata remains constant throughout the execution of the application. In earlier Java versions, if we loaded a large number of classes dynamically, or had many classloaders, we would start approaching the limit set by the `-XX:MaxPermSize`. Note that in the main function, we are also accessing static members directly through the class. Static members also reside in permgen space.

**Code Example 2: Dynamic class creation using reflection.**

```java
import java.lang.reflect.Proxy;

interface MyInterface {
    void someMethod();
}
public class MyProxyTest {
    public static void main(String[] args) {
       for(int i=0; i < 100000; i++){ //A large loop generating proxies
           MyInterface proxy = (MyInterface) Proxy.newProxyInstance(
                MyInterface.class.getClassLoader(),
                 new Class[] { MyInterface.class },
                 (proxy1, method, args1) -> {
                     return null;
                 });
            proxy.someMethod();
        }
    }
}
```

**Commentary:** This example showcases the problem with excessive dynamic class generation. Every call to `Proxy.newProxyInstance` generates a new proxy class at runtime. Each of these generated classes has its metadata stored in PermGen. If this loop runs many times, it could easily fill PermGen and trigger the out-of-memory error. This is a common scenario in applications that use reflection extensively or that rely on libraries that generate dynamic proxies, such as many Java application frameworks. If these proxy objects and related class loaders are not cleaned up properly, the problem is even worse. This example highlights that permgen's issues were not just about size but about improper cleanup of resources as well.

**Code Example 3:  Classloader and its metadata lifecycle.**

```java
import java.net.URL;
import java.net.URLClassLoader;
import java.io.File;

public class ClassloaderLeak {
    public static void main(String[] args) throws Exception {
        for (int i = 0; i < 5000; i++) {
            URL[] urls = {new File(".").toURI().toURL()};
            URLClassLoader classLoader = new URLClassLoader(urls);
            Class<?> clazz = classLoader.loadClass("MyClass");
           // classLoader.close() will help with leak, if supported in Java version
        }

    }
}
```

**Commentary:**  In this example, every iteration of the loop creates a new `URLClassLoader`. Each of these classloaders loads the `MyClass` creating a new set of class metadata and the class loader instances. While the specific behavior of class loader unloading and garbage collection can be complex and vary across JVM implementations, in older versions of Java, repeated loading and creation of many classloaders will almost certainly lead to the growth of data in PermGen. Often, this growth was not automatically cleaned up as PermGen was designed to hold permanent data. Though not a direct PermGen issue, this often manifested as a PermGen error. This is why class loader leaks were frequently correlated with PermGen memory issues. Closing the classloader and nullifying references to it helps with preventing issues, though it might not be supported across older versions of Java.

In Java 8, PermGen was entirely removed and its role was taken over by a concept called MetaSpace, which, while serving the same function of storing class metadata, offers several key improvements. Unlike PermGen, MetaSpace is allocated in native memory and its size is not fixed at startup. Instead, MetaSpace is sized dynamically, and its memory is released back to the operating system when class metadata is no longer needed. This eliminates the need for a hard limit on the space dedicated to class metadata, resolving many of the out-of-memory errors that plagued earlier Java versions.

For those wanting to learn more about this topic, I suggest exploring the following resources: *Understanding the Java Virtual Machine* by Bill Venners, which provides an in-depth look into the JVM internals. Articles and documentation related to the different Java garbage collection algorithms are also beneficial, as they help understand how memory is managed both within and outside the heap and why these changes were so crucial. Finally, examining the Java platform specification regarding class loading and reflection will assist greatly in grasping how class metadata is created and utilized.
