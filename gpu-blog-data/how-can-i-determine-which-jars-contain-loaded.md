---
title: "How can I determine which JARs contain loaded classes?"
date: "2025-01-30"
id: "how-can-i-determine-which-jars-contain-loaded"
---
The Java Virtual Machine (JVM) maintains a map of loaded classes and their origins, allowing programmatic access to the JAR files from which those classes were loaded. This information is critical for debugging classloading issues, auditing dependency conflicts, or understanding the runtime environment's structure.

My experience troubleshooting several complex application deployments has driven home the importance of programmatically discovering the source JAR of loaded classes.  The most straightforward method utilizes the `ProtectionDomain` associated with each class and its corresponding `CodeSource`. These classes, part of the core Java API, provide a handle to the JAR file or directory where a class is located.

**Core Mechanisms and Explanation**

Each loaded class within a JVM instance possesses a `Class` object. The `Class` object exposes the `getProtectionDomain()` method, which returns a `ProtectionDomain` instance. This `ProtectionDomain` encapsulates security-related information for the class. More pertinent to this problem, the `ProtectionDomain` contains the `getCodeSource()` method which, when invoked, returns a `CodeSource` object, if one exists. The `CodeSource`, finally, holds the crucial URL that points to the origin of the class. This URL usually represents the location of the JAR file. If the class was not loaded from a file but was generated dynamically or is a part of the JVM's core classes, then `getCodeSource()` might return `null`.

Therefore, to pinpoint the JAR file (or directory) for a loaded class, we must traverse this chain: `Class` -> `ProtectionDomain` -> `CodeSource` -> `URL`.

This mechanism relies on the JVM's class loading process where each classloader manages the resources it's responsible for loading. The standard classloaders (bootstrap, extension, and application) record the origin information when they load a class from a JAR. When a custom classloader is used, it also has to handle this process of linking classes with their sources during loading.

**Code Examples**

Here are three code examples illustrating different approaches to retrieve this information:

**Example 1: Retrieving a Single Class’s Source**

This example demonstrates how to find the source JAR for a single, known class. It uses a try-catch block to handle potential null-pointer exceptions from calling getCodeSource and its related methods.

```java
import java.net.URL;
import java.security.CodeSource;
import java.security.ProtectionDomain;

public class ClassSource {

    public static void main(String[] args) {
        Class<?> targetClass = String.class; // Example: get source of String class

        URL classLocation = getClassLocation(targetClass);

        if (classLocation != null) {
            System.out.println("Source of " + targetClass.getName() + ": " + classLocation.getPath());
        } else {
            System.out.println("Could not determine source for: " + targetClass.getName());
        }
    }

    private static URL getClassLocation(Class<?> cls) {
        try {
            ProtectionDomain protectionDomain = cls.getProtectionDomain();
            if (protectionDomain != null) {
                CodeSource codeSource = protectionDomain.getCodeSource();
                if (codeSource != null) {
                     return codeSource.getLocation();
                }
            }
            return null;
        } catch (NullPointerException e){
            return null;
        }
    }
}

```

*Commentary*: This code first retrieves the `Class` object for `String`. Then it calls `getClassLocation`, which navigates through the `ProtectionDomain` and `CodeSource` to extract the location. If `getCodeSource` returns null, or an exception occurs, it will return null and print a message accordingly. This handles the case where a class may be generated at runtime, or be a system level class. The `getPath()` is used to get the specific location string.

**Example 2: Finding JARs of Loaded Classes in the Current Classloader**

This example iterates through all loaded classes in the current classloader and prints their sources. This is useful for analyzing the dependencies of your application.

```java
import java.net.URL;
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Set;

public class LoadedClasses {
    public static void main(String[] args) {
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        Set<Class<?>> loadedClasses = getLoadedClasses(classLoader);

        for (Class<?> cls : loadedClasses) {
            URL classLocation = getClassLocation(cls);
            if (classLocation != null) {
                System.out.println("Class: " + cls.getName() + ", Source: " + classLocation.getPath());
            } else {
                System.out.println("Class: " + cls.getName() + ", Source: Unknown");
            }
        }
    }

    private static Set<Class<?>> getLoadedClasses(ClassLoader classLoader) {
        Set<Class<?>> classes = new HashSet<>();
        Enumeration<java.net.URL> resources;
        try {
            resources = classLoader.getResources("");
            while (resources.hasMoreElements()){
              URL url = resources.nextElement();
              java.io.File file = new java.io.File(url.getFile());
              if(file.isDirectory()){
                   java.io.File[] classFiles = file.listFiles( (dir, name) -> name.endsWith(".class") );
                   for (java.io.File classFile : classFiles){
                      String className = classFile.getPath().replace(file.getPath(), "").replace("/", ".").replace(".class", "");
                      className = className.startsWith(".") ? className.substring(1) : className;
                      try {
                         Class<?> c = classLoader.loadClass(className);
                         classes.add(c);
                      } catch (ClassNotFoundException ignored) {
                       // Class may not be on class path, so ignore and continue
                      }
                   }
              }
            }
            return classes;

        } catch (java.io.IOException e) {
            return classes; // return empty list
        }
    }
     private static URL getClassLocation(Class<?> cls) {
        try {
            ProtectionDomain protectionDomain = cls.getProtectionDomain();
            if (protectionDomain != null) {
                CodeSource codeSource = protectionDomain.getCodeSource();
                if (codeSource != null) {
                     return codeSource.getLocation();
                }
            }
            return null;
        } catch (NullPointerException e){
            return null;
        }
    }
}
```
*Commentary:*  The `getLoadedClasses` method in this example attempts to get the resources of the context class loader.  It filters these resources looking for directories, then it assumes all `.class` files under these directories represent class names. It then loads these classes using `loadClass` and adds them to a set.  This approach of scanning all directory resources for `.class` files is rather simplistic and will not capture all loaded classes. In particular, classes loaded from JAR files are not found using this resource enumeration. However, it highlights how class loading can be explored using Java's core APIs. It then iterates through the list and prints their JARs using the `getClassLocation()` function from example one.  This approach has its limitations and is not intended to be production quality, but demonstrates a method for identifying some class files.

**Example 3: Using Reflection to Access the System Class Loader (Careful Use)**

This example explores using reflection to access the system class loader and inspect its loaded classes. This approach, while providing more access to classes, requires more caution.

```java
import java.lang.reflect.Field;
import java.net.URL;
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SystemClassLoader {
    public static void main(String[] args) {
        ClassLoader systemClassLoader = ClassLoader.getSystemClassLoader();
        List<Class<?>> allLoadedClasses = getAllLoadedClasses(systemClassLoader);

         for (Class<?> cls : allLoadedClasses) {
            URL classLocation = getClassLocation(cls);
            if (classLocation != null) {
                System.out.println("Class: " + cls.getName() + ", Source: " + classLocation.getPath());
            } else {
                System.out.println("Class: " + cls.getName() + ", Source: Unknown");
            }
        }
    }

     private static List<Class<?>> getAllLoadedClasses(ClassLoader classLoader) {
        List<Class<?>> loadedClasses = new ArrayList<>();
        try {
            Class<?> classLoaderClass = Class.forName("java.lang.ClassLoader");
            Field classesField = classLoaderClass.getDeclaredField("classes");
            classesField.setAccessible(true);

             java.util.Vector<?> classes = (java.util.Vector<?>) classesField.get(classLoader);

            for( Object classObj : classes ) {
              loadedClasses.add( (Class<?>) classObj );
            }

           } catch (ClassNotFoundException | NoSuchFieldException | IllegalAccessException | ClassCastException ignored) {
            // Fail silently - these errors indicate problems, but we are still doing our best.
        }
        return loadedClasses;
    }

    private static URL getClassLocation(Class<?> cls) {
        try {
            ProtectionDomain protectionDomain = cls.getProtectionDomain();
            if (protectionDomain != null) {
                CodeSource codeSource = protectionDomain.getCodeSource();
                if (codeSource != null) {
                     return codeSource.getLocation();
                }
            }
            return null;
        } catch (NullPointerException e){
            return null;
        }
    }
}
```

*Commentary*: This example uses reflection to access a non-public field called "classes" within the `ClassLoader` class. The field is of type `java.util.Vector` and is believed to hold the loaded classes.   While accessing non-public fields via reflection is not advisable for production, this method reveals how all classes loaded by a class loader (including the system class loader) can be examined. It again demonstrates the steps to retrieve the locations of the loaded classes. This approach has certain restrictions: it will not work if a SecurityManager is in place that prevents accessing non-public members using reflection. Furthermore, the internal structure of the `ClassLoader` is not guaranteed, so this may break in different JVM versions.

**Resource Recommendations**

For a deeper understanding, consult resources on the following topics:

*   **Java Classloaders:** This will give you context into how classes are found and loaded at runtime. Pay specific attention to the delegation hierarchy and custom classloader implementations.
*   **Java Security API:** Learn about `ProtectionDomain` and `CodeSource` and their roles within the Java security model. This will provide further understanding of their function.
*   **Reflection:** Understanding reflection is useful to exploring parts of Java's core API that are normally hidden from a developer. Learn how to use it responsibly.
*   **JVM Specifications**: Understand the internals of JVM's operation. Focus on class loading and runtime data areas. This is an advanced but useful topic.

By understanding these concepts and exploring the core API, one can effectively determine the JAR files that contain loaded classes within a Java application. This capability is invaluable for troubleshooting deployment and dependency issues.
