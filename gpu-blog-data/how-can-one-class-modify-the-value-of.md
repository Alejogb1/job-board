---
title: "How can one class modify the value of an object from another class?"
date: "2025-01-30"
id: "how-can-one-class-modify-the-value-of"
---
The core challenge in allowing one class to modify an object instantiated in another lies in controlling access and maintaining encapsulation. Direct modification, while seemingly straightforward, often leads to tight coupling and brittle code. Instead, one should focus on establishing controlled communication pathways.

My experience in building distributed systems and concurrent applications has consistently underscored the importance of carefully managing data access.  Unrestricted modification from disparate areas of the codebase creates debugging nightmares and hinders maintainability. To address this, we need to explore mechanisms beyond simply accessing an object’s public fields. Instead, we should favor methods that mediate access, offering both the ability to read values and, where appropriate, to modify them.

There are several established approaches to facilitate modification of an object from another class, each with varying implications regarding coupling and flexibility. The most common are: 1) Public methods offering controlled modification; 2) Callback mechanisms; and 3) Using dependency injection alongside setters.

**1. Public Methods for Controlled Modification:**

The most fundamental and arguably the safest approach involves creating public methods within the class holding the object, specifically designed to modify its state. These methods act as gatekeepers, allowing external classes to influence the object's internal values but only through predefined interfaces. Crucially, this approach preserves encapsulation, the practice of bundling data and the methods that operate on that data within a single unit. This approach ensures the class maintaining the object can validate or transform the data being provided.

Consider a `Counter` class, which maintains a numerical count. If another class needs to modify that counter, providing a direct public member to the counter is not the recommended approach. Instead, the Counter class should expose methods like `increment()` or `setCount(int newCount)`.

```java
// Example: Counter class with controlled modification.
public class Counter {
    private int count;

    public Counter() {
        this.count = 0;
    }

    public int getCount() {
        return this.count;
    }

    public void increment() {
        this.count++;
    }

    public void setCount(int newCount) {
        if (newCount >= 0) {
            this.count = newCount;
        } else {
           System.out.println("Cannot set count to negative value");
        }

    }
}

// Example: External class using Counter
public class CounterClient {
   public static void main(String[] args){
     Counter myCounter = new Counter();
     System.out.println("Initial count: " + myCounter.getCount()); //Output: 0
     myCounter.increment();
     System.out.println("Count after increment: " + myCounter.getCount()); // Output: 1
     myCounter.setCount(10);
     System.out.println("Count after setCount: " + myCounter.getCount()); //Output: 10
     myCounter.setCount(-2); //Output: Cannot set count to negative value
   }
}
```

Here, the `CounterClient` class modifies the count by calling the public `increment()` and `setCount()` methods. The internal state of the `Counter` remains private and its modification is governed through these methods. This maintains control and allows additional logic within `setCount` (checking for negative numbers, in this instance).

**2. Callback Mechanisms:**

Callbacks provide a more flexible interaction model. In this pattern, the class holding the object provides a mechanism for external classes to supply a function or a method. This supplied code is executed when a certain event occurs in the holding class, allowing the external class to dynamically modify the state of the object as needed. This is especially effective when responding to events or reacting to changing conditions.

Suppose a `DataProcessor` class processes data. Instead of directly changing the processed data, it can invoke a supplied callback function to perform modifications on the data. This decouples the data processing logic from the modification strategy.

```java
// Example: DataProcessor with callback.
import java.util.function.Function;
import java.util.ArrayList;
import java.util.List;

public class DataProcessor {
    private List<String> data;

    public DataProcessor(List<String> initialData) {
        this.data = new ArrayList<>(initialData);
    }

    public List<String> processData(Function<String, String> modifier) {
        List<String> processedData = new ArrayList<>();
        for (String item : data) {
            String modifiedItem = modifier.apply(item); // invoke callback
            processedData.add(modifiedItem);
        }
        return processedData;
    }
}


// Example: External class using DataProcessor with modifier callback
public class DataProcessorClient {
  public static void main(String[] args){
    List<String> initialData = List.of("apple", "banana", "cherry");
    DataProcessor processor = new DataProcessor(initialData);
    Function<String, String> upperCaseModifier = str -> str.toUpperCase();
    List<String> modifiedData = processor.processData(upperCaseModifier);
    System.out.println("Modified Data: " + modifiedData); //Output: [APPLE, BANANA, CHERRY]
  }
}
```

In this example, the `DataProcessorClient` defines a function `upperCaseModifier` that converts a string to uppercase.  This function is then passed to the `processData` method as a callback. The `DataProcessor` does not know how modification works, merely that it must call the passed-in callback on each processed element.

**3. Dependency Injection with Setters:**

Dependency injection, combined with setter methods, offers another powerful way to influence the state of an object. In this scenario, instead of directly instantiating dependencies, a class receives the needed dependencies as constructor parameters or through setter methods.  This allows an external class to inject or change the specific dependency. If the dependency object itself provides methods for state modification, then the injecting class can influence the original object. This is most relevant in complex systems where component swapping or dynamic configuration is needed.

Consider a `Logger` class, whose behavior can be configured. Another class may need to set the logging level. The Logger class can then be implemented to allow a logging level setter, allowing other classes to determine the verbosity of the log output. The logger object itself is passed into another class.

```java
//Example: Logger class with Setter.
public class Logger {
  private LogLevel level = LogLevel.INFO;

  public enum LogLevel {
    DEBUG, INFO, WARN, ERROR
  }

  public void log(String message, LogLevel msgLevel) {
    if (msgLevel.ordinal() >= this.level.ordinal()){
       System.out.println(msgLevel + ": " + message);
    }
  }
  public void setLevel(LogLevel level) {
    this.level = level;
  }
}
// Example: External class using dependency injection and Logger
public class Service {
   private Logger logger;

   public Service(Logger logger){
      this.logger = logger;
   }
   public void doSomething(){
     logger.log("Starting service operation...", Logger.LogLevel.INFO);
     // Perform some service logic.
     logger.log("Service operation completed successfully", Logger.LogLevel.INFO);
   }
   public void doSomethingDebug(){
      logger.log("Debug message during debugging", Logger.LogLevel.DEBUG);
   }
   public void setLoggerLevel(Logger.LogLevel level){
      this.logger.setLevel(level);
   }

   public static void main(String[] args){
      Logger myLogger = new Logger();
      Service service = new Service(myLogger);
      service.doSomething();
      service.setLoggerLevel(Logger.LogLevel.DEBUG);
      service.doSomethingDebug();
   }
}

```

In this example, the `Service` class receives a `Logger` instance through its constructor (dependency injection). The `Service` can then use the logger’s `setLevel()` method to dynamically change the logging level for the entire execution. This provides the flexibility to change the logging behavior based on the needs of each execution environment.

**Recommendation:**

For more in-depth understanding of object-oriented design principles, research classic works on design patterns. The concepts of encapsulation, loose coupling, and dependency injection, as discussed above, are fundamental to building robust and scalable applications. Further explorations of design patterns like the Observer Pattern or Strategy Pattern, also related to this question, can be beneficial. Additionally, consider resources that focus on clean coding practices, which offer guidelines for making code more readable and maintainable, a crucial skill when dealing with cross-class interactions. Textbooks and online courses that focus on the specific paradigm used, either object oriented or other, will also greatly enhance understanding.
