---
title: "How can proxy classes be used for self-profiling?"
date: "2025-01-30"
id: "how-can-proxy-classes-be-used-for-self-profiling"
---
When developing a complex system handling large volumes of real-time data, I discovered a need for fine-grained performance insights without heavily instrumenting the core application logic with intrusive profiling code. This need led me to implement proxy classes as a mechanism for self-profiling, enabling the collection of execution statistics with minimal impact on the primary code path.

A proxy class, at its core, is a structural design pattern that provides a surrogate or placeholder for another object. It controls access to this underlying object, allowing the insertion of pre- or post-processing logic. In the context of self-profiling, the proxy class acts as an intermediary, intercepting method calls intended for the target object, and recording metrics like execution time, call frequency, and even passed arguments before delegating the actual work. The key advantage is that profiling is encapsulated within the proxy, leaving the underlying object (the one being profiled) entirely agnostic to this process. This separation of concerns makes the system maintainable and prevents performance profiling code from polluting the business logic.

The profiling data can be logged, aggregated, or exposed through monitoring endpoints depending on the desired level of analysis and integration with existing systems. Moreover, this approach allows for dynamic enabling or disabling of profiling without modifying the code of the target object; this can be crucial in production environments where performance overhead must be controlled. The choice of profiling data to record, such as elapsed time, can be tailored depending on the specific area that requires observation. The degree of invasiveness of the proxy is also configurable allowing for a balance between accuracy and overhead.

Let's consider a concrete example. Suppose we have a service responsible for processing incoming user requests; this is represented by an interface `RequestProcessor` and a corresponding concrete implementation.

```java
// Interface for Request Processing Service
interface RequestProcessor {
    String processRequest(String request);
}

// Concrete implementation of RequestProcessor
class DefaultRequestProcessor implements RequestProcessor {
    @Override
    public String processRequest(String request) {
       // Simulate processing delay
       try {
          Thread.sleep((long)(Math.random() * 200));
        }
       catch (InterruptedException ex){
        Thread.currentThread().interrupt();
        }
        return "Processed: " + request;
    }
}
```

We can create a proxy class, `ProfilingRequestProcessor`, that wraps the concrete implementation and collects profiling data:

```java
// Proxy class implementing RequestProcessor, with profiling
import java.util.HashMap;
import java.util.Map;

class ProfilingRequestProcessor implements RequestProcessor {
    private final RequestProcessor target;
    private final Map<String, Long> methodCallCounts = new HashMap<>();
    private final Map<String, Long> methodTotalTime = new HashMap<>();
    public ProfilingRequestProcessor(RequestProcessor target) {
        this.target = target;
    }

    @Override
    public String processRequest(String request) {
        long startTime = System.nanoTime();
        String result = target.processRequest(request);
        long endTime = System.nanoTime();

        String methodName = "processRequest"; // Store method name to be used for counting
        methodCallCounts.put(methodName, methodCallCounts.getOrDefault(methodName, 0L) + 1);
        methodTotalTime.put(methodName, methodTotalTime.getOrDefault(methodName, 0L) + (endTime - startTime));

        return result;
    }
    public void printProfile(){
        System.out.println("Profiling Data");
        for (Map.Entry<String,Long> entry: methodCallCounts.entrySet()){
            String methodName = entry.getKey();
            long count = entry.getValue();
            long totalTime = methodTotalTime.get(methodName);
            System.out.println(methodName + ": Count =" + count + ", Total Execution Time =" + (totalTime/1000000.0) + "ms");
        }
    }
}
```

In this example, the `ProfilingRequestProcessor` intercepts calls to `processRequest`. It records the start and end times, computes the elapsed time, and updates a counter as well as the total execution time. The `printProfile` method could be expanded to export this data through a variety of mechanisms. Using this proxy is straightforward:

```java
public class Main {
    public static void main(String[] args) {
        // Create an instance of the concrete processor
        RequestProcessor defaultProcessor = new DefaultRequestProcessor();

        // Create a proxy around the concrete processor for profiling
        ProfilingRequestProcessor profilingProcessor = new ProfilingRequestProcessor(defaultProcessor);
        //Use the proxy instead of direct calls to the concrete processor
        for (int i =0; i<10; i++) {
          profilingProcessor.processRequest("Request " + i);
        }

        //Display collected profiling data
        profilingProcessor.printProfile();
    }
}
```

The `main` method demonstrates how the `ProfilingRequestProcessor` intercepts requests before they reach the underlying `DefaultRequestProcessor`. The original processor is unaware of the wrapping, and its code is untouched. This modular design makes it easy to swap out different proxy implementations for varied profiling strategies. For instance, a proxy could aggregate execution statistics across multiple threads or log them to a remote system.

Expanding on the previous example, we could introduce more detailed profiling, such as argument logging or specific error handling within the proxy. Let’s introduce a version of `ProfilingRequestProcessor` which also logs arguments.
```java
class ProfilingRequestProcessorWithArgs implements RequestProcessor {
    private final RequestProcessor target;
    private final Map<String, Long> methodCallCounts = new HashMap<>();
    private final Map<String, Long> methodTotalTime = new HashMap<>();
    private final Map<String, String> methodLastArgs = new HashMap<>();
    public ProfilingRequestProcessorWithArgs(RequestProcessor target) {
        this.target = target;
    }

    @Override
    public String processRequest(String request) {
        long startTime = System.nanoTime();
        String result = target.processRequest(request);
        long endTime = System.nanoTime();

        String methodName = "processRequest";
        methodCallCounts.put(methodName, methodCallCounts.getOrDefault(methodName, 0L) + 1);
        methodTotalTime.put(methodName, methodTotalTime.getOrDefault(methodName, 0L) + (endTime - startTime));
        methodLastArgs.put(methodName, request); // Logging arguments
        return result;
    }
    public void printProfile(){
        System.out.println("Profiling Data:");
        for (Map.Entry<String,Long> entry: methodCallCounts.entrySet()){
            String methodName = entry.getKey();
            long count = entry.getValue();
            long totalTime = methodTotalTime.get(methodName);
            String lastArgs = methodLastArgs.get(methodName);
            System.out.println(methodName + ": Count=" + count + ", Total Execution Time=" + (totalTime/1000000.0) + "ms, last arg=" + lastArgs);
        }
    }
}
```
The primary change is the addition of the `methodLastArgs` map. This map stores the latest argument that was used when calling processRequest. This is just one of many possible expansions for our proxy.

Furthermore, proxy classes can be used to profile code that does not originally support the creation of proxies. For example, if we wanted to profile a standard library class, such as `String`, this can be achieved using dynamic proxy classes. This involves a slight change in how the proxy is created as the proxy now does not directly implement an interface.

```java
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.HashMap;
import java.util.Map;
class ProfilingInvocationHandler implements InvocationHandler {
  private Object target;
  private final Map<String, Long> methodCallCounts = new HashMap<>();
  private final Map<String, Long> methodTotalTime = new HashMap<>();
  public ProfilingInvocationHandler(Object target){
     this.target = target;
  }
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
      long startTime = System.nanoTime();
      Object result = method.invoke(target, args);
      long endTime = System.nanoTime();

        String methodName = method.getName();
      methodCallCounts.put(methodName, methodCallCounts.getOrDefault(methodName, 0L) + 1);
      methodTotalTime.put(methodName, methodTotalTime.getOrDefault(methodName, 0L) + (endTime - startTime));
      return result;
    }
  public void printProfile(){
        System.out.println("Profiling Data:");
        for (Map.Entry<String,Long> entry: methodCallCounts.entrySet()){
            String methodName = entry.getKey();
            long count = entry.getValue();
            long totalTime = methodTotalTime.get(methodName);
            System.out.println(methodName + ": Count=" + count + ", Total Execution Time=" + (totalTime/1000000.0) + "ms");
        }
    }
}

```

Here, we use Java’s reflection API to generate a proxy implementing the same interfaces as the wrapped class. The `InvocationHandler` intercepts all method calls and adds the profiling logic. Usage of this dynamic proxy is different:

```java
public class Main {
    public static void main(String[] args) {
      String originalString = "Hello, world!";
      //Create a proxy for String class
        ProfilingInvocationHandler handler = new ProfilingInvocationHandler(originalString);
        String proxyString = (String)Proxy.newProxyInstance(String.class.getClassLoader(),
                originalString.getClass().getInterfaces(),handler);

        proxyString.charAt(0); //Trigger method call
        proxyString.substring(5); //Trigger method call

        handler.printProfile();
    }
}
```

In this example, we have generated a dynamic proxy that intercepts methods of the String class. The `InvocationHandler` allows us to create proxies for classes that don’t explicitly implement interfaces, demonstrating the adaptability of this approach.

For further reading on this topic, I suggest researching “Design Patterns: Elements of Reusable Object-Oriented Software” by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides; this book provides a detailed explanation of the proxy pattern and its various uses. Furthermore, books on aspect-oriented programming delve deeper into interception strategies. Exploring literature about Java reflection will also prove beneficial when building dynamic proxies. Finally, study material covering common profiling techniques can provide valuable background information.
