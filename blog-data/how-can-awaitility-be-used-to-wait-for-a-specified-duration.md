---
title: "How can Awaitility be used to wait for a specified duration?"
date: "2024-12-23"
id: "how-can-awaitility-be-used-to-wait-for-a-specified-duration"
---

Alright, let’s talk about `awaitility` and how to wrangle time-based conditions with it—something I've spent quite a few hours doing, trust me. It's more nuanced than just saying "wait for five seconds." We need to delve into how it manages that, and what makes it a more reliable alternative to, say, a simple `Thread.sleep()` in many concurrent scenarios.

First off, when we're speaking of using `awaitility` to wait for a specified duration, we're not *just* passively sleeping. Instead, we're often dealing with situations where asynchronous processes, message queues, or other operations are involved. We want to ensure that a certain condition becomes true within that time, or at least confirm it hasn't become true if we're expecting a failure scenario. A fixed duration of waiting isn’t always the best fit; rather, we should wait until something *happens*.

The fundamental concept in `awaitility` revolves around continuously evaluating a condition until it is met, or a timeout occurs. This is incredibly useful in integration tests or system tests where we need to observe a change of state driven by other parts of the application. Let’s get this clarified further with concrete examples.

**Understanding the `pollDelay` and `timeout`**

The two primary mechanisms that control the waiting are `pollDelay` and `timeout`. The `timeout` is the total duration you are willing to wait for a condition to be satisfied. The `pollDelay`, often referred to as `pollInterval`, dictates the frequency at which the condition is checked. It’s not like `awaitility` sleeps for the entire timeout then checks the result, nor does it check continuously in a busy-loop. Rather, it's this iterative process of checking, waiting for `pollDelay`, and repeating if necessary, until either the condition is true, or the timeout is reached.

Let's get to some code:

**Example 1: Waiting for a boolean flag**

Imagine a scenario where an asynchronous operation sets a flag once it completes.

```java
import org.awaitility.Awaitility;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import static org.awaitility.Awaitility.await;
import static org.hamcrest.Matchers.is;

public class Example1 {
    private AtomicBoolean operationComplete = new AtomicBoolean(false);

    public void performAsyncOperation() {
        new Thread(() -> {
            try {
                Thread.sleep(1000); // Simulate async work
                operationComplete.set(true);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }


    public void waitForCompletion() {
      performAsyncOperation(); // starts the asynchronous process
        await()
          .atMost(5, TimeUnit.SECONDS)
          .pollInterval(250, TimeUnit.MILLISECONDS)
          .untilAtomic(operationComplete, is(true));
       }

    public static void main(String[] args){
      Example1 example = new Example1();
      example.waitForCompletion();
      System.out.println("Operation completed successfully within the timeout.");
    }
}
```

In this first example, `performAsyncOperation()` simulates asynchronous work that sets `operationComplete` to `true`. `await()` establishes a timeout of 5 seconds. It will check `operationComplete.get()` every 250 milliseconds and if it’s not true, it waits the `pollDelay` period and then re-checks. If `operationComplete.get()` becomes true within the 5-second timeout, `await()` returns and the program proceeds. If the timeout is reached without the condition being satisfied, it throws a `ConditionTimeoutException`. We are using `is(true)` from the Hamcrest matcher library to provide a clear definition of the state we're waiting for. This is cleaner and easier to read than relying on equals comparisons.

**Example 2: Waiting for a collection size**

Here’s a more complex situation where we check a collection size:

```java
import org.awaitility.Awaitility;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import static org.awaitility.Awaitility.await;
import static org.hamcrest.Matchers.hasSize;

public class Example2 {
    private List<String> items = new ArrayList<>();

    public void addItem(String item) {
      new Thread(() -> {
          try{
             Thread.sleep(500);
             items.add(item);
           } catch(InterruptedException e) {
               Thread.currentThread().interrupt();
            }
        }).start();
    }

    public void waitForItemsToBeAdded() {
        addItem("item1");
        addItem("item2");
      await()
        .atMost(3, TimeUnit.SECONDS)
        .pollInterval(100, TimeUnit.MILLISECONDS)
        .until(items::size, hasSize(2));
    }
    
    public static void main(String[] args){
      Example2 example = new Example2();
      example.waitForItemsToBeAdded();
      System.out.println("List has reached the expected size.");
    }
}

```
This example showcases `awaitility` checking for a specific condition based on a collection size. We’re explicitly defining the expected state as a size of 2 using `hasSize(2)` again with Hamcrest matchers. Note here, I’ve deliberately included the method `items::size`, indicating a method reference being used as a supplier for awaiting. It will check the size of list `items` every 100 milliseconds for three seconds or until the size is 2.

**Example 3: Waiting using a custom condition**

Finally, suppose we have a more complex conditional check; then `awaitility` supports that too, as shown below.

```java
import org.awaitility.Awaitility;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

import static org.awaitility.Awaitility.await;

public class Example3 {
  private Integer randomValue = 0;
    
    public void generateValue() {
      new Thread(() -> {
           try {
               Thread.sleep(500);
               randomValue = new Random().nextInt(100);
           }
          catch (InterruptedException e){
             Thread.currentThread().interrupt();
          }
       }).start();
    }

    public void waitForCondition() {
        generateValue();
       await()
           .atMost(2, TimeUnit.SECONDS)
           .pollInterval(100, TimeUnit.MILLISECONDS)
           .until(customConditionSupplier());
    }
   
    private Supplier<Boolean> customConditionSupplier() {
       return () -> randomValue > 50;
    }
    
   public static void main(String[] args){
        Example3 example = new Example3();
        example.waitForCondition();
      System.out.println("The custom condition was satisfied.");
   }
}
```

Here, we are using a lambda expression that returns a boolean to define our custom condition within the `customConditionSupplier` method. We’re polling `randomValue` repeatedly, checking if it’s greater than 50. We are not using any Hamcrest matcher in this instance. This shows how versatile `awaitility` is; it isn't limited to just standard checks.

**Beyond the Basics**

These are some fundamental ways `awaitility` can be used. The actual implementation details, for example how exactly it polls the condition, aren’t essential knowledge to use the library; but they are interesting. You will find that it's not as simple as calling `Thread.sleep()` repeatedly, because under the hood, it’s using concepts such as `ScheduledExecutorService` with appropriate configuration to execute your condition checks.

**Recommendations for Further Learning:**

To truly understand the details and how to use them effectively, I’d suggest the following:

*   **"Java Concurrency in Practice" by Brian Goetz:** While not specifically about `awaitility`, this book provides an essential deep dive into concurrency concepts that are necessary for understanding asynchronous programming, which `awaitility` so elegantly helps with testing.
*  **"Effective Java" by Joshua Bloch:** It's a general Java programming book, but it contains sections that are immensely helpful regarding using immutability and concurrency in Java.
*   **`Awaitility`'s official documentation:** The official documentation has a wealth of information, examples and insights that is essential for getting the best out of the library. This can be found on the project's github page.

`Awaitility` does much more, including custom exception messages and much more sophisticated polling strategies. These three examples should at least get you started in how to approach the problem of checking conditions based on time. It is not just about the time it takes; rather, it is about waiting for the right state to be achieved. As always, focus on clarity in your tests, and `awaitility` is a key tool for that.
