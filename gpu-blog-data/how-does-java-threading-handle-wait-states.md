---
title: "How does Java threading handle wait states?"
date: "2025-01-30"
id: "how-does-java-threading-handle-wait-states"
---
Java threading employs a sophisticated mechanism for managing threads that are temporarily blocked, often referred to as "wait states," to conserve resources and enable efficient inter-thread communication. These wait states aren't just passive pauses; they involve a deliberate relinquishing of the thread's execution context, and require a specific signal for resumption. Understanding this mechanism is crucial for writing robust and responsive multithreaded applications.

I've spent considerable time debugging race conditions and deadlocks in large-scale Java systems, often tracing these issues back to improper handling of wait states, so I can confidently address this. A thread enters a wait state typically by invoking the `wait()` method, which is part of the `java.lang.Object` class. This method must be called from within a `synchronized` block (or method), guaranteeing that the thread executing `wait()` holds the intrinsic lock (monitor) associated with the object on which the method is invoked. This requirement is paramount because the `wait()` call fundamentally relinquishes the lock. If this synchronization guarantee isn't met, an `IllegalMonitorStateException` is thrown.

Upon calling `wait()`, the current thread does the following atomically: it releases the object's monitor and then adds itself to the wait set associated with that object. Crucially, the thread is now dormant, consuming minimal CPU cycles. It remains suspended until it receives a notification signal. This notification is delivered by another thread calling either the `notify()` or `notifyAll()` method on the same object. The `notify()` method will wake up one arbitrary thread that is waiting on the object's monitor, while `notifyAll()` wakes up all threads in the wait set.

The precise order in which threads are awakened is not guaranteed. The Java Virtual Machine (JVM) is responsible for thread scheduling, and it is implementation-specific. Furthermore, after a thread is awakened, it does not immediately regain the lock it had previously released. Instead, it enters the *entry set*, essentially a queue of threads waiting to acquire the lock. Once the lock becomes available, the awakened thread will contend for it like any other thread and proceed to execution after successfully acquiring the lock. This two-step transition from wait set to entry set, followed by the subsequent contention for the monitor, is important to understand. Also, upon reacquiring the lock, the thread must then re-evaluate its conditions (the original reason it entered the wait state) to prevent spurious wakeups, which are possible due to some implementation details. This often results in using a `while` loop to recheck the condition after a thread has been notified.

Here are some code examples to demonstrate these principles:

**Example 1: Producer-Consumer with Single Consumer**

```java
public class SingleProducerConsumer {

    private final Object lock = new Object();
    private int data = 0;
    private boolean dataAvailable = false;

    public void produce(int newData) throws InterruptedException {
        synchronized (lock) {
            while (dataAvailable) {
                lock.wait(); // wait if data already exists
            }
            data = newData;
            dataAvailable = true;
            System.out.println("Produced: " + data);
            lock.notify(); // signal consumer that new data is available
        }
    }

    public int consume() throws InterruptedException {
        int result;
        synchronized (lock) {
            while (!dataAvailable) {
                lock.wait(); // wait if no data is available
            }
            result = data;
            dataAvailable = false;
            System.out.println("Consumed: " + result);
            lock.notify(); // signal producer that consumption is complete
        }
        return result;
    }
}
```

*Commentary*: This code presents a simple Producer-Consumer pattern. The `produce()` method checks if data already exists; if so, it waits. After producing, it notifies the consumer. Conversely, the `consume()` method waits if no data is available and notifies the producer after consuming. The use of a `while` loop handles the potential spurious wakeups where a thread might wake up without having received a `notify()` signal. The `synchronized` block using `lock` object ensures that only one thread can access the shared resources at any given time.

**Example 2: Multiple Producers and Consumers**

```java
import java.util.LinkedList;
import java.util.Queue;

public class MultipleProducerConsumer {
    private final Object lock = new Object();
    private final Queue<Integer> buffer = new LinkedList<>();
    private final int maxSize;

    public MultipleProducerConsumer(int maxSize) {
        this.maxSize = maxSize;
    }

    public void produce(int value) throws InterruptedException {
        synchronized (lock) {
            while (buffer.size() == maxSize) {
                lock.wait(); // Wait if the buffer is full
            }
            buffer.add(value);
            System.out.println("Produced: " + value + ", Size: " + buffer.size());
            lock.notifyAll(); // Notify all waiting threads
        }
    }

    public int consume() throws InterruptedException {
        int value;
        synchronized (lock) {
            while (buffer.isEmpty()) {
                lock.wait();  // wait if the buffer is empty
            }
            value = buffer.remove();
            System.out.println("Consumed: " + value + ", Size: " + buffer.size());
            lock.notifyAll(); // Notify all waiting threads
        }
        return value;
    }

}
```

*Commentary*:  This example showcases the use of a bounded buffer using a `LinkedList`, with multiple producers and consumers operating concurrently. The `produce()` method adds a value to the buffer if it's not full and notifies all waiting threads. Similarly, `consume()` removes an item from the buffer if it is not empty and notifies all waiting threads. Here `notifyAll()` is used instead of `notify()` to avoid deadlocks which is more likely to happen with multiple producers and consumers. Using notifyAll allows any waiting producer or consumer to wake up in order to continue its process. The shared state buffer is protected by the monitor of the shared lock object.

**Example 3: Condition Variables and Explicit Locking**

```java
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class ConditionVariableExample {

    private final Lock lock = new ReentrantLock();
    private final Condition condition = lock.newCondition();
    private boolean dataReady = false;
    private int data = 0;

    public void produce(int newData) throws InterruptedException {
        lock.lock();
        try {
            while(dataReady) {
                condition.await(); // wait until dataReady is false
            }
            data = newData;
            dataReady = true;
            System.out.println("Produced: " + data);
            condition.signal(); // notify a waiting consumer
        } finally {
            lock.unlock();
        }
    }

    public int consume() throws InterruptedException{
        int result;
        lock.lock();
        try {
             while(!dataReady){
                 condition.await();  // wait until dataReady is true
             }
             result = data;
             dataReady = false;
             System.out.println("Consumed: " + result);
             condition.signal(); // Notify a waiting producer
        } finally {
            lock.unlock();
        }
        return result;
    }
}
```

*Commentary*:  This example uses explicit `Lock` and `Condition` objects from the `java.util.concurrent.locks` package.  The `Condition` object allows for more control over waiting and notification compared to intrinsic monitors. `condition.await()` is analogous to `wait()` and `condition.signal()` is similar to `notify()`. Notice that `lock.lock()` must be explicitly called, and `lock.unlock()` is called within the finally block, ensuring the lock is released regardless of exceptions. The `Condition` object, associated with the `lock`, provides named wait sets, enabling better control and readability compared to the implicit monitor of an object. It permits separating threads by conditions of interest.

For further exploration, I recommend researching the `java.util.concurrent` package, particularly the classes within `java.util.concurrent.locks`, including `ReentrantLock`, `ReadWriteLock`, and `Condition`. Studying the implementations of concurrent data structures such as `BlockingQueue`, `ConcurrentHashMap`, and the various executors can be beneficial. Specifically, delving deeper into the concepts of thread pools, fork/join frameworks and understanding the nuances of happens-before relationships can enhance comprehension of threading in Java. Textbooks focusing on concurrent programming in Java, often containing detailed explanations and examples, provide a structured approach to learning and are generally better than online blogs. Understanding memory barriers in relation to synchronized blocks will also deepen your knowledge of how threading works at the JVM level.
