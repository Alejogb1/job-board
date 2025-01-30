---
title: "How can two threads in Java print odd and even numbers?"
date: "2025-01-30"
id: "how-can-two-threads-in-java-print-odd"
---
Java's inherent multithreading capabilities offer several mechanisms to coordinate the execution of distinct threads. The scenario of having two threads concurrently print odd and even numbers requires precise synchronization to avoid race conditions and ensure the desired output sequence. I’ve often encountered this challenge in simulations where interleaved event processing is crucial, and I've found that using a shared monitor with `wait()` and `notify()` is a reliable method.

The core principle here revolves around a shared object acting as a monitor, with threads taking turns based on a boolean condition representing whether the next number to be printed should be odd or even. Each thread, after printing its designated type of number, relinquishes control using `wait()` and signals the other thread using `notify()` to proceed. This interplay prevents both threads from concurrently attempting to print, thereby preserving the integrity of the sequence. Crucially, the shared monitor's lock is acquired and released each time, providing a crucial memory visibility guarantee for changes to the shared state. Without this, issues like stale data or race conditions would arise.

Let's illustrate this with three different implementation examples, progressing from a basic approach to a more robust, production-ready solution.

**Example 1: Basic Synchronization Using `synchronized`, `wait()`, and `notify()`**

This first example establishes the foundational logic using the core Java concurrency primitives. I’ve seen this employed in basic exercises, though real-world applications often require greater nuance.

```java
public class OddEvenPrinter {

    private boolean isOddTurn = true;
    private int counter = 1;

    public synchronized void printOdd() throws InterruptedException {
        while (!isOddTurn) {
            wait();
        }
        System.out.println("Odd: " + counter++);
        isOddTurn = false;
        notify();
    }

    public synchronized void printEven() throws InterruptedException {
        while (isOddTurn) {
            wait();
        }
        System.out.println("Even: " + counter++);
        isOddTurn = true;
        notify();
    }

    public static void main(String[] args) {
        OddEvenPrinter printer = new OddEvenPrinter();

        Thread oddThread = new Thread(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    printer.printOdd();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        Thread evenThread = new Thread(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    printer.printEven();
                }
            } catch (InterruptedException e) {
                 Thread.currentThread().interrupt();
            }
        });

        oddThread.start();
        evenThread.start();

        try {
            oddThread.join();
            evenThread.join();
        } catch (InterruptedException e){
          Thread.currentThread().interrupt();
        }
    }
}
```

In this code, `isOddTurn` controls which thread can print the next number. The `synchronized` keyword ensures mutual exclusion on the `printOdd` and `printEven` methods. The `wait()` method releases the monitor and suspends the thread until a notification is received. When a thread's turn arrives it prints and then signals the other thread using `notify()`.  The `main` method creates the threads, starts them and waits for them to complete with `join()`.

**Example 2: Using Explicit Locks and Conditions**

My second example utilizes `java.util.concurrent.locks.Lock` and `java.util.concurrent.locks.Condition`, offering more explicit control over locking and waiting.  I’ve favored these in more involved applications where resource management is paramount, requiring more fine-grained control.

```java
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class OddEvenPrinterLocks {

    private Lock lock = new ReentrantLock();
    private Condition oddCondition = lock.newCondition();
    private Condition evenCondition = lock.newCondition();
    private boolean isOddTurn = true;
    private int counter = 1;

    public void printOdd() throws InterruptedException {
        lock.lock();
        try {
            while (!isOddTurn) {
                oddCondition.await();
            }
            System.out.println("Odd: " + counter++);
            isOddTurn = false;
            evenCondition.signal();
        } finally {
            lock.unlock();
        }
    }

    public void printEven() throws InterruptedException {
        lock.lock();
        try {
            while (isOddTurn) {
               evenCondition.await();
            }
            System.out.println("Even: " + counter++);
            isOddTurn = true;
           oddCondition.signal();
        } finally {
           lock.unlock();
        }
    }

    public static void main(String[] args) {
         OddEvenPrinterLocks printer = new OddEvenPrinterLocks();

        Thread oddThread = new Thread(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    printer.printOdd();
                }
            } catch (InterruptedException e) {
                 Thread.currentThread().interrupt();
            }
        });

        Thread evenThread = new Thread(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    printer.printEven();
                }
            } catch (InterruptedException e) {
                 Thread.currentThread().interrupt();
            }
        });

        oddThread.start();
        evenThread.start();

        try {
            oddThread.join();
            evenThread.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

The introduction of `ReentrantLock` offers greater flexibility as it is not tied to the implicit monitor of a synchronized block. Each thread has its own `Condition` to `await` on until it's its turn, controlled by the `isOddTurn` variable. The `lock()` method acquires the lock, and `unlock()` releases it, often in a `finally` block to ensure the lock is always released. The `signal()` method notifies threads waiting on a specific condition, unlike the more general `notify()` method.

**Example 3: Utilizing a Blocking Queue**

This final example employs a `java.util.concurrent.ArrayBlockingQueue`, a construct designed for thread-safe communication. This method is often more adaptable in system architecture, as it is loosely coupled and message based. I've utilized this pattern in producer-consumer setups where data is communicated between disparate parts of an application.

```java
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class OddEvenPrinterQueue {

    private BlockingQueue<Integer> queue = new ArrayBlockingQueue<>(1);
    private int counter = 1;

    public void printOdd() throws InterruptedException {
        queue.put(1); // Signal odd can print
        System.out.println("Odd: " + counter++);
        queue.take();  // Wait for the other thread
    }

    public void printEven() throws InterruptedException {
      queue.take();  // Wait for its turn
        System.out.println("Even: " + counter++);
       queue.put(1); // Signal that odd can print
    }


    public static void main(String[] args) {
        OddEvenPrinterQueue printer = new OddEvenPrinterQueue();

        Thread oddThread = new Thread(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    printer.printOdd();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        Thread evenThread = new Thread(() -> {
            try {
                for (int i = 0; i < 5; i++) {
                    printer.printEven();
                }
            } catch (InterruptedException e) {
                 Thread.currentThread().interrupt();
            }
        });

        oddThread.start();
        evenThread.start();

        try {
            oddThread.join();
            evenThread.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}

```

Here, the `ArrayBlockingQueue` acts as a synchronization tool. The queue has a capacity of 1, forcing one thread to wait while the other proceeds. When the odd thread enters, it adds an item to the queue, signaling to proceed, prints its number, and then tries to take, which blocks it. The even thread, when it gets its turn, removes that item, prints its even number, and puts back an item, allowing the first thread to proceed. This approach is more loosely coupled as the threads do not directly access a shared state but communicate through the queue.

**Resource Recommendations**

For a deep understanding of Java concurrency, I recommend the Oracle Java documentation on concurrency.  Additionally, books such as "Java Concurrency in Practice" by Brian Goetz are invaluable resources, providing detailed information regarding different concurrency constructs and design patterns. Studying examples in the `java.util.concurrent` package can also give practical understanding of various utilities available. Furthermore, actively engaging with code examples and modifying them helps to grasp the subtle nuances of concurrency. Consistent exploration through coding is the best way to solidifying these concepts.
