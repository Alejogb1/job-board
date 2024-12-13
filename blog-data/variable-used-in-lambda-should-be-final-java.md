---
title: "variable used in lambda should be final java?"
date: "2024-12-13"
id: "variable-used-in-lambda-should-be-final-java"
---

Okay so you're asking about why variables used inside a Java lambda need to be effectively final right I've been there seen that and debugged that a few times it’s a common gotcha for people new to lambdas or closures in general

Alright let's break this down like we're debugging a nasty NullPointerException at 3 am

First off "effectively final" isn't some weird made-up Java term it just means a variable's value isn't changed after it's initialized If it's never reassigned it's effectively final Java doesn’t force you to use the `final` keyword for this but it *behaves* like it's `final` inside the lambda

Think of it this way when you create a lambda you're essentially creating an anonymous inner class under the hood Java needs to capture the environment at the point where the lambda is defined If it allowed you to modify variables in that outer scope the lambda could operate on a value inconsistent with the rest of the program It would be a mess of race conditions and headache inducing concurrency bugs that would even make the most experienced dev question their coding life choices

And let me tell you I've seen some real doozies of bugs

One time I was working on a distributed task processing system this was back in my early days using lambdas I had a shared counter variable that was supposed to keep track of how many tasks were completed It was simple really just increment it inside the lambda processing each task and update some ui element based on that progress but the whole thing blew up spectacularly it was so bad it even crashed a few of the integration servers that were deployed I’d say a good 8 hours was spent on debugging this issue. The UI was updating with random numbers out of thin air some values not even in the right range like numbers into the millions when it should have been under 5000. I’d seen strange things in the past but this was unique. Turns out I wasn’t using a proper atomic variable and this happened because the lambda was capturing this counter variable but the variable was being incremented by other threads not captured by the lambda scope. It was my fault I will say that again and again it was my fault I’d never forget that day. After a deep dive and hours of searching I switched to using a AtomicInteger which then did the trick

To avoid issues like that Java decided to make variables inside a lambda effectively final This guarantees the value captured inside the lambda stays consistent with the value at the time the lambda was created This behavior is called lexical scoping and Java adopted this rule

Now why not just use the original variable directly inside the lambda Well remember that the lambda can execute much later possibly even on a different thread than where it was created If the original variable was still in the outer scope it could be modified by the main code while the lambda is still using it this is again a race condition you never want to be part of

Okay let’s look at some examples

**Example 1 The Bad Case**

Here’s what happens when you try to change a captured variable and make the lambda unhappy:

```java
public class LambdaCaptureBad {
    public static void main(String[] args) {
        int counter = 0;

        Runnable badRunnable = () -> {
            // This is a NO NO
            // counter++;  // Compile error: Variable used in lambda expression should be final or effectively final
             System.out.println("counter: " + counter);
         };

        // Some code here
        // counter = 10; // This change is the problem


        badRunnable.run();
    }
}
```

You can see that if you uncomment `counter++` or `counter = 10` the code will give you a compile time error because the counter variable is being modified that makes the lambda angry. This is not allowed.

**Example 2 How to Fix it Properly**

Here's how to get around the effectively final limitation and still change a "counter" when needed:

```java
import java.util.concurrent.atomic.AtomicInteger;

public class LambdaCaptureGood {
    public static void main(String[] args) {
        AtomicInteger counter = new AtomicInteger(0);

        Runnable goodRunnable = () -> {
           counter.incrementAndGet();
           System.out.println("counter: " + counter.get());
        };
        
        goodRunnable.run();
        goodRunnable.run();
       
    }
}
```

Notice we're using an `AtomicInteger` here This lets us modify the *value* inside the object which is captured by the lambda while the reference to the `AtomicInteger` itself remains effectively final. If you’re dealing with multi-threaded code this is the way to go

**Example 3 Local Variables within Lambda**

If you need to use and modify a variable within the lambda you can just declare it inside the lambda its own scope:

```java
public class LambdaLocalVar {
    public static void main(String[] args) {
        Runnable localVarRunnable = () -> {
           int localCounter = 0;
           localCounter++;
           System.out.println("Local counter: " + localCounter);
        };
        
        localVarRunnable.run();
        localVarRunnable.run();
    }
}
```

Here the localCounter variable is declared and used only within the lambda its scope each time the lambda is executed `localCounter` gets initialized to 0

Think of lambdas as like those old school camera dark rooms if the chemical solution is exposed to light it gets ruined and the pictures are useless Similarly a captured variable needs to be in a consistent state otherwise the lambda is as useless as a blank photo. (And yes I’ve also seen some weird things happen in the dark rooms also)

Resources to look at for understanding closures lexical scoping and how to handle multithreading in Java:

1.  **Java Concurrency in Practice** by Brian Goetz et al: This is a classic text about multithreading in Java it covers all you need to know about concurrency primitives like `AtomicInteger` and will help you avoid some of those annoying and dreadful concurrency bugs that seem impossible to fix
2.  **Effective Java** by Joshua Bloch: While not solely about lambdas this book has excellent guidance on writing idiomatic Java code and using lambdas properly and some really important information on creating functional interfaces
3.  **The Java Language Specification**: This is the official language specification and can be a bit dry but it's the ultimate source of truth for understanding how Java works if you want to go deep and understand it like the compiler would
4.  **Lambda Calculus**: While this is pure theory it’s interesting to read about the theory behind lambda expressions and programming language constructs if you're that way inclined It’s not necessary but can be interesting. It will help you see where a lot of this has come from it’s always nice to know the foundations of the buildings that we live in right?

So yeah effectively final variables in lambdas are annoying at first but once you understand the "why" behind it all and especially why its not a good idea to use mutable shared state it makes sense And prevents some weird bugs in your code

Hope this helps happy coding and remember to always clean your code and use meaningful variable names so your colleagues (or yourself in 3 months) won’t curse you too much I’ve seen developers who have coded the worst bugs and that’s nothing compared to developers that don’t name their variables properly.
