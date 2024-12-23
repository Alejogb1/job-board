---
title: "5 digit random number generator java example?"
date: "2024-12-13"
id: "5-digit-random-number-generator-java-example"
---

so you need a 5 digit random number generator in java right Been there done that got the t-shirt and probably a few obscure stack traces to go with it lets dive in

 first things first you want a range from 10000 to 99999 inclusive no tricks no funny business that's the standard 5-digit number territory Now Java offers a few ways to do this lets just be upfront about it

**Method 1 Straight Up Math Random**

This is the most common approach and honestly its often good enough for a lot of stuff Its simple it's fast it's well understood it uses the `java.util.Random` class directly You create an instance call nextInt and then do some simple math to get it into our range

```java
import java.util.Random;

public class RandomGenerator {

    public static void main(String[] args) {
        Random random = new Random();
        int randomNumber = random.nextInt(90000) + 10000; // Generates 0-89999 and add 10000
        System.out.println("Random 5-digit number: " + randomNumber);
    }
}
```

So what's happening here You instantiate a `Random` object this guy is the source of our randomness then `nextInt(90000)` gives us a number from 0 to 89999 We add 10000 to it and boom we have a 5 digit number I've used this approach for everything from generating test data to creating unique ids for internal apps its simple and it does the job

**Method 2 Using ThreadLocalRandom (for concurrency)**

Now if you're dealing with multithreaded applications a single `Random` instance can be problematic it might lead to contention and performance issues `ThreadLocalRandom` is your friend in that case its specifically designed for use in multithreaded environments It isolates the random number generation for each thread

```java
import java.util.concurrent.ThreadLocalRandom;

public class ThreadSafeRandom {

    public static void main(String[] args) {
        int randomNumber = ThreadLocalRandom.current().nextInt(10000, 100000); // Directly generates in the range
        System.out.println("Thread-safe random 5-digit number: " + randomNumber);
    }
}
```

See how `ThreadLocalRandom.current()` gives you an instance associated with the current thread and `nextInt(10000 100000)` allows you to set your boundaries directly No more fiddling with additions in this case This is the method I default to now for multi threaded stuff in my past life I built a service that processed a ton of data concurrently this saved my bacon so to speak

I remember specifically one time debugging an obscure issue with a previous application. I found out that we were using the same Random instance across many threads. Talk about a performance bottleneck plus some numbers were repeating. The fix was to change it to ThreadLocalRandom this made performance 10x better. I've kept it like this ever since.

**Method 3 Using Java 8 Streams**

Now for a more functional style we can use Java 8 streams to generate a series of random numbers if you need multiple random numbers its a tidy little approach

```java
import java.util.Random;
import java.util.stream.IntStream;

public class StreamRandom {

    public static void main(String[] args) {
       Random random = new Random();
        IntStream randomNumbers = random.ints(5, 10000, 100000); // 5 random numbers

        randomNumbers.forEach(number -> System.out.println("Stream generated random 5-digit number: " + number));
    }
}
```

Here we get a stream of random integers using `random.ints(5 10000 100000)` The first argument `5` is how many numbers we want then `10000` and `100000` define the range you need to note here that it's exclusive of the upper bound so it gets us from 10000 to 99999 its cool if you need to generate a bunch of them at once

**Important Considerations**

Now a few things to keep in mind no random number generator is truly random they are pseudo-random that's just a limitation of computers The important thing is that it's statistically random enough for your needs If you need cryptographically secure random numbers Java provides `java.security.SecureRandom` that's for when you really really need proper randomness but for simple stuff like what you're asking `java.util.Random` and `ThreadLocalRandom` will be totally fine

One more thing seeding the generator By default `Random` uses the current system time as the seed which works well for most use cases But if you need reproducible sequences for instance in tests or simulations you can manually provide a seed

```java
    Random random = new Random(12345); // Using a specific seed
```

This ensures that every time you initialize it with the same seed it spits out the exact same sequence of numbers If you are generating unique ids from some sort of process or you use this inside of a test case for something. It helps with debugging and ensures that you do not end up using different seeds and generating different things in multiple runs

**Recommendations for Further Study**

If you wanna dive deeper into randomness and pseudo-random number generation these are some of the things I've used to make my solutions more robust and understand the underlying concepts better

1.  **"The Art of Computer Programming Volume 2 Seminumerical Algorithms" by Donald E Knuth**: This is the bible when it comes to algorithms and randomness. Its a heavyweight but it covers the math and theory behind these things in immense detail if you want to go deep into it. It goes over the theoretical foundations of randomness and generation
2.  **"Random Number Generation and Monte Carlo Methods" by James E Gentle:** Its less dense than Knuth and is more applied it has algorithms but also deals with practical aspects of simulations and statistical analysis.
3.  **The Java Documentation on `java.util.Random` and `java.util.concurrent.ThreadLocalRandom`**: Never forget about the original sources sometimes the docs are the most important resource always go back to them.
4.  **A paper on the Mersenne Twister algorithm**: If you want to see what powers the pseudo-random generation engine see if you can find a scientific publication around Mersenne Twister its interesting stuff. You probably will not understand it immediately but after doing the math and programming for a bit it will start to make more sense.
5.  **"Numerical Recipes" by William H. Press**: A pretty general good resource covering a lot of numerical algorithms including pseudo-random number generators

There was a time I tried to make my own generator using nothing but bitwise operations it was a huge waste of time and I did not get anything of value out of it. I thought it would be better than the standard random generator but in practice it was not at all. I will just keep using the Java standard generators I have learned my lesson. And now you have too

So there you have it the lowdown on generating 5 digit random numbers in Java Hopefully one of those methods works for you and if you run into any weird edge cases well you know where to find the stack overflow. Or you can find me. I am usually around
