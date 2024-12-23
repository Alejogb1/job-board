---
title: "5 digit random number generation program?"
date: "2024-12-13"
id: "5-digit-random-number-generation-program"
---

so you want to generate a 5 digit random number yeah I’ve been down that road more times than I can count honestly it’s surprisingly not as straightforward as some beginners might think I've seen plenty of newbies fumble around with this thinking `random.nextInt(99999)` will cut it but oh boy that gets you into some weird stuff mostly numbers that are not quite five digits it can be a real pain in the rear

Let me tell you about this one time back in my early days I was working on a data entry system and it needed unique identifiers for each record I was under the gun to get it out and I was just using java's `java.util.Random` thingy thinking it would be fine. I just whipped up some code something along the lines of this

```java
import java.util.Random;

public class BadRandom {
    public static void main(String[] args) {
        Random random = new Random();
        for (int i = 0; i < 10; i++) {
             int randomNumber = random.nextInt(99999);
             System.out.println(randomNumber);
        }
    }
}

```

And it kind of worked for a while until I started seeing those pesky numbers that started with 0's I needed to make sure every number I got was a true 5-digit number ranging from 10000 to 99999 inclusive and that `random.nextInt(99999)` just wasn’t cutting it for what I was trying to do it was all over the place getting 4 digits 3 digits even sometimes just 2 or 1 digits which wasn't very useful when i needed 5 it messed up my data structure in ways that I don't care to revisit right now I even had a bug report from the user saying there was a problem with my IDs because they were way too short some of them even just one digit which was very embarassing and also a good lesson on double checking everything that's for sure

The naive solution that most newbies try is what I showed you but it has its problems because `random.nextInt(n)` returns a random number between `0` and `n-1` so if you do `random.nextInt(99999)` it can return a number between 0 and 99998.

So my next try I remember this like it was yesterday I said to myself  I know what's going on here I need to set some boundaries so I started looking around I knew I needed to specify a lower bound and a higher bound I did some quick thinking and I thought let’s just go ahead and use `random.nextInt(max - min) + min` so for 5 digit numbers that would mean I’d be using `random.nextInt(99999 - 10000) + 10000` at the time I thought this would get me what I needed. I think I ended up making something like this

```java
import java.util.Random;

public class ImprovedRandom {
    public static void main(String[] args) {
        Random random = new Random();
        for (int i = 0; i < 10; i++) {
           int randomNumber = random.nextInt(99999 - 10000 + 1) + 10000;
           System.out.println(randomNumber);
        }
    }
}
```
This worked pretty well but it's still not ideal.

The proper way to do this in java is to avoid dealing with these range calculations yourself and instead just make sure you get a 5 digit number each time. We use a simple loop and the random function with a upper bound instead and make sure that it is greater or equal to our desired lower bound. If the number generated does not meet that criteria we simply re-generate.

This is the best way of doing it that will generate good random numbers and you know for sure you get 5 digits every time without having to think much.

```java
import java.util.Random;

public class CorrectRandom {
    public static void main(String[] args) {
        Random random = new Random();
        for (int i = 0; i < 10; i++) {
            int randomNumber;
            do {
                randomNumber = random.nextInt(100000); // Generate a random number up to 99999
            } while (randomNumber < 10000);

            System.out.println(randomNumber);
        }
    }
}

```

This works because `random.nextInt(100000)` will give you a random integer between 0 and 99999 inclusive. We then use a do-while loop to check if it's less than 10000 if it is we regenerate otherwise we are done and we now have a 5 digit number. This ensures you get the desired range every single time which is exactly what I needed in my earlier project.

If you want to understand more about random number generation I'd strongly recommend "The Art of Computer Programming Volume 2 Seminumerical Algorithms" by Donald Knuth it's a classic for a reason it goes deep into the theory and practicalities of pseudo random number generation. It’s a bit dense but it will teach you everything you need to know and some more. There is also "Numerical Recipes: The Art of Scientific Computing" by William Press but its more of a hands on book but it also has a good chapter on random number generation.

Oh before I forget about this it reminds me of one of my friends who when he had to deal with similar problems he said "random numbers are like snowflakes no two are exactly alike but they are all pretty cool in their own way" cheesy right but still accurate.

This simple technique using do-while loops will give you five digit numbers consistently without too much fuss and will avoid all the problems and errors I had back then when I first started trying to solve this kind of problems if you need to scale this up to more digits you just adjust the upper bounds to the power of 10 of your desired digit length. If you have any more questions I'm happy to help you out.
