---
title: "2.17 lab driving cost methods zybooks coding?"
date: "2024-12-13"
id: "217-lab-driving-cost-methods-zybooks-coding"
---

 so you're asking about the Zybooks lab specifically 2.17 driving cost methods probably in a Java context if I had to guess And yeah I've definitely been down that road before This kind of problem comes up a lot in intro CS stuff and it's really about breaking down a task into smaller more manageable functions

Let's get this straight right away We're talking about calculating driving costs based on distance fuel efficiency and fuel price The core issue here isn't rocket science its about organizing your code and making it readable and reusable which is the goal in the long term I remember back in college a similar assignment actually caused a mini-meltdown I spent hours debugging one giant main method that was just a pile of variables and calculations it was a mess and I vowed never to repeat that mistake again the real learning started after that trainwreck

So first things first lets talk about what we need We need a way to get the distance the fuel efficiency and the price per gallon We also need a way to calculate the cost so let's just make a method for that It's straightforward stuff really but the design is what matters more in the long run A method keeps that logic isolated instead of stuffing it everywhere also makes code more testable easier to debug and we know we love that

Here's some pseudocode to set the stage and guide us towards what we want:

```
function calculateCost(distance milesPerGallon pricePerGallon)
  gallonsNeeded = distance / milesPerGallon
  totalCost = gallonsNeeded * pricePerGallon
  return totalCost
```

Simple right That's the core of it Now let's actually see some code in Java since it seems that's what we are talking about here.

```java
public class DrivingCost {

    public static double calculateCost(double distance, double milesPerGallon, double pricePerGallon) {
        double gallonsNeeded = distance / milesPerGallon;
        double totalCost = gallonsNeeded * pricePerGallon;
        return totalCost;
    }

    public static void main(String[] args) {
        double distance = 100;
        double mpg = 25;
        double price = 3;
        double cost = calculateCost(distance, mpg, price);
        System.out.println("The cost for this trip is: " + cost);
        // Test some other numbers here if you like.
    }
}
```

that works but Zybooks probably wants you to handle the input stuff on your own in a different manner than what the main method shows above Here is another slightly modified version that uses some inputs with Scanner object from user (or from Zybooks input) just for showing how we can make the user input it for the program:

```java
import java.util.Scanner;

public class DrivingCost {

    public static double calculateCost(double distance, double milesPerGallon, double pricePerGallon) {
        double gallonsNeeded = distance / milesPerGallon;
        double totalCost = gallonsNeeded * pricePerGallon;
        return totalCost;
    }

    public static void main(String[] args) {
         Scanner input = new Scanner(System.in);
         System.out.print("Enter the distance: ");
         double distance = input.nextDouble();
         System.out.print("Enter the miles per gallon: ");
         double mpg = input.nextDouble();
         System.out.print("Enter the price per gallon: ");
         double price = input.nextDouble();

         double cost = calculateCost(distance, mpg, price);
         System.out.println("The cost for this trip is: " + cost);
         input.close();
    }
}
```

See how we broke the process down We have a `calculateCost` method that does the math then the `main` method that sets up the variables and calls that method and prints the result Also I added a `Scanner` input process which is usually asked by Zybooks

Now about that method design thing I mentioned Earlier some of you might be tempted to just cram all this inside main well the short term it might work but you would be setting up some bad habits as you continue to code so I highly encourage you to stick to making the methods reusable and doing one main task only so we can reuse that logic somewhere else if needed in other places in our program or even other projects and this kind of thinking is critical in the long run it’s a good programming practice.

Let's talk about error handling or maybe input validation you need to be a little careful with division by zero or negative numbers. A miles per gallon value less or equal to zero doesn't make any sense, or price per gallon etc So adding some checks will help you with that Here is one way of doing it

```java
import java.util.Scanner;

public class DrivingCost {

    public static double calculateCost(double distance, double milesPerGallon, double pricePerGallon) {
       if (milesPerGallon <= 0) {
            System.out.println("Invalid miles per gallon. Please make sure it is bigger than 0");
           return -1;
        }
       if(distance <=0){
           System.out.println("Invalid distance. Please make sure it is bigger than 0");
           return -1;
       }
        if(pricePerGallon<=0){
           System.out.println("Invalid price per gallon. Please make sure it is bigger than 0");
           return -1;
       }
        double gallonsNeeded = distance / milesPerGallon;
        double totalCost = gallonsNeeded * pricePerGallon;
        return totalCost;
    }


    public static void main(String[] args) {
         Scanner input = new Scanner(System.in);
         System.out.print("Enter the distance: ");
         double distance = input.nextDouble();
         System.out.print("Enter the miles per gallon: ");
         double mpg = input.nextDouble();
         System.out.print("Enter the price per gallon: ");
         double price = input.nextDouble();

         double cost = calculateCost(distance, mpg, price);
        if(cost!=-1) {
             System.out.println("The cost for this trip is: " + cost);
        }
        input.close();
    }
}
```

I added a few if checks to see if we get some correct values this makes your code more robust to unexpected inputs Now the method returns -1 which is a bit clunky and you can make your own error code for that but just for simplicity this return -1 will make the program not calculate the total cost and instead notify the user that there was a problem

Also you could throw some exceptions too that might be useful as a next level up

As for resources If you really wanna improve your foundational knowledge check out "Structure and Interpretation of Computer Programs" it's a bit old school but it's a gold mine for learning how to think about problems and design solutions Also "Code Complete" by Steve McConnell is great for learning the practices of creating robust and maintainable code for Java specifics a book like "Effective Java" by Joshua Bloch goes really deep into good coding practices

And yeah I hope this helped Now you can go back to Zybooks or whatever and conquer this programming task Good luck and remember that the struggle is part of the learning process so embrace it (and maybe use a debugger when you get stuck lol) Oh and also please don’t forget the semicolons in Java otherwise the Java compiler will have a hard time interpreting your code (and so will you probably) its the equivalent of forgetting to put the key in the car and wondering why it wont start ahaha ok I am done here cheers!
