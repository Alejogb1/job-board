---
title: "expected expression before token?"
date: "2024-12-13"
id: "expected-expression-before-token"
---

Okay so you hit the dreaded "expected expression before token" error right I know that one way too well Been there done that got the t-shirt and probably a few scars too let me tell you a bit about my journey with this particular compiler grievance and how i've handled it in the past

First things first this error usually means the compiler saw something it wasn't expecting Like a variable where it wanted an operator or a semicolon when it expected a curly brace Stuff like that Its the kind of thing that makes you double check every single line of code several times and by the end you just feel a bit defeated but we will beat this together dont worry

Lets talk about my experience with this specific pain point way back when i was just cutting my teeth in C++ back in my early days in university i was working on a really ambitious project a custom ray tracer from scratch which i believed at the time was going to be really popular turns out it wasnt but anyway its irrelevant here one day i got this error staring back at me i remember spending hours looking at my code convinced that the problem was a bug in the compiler itself or that i had accidentally moved a semi-colon in my keyboard when that was not the case My code was a mess of custom classes for vectors lights and materials and every time I compiled I was met with this "expected expression before token" error I was frustrated you dont even know I felt defeated I started thinking of giving up It felt like the compiler was talking to me in riddles I remember i had this one line that was something like this

```cpp
 Vector3 light_direction =  - (light.position - hit_point) .normalize();
```

I spent a long time scratching my head at this trying to figure out why it was not working My problem was that I was expecting the normalize method to operate right after the subtraction operation but the compiler was not having it So it was seeing the dot as something it wasnt expecting it was expecting an operator instead and it was very mad at me as it seems to be at you too now

The compiler is very literal you see it doesnt understand your grand intentions it only understands the rules that were written for it by other very smart and very annoyed engineers so it was my responsibility to obey and not to expect him to understand my own language this is an important piece of advice my friend It took a few cups of lukewarm coffee and a lot of debugger stepping before i realized i had made a classic newbie mistake it turns out the compiler didn't understand that `.normalize()` was meant to be applied to the result of `(light.position - hit_point)` it was interpreting the dot as something else something incorrect. I had to be explicit i had to tell him precisely what I wanted. The fix was a simple one i just used parentheses to explicitly say what I wanted to do

```cpp
 Vector3 light_direction =  - ((light.position - hit_point) .normalize());
```

that did the trick and things worked fine after that but the lessons learned were very valuable

The key takeaway here is that sometimes the compiler just doesn't know what we're thinking and we have to be very precise and explicit when it comes to what we want to execute and how we want to execute it You cant just assume things will work you have to follow the rules of the compiler like if you were using a very old and grumpy calculator if you dont press the buttons in the order it expects then you get errors that dont make any sense

Now lets dig into a few more examples this error comes up in a lot of places and its good to have a few tricks in the bag for whenever this happens Here's a little gotcha that I've seen in other people's codes a lot. When you try to directly create a string or other kind of object that requires a constructor with some values without giving the constructor a name you get it again it's another variation of the same basic problem

```cpp
class Person {
public:
    string name;
    int age;
    Person(string personName, int personAge): name(personName), age(personAge) {};
};
// This will cause an error
Person john =  {"John Doe", 30};

```

The compiler here is going to be mad again expecting an expression before token. It is looking for a type of assignment it is looking to assign a value to a variable not to create a new object with its contructor

To fix it and make the compiler happy you have to explicitly call the constructor this looks like this

```cpp
 Person john = Person{"John Doe", 30};
```

See how it works now The constructor is called using its class name so that it is clear for the compiler that you want to initialize an object not perform an invalid operation

Now lets talk about function calls again sometimes this error can appear there too mainly if the function call is not properly constructed or if we make silly typos such as using commas instead of semicolons or viceversa When you want to invoke a function with some specific parameters you have to make sure that all parameters are given and are provided in the right order and with the correct types otherwise the compiler will not be able to understand your intention It will get mad at you because its job is to make sure that everything is correct before turning that code into an executable

Lets say for example you have a function like this

```cpp
 int add (int a , int b) {
    return a+b;
 }
```

And you want to use that function with specific values

this is going to generate the dreaded error again

```cpp
int result = add 5 3;
```

Because the syntax is wrong the compiler is going to complain about that again expecting an expression before token because it expects you to call the function in the correct way

The correct way would be

```cpp
int result = add(5, 3);
```

See the parethesis and comma now the compiler can understand what you mean and all is well in the universe

This specific error "expected expression before token" is a very broad one and it can pop up in many different contexts and situations depending on the language you are using It can be something as simple as a wrong operator or as complicated as a problem with your object declarations or your templating syntax or any other complex feature you can find in modern C++ code

So basically this error is a red flag that something is wrong in your syntax and the compiler is lost and doesn't know what to do with what you are trying to do So that is when we have to carefully check the syntax and see that everything is in order I have personally spent a lot of hours of my life trying to fix this error but its one of those lessons that makes you a better programmer when you finally understand it

Now about further reading if you want to dive deeper into compiler theory and understanding these types of errors I'd recommend you looking into these resources:

*   "Compilers Principles Techniques & Tools" also known as "The Dragon Book" This is a very standard textbook in the area of compilers its heavy and dense but it will give you a very very good understanding on how compilers actually work
*   "Modern Compiler Implementation in C" by Andrew Appel this book provides a more practical approach to compiler construction If you prefer hands-on learning this might be a very good option
*   The documentation for the specific compiler you are using I would recommend you to read the documentation of your compiler you would be surprised by how much useful information you can find in there This documentation is the final authority in your situation

And please be mindful about the fact that most of the time the compiler is correct and you are wrong. Okay this might sound a bit harsh so i will tone it down a bit. Most of the time the compiler is interpreting what you are writing exactly the way it was programmed to do so and it is you who is not writing things in the correct manner. It is not personal its just the way these things work.

So take a deep breath debug your code line by line if you need to make sure you have a very good understanding of the syntax and be meticulous when you write your code so you wont end up with this error again. Happy coding and I wish you good luck! And i wish you find the semicolon that made your life miserable.
