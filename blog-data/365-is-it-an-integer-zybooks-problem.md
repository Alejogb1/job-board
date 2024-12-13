---
title: "3.6.5 is it an integer zybooks problem?"
date: "2024-12-13"
id: "365-is-it-an-integer-zybooks-problem"
---

Alright let's break this down looks like someone's wrestling with a classic zybooks hiccup 365 is definitely a tricky one if you're approaching it with the wrong assumptions yeah I've been there man let me tell you about my own run-in with this exact problem

So picture this it's probably 2015 or 16 I'm banging my head against a similar problem in a intro to programming course not zybooks specifically but the same kind of integer check concept I remember thinking "this should be trivial" and then spending an embarrassingly long time debugging something stupid like this so you're not alone alright

The core of the issue as I see it is how zybooks often tests input values It's not always explicit about the format they're expecting they can give you inputs that *look* like integers but are under the hood stored as strings or even worse floats and then you try to do integer math on them and boom it goes haywire so the first thing you gotta realize is that `3.6.5` in the traditional sense of numerical representation is not a number it's a string or maybe some wonky hybrid data structure so yeah

When you get `3.6.5` in a zybooks problem it's not going to be recognized by your program as an actual number in the way you are expecting it will typically be treated as text even in languages like python which try to do type conversion it is still usually a string this is because of the multiple decimal places which is non standard for floating-point or integers It's not a floating point number because of the format it is not an integer because of the same reasons but what if you wanted to process something like this anyway ok then let's get down to it let me try to explain my mental process as if we were pairing

First you need to check what you're actually getting So start there use the debug tool if it's available or print the input values to your console before you begin any real operation and let's say I am doing this in python ok

```python
input_value = "3.6.5"
print(f"Type of input: {type(input_value)}")
print(f"The value is {input_value}")

```
Run this thing in zybooks or whereever you're coding if you are coding and it will tell you exactly what you're working with and as expected you are probably dealing with a string. Now the question at hand is an integer check and you have a string that is trying to be passed as a integer how do you approach this? it will require a bit of string manipulation before you can even think about integer operations so that is what we need to tackle.

Now let's think about what you need to do here if the goal is to see if something can be an integer you have to clean this mess and find a pattern for extracting integers from these values maybe you have to deal with inputs that are `1.2` or `5.0` but not necessarily `3.6.5` so you could just simply filter out the stuff after the decimal or you could extract all the numbers for later operations or you could test the initial text to see if it matches certain conditions all are valid if the problem you are trying to solve is that the text input is meant to be an integer input

Lets say you need to test each part so you will have to split the text into it's individual parts I will use python again because its pretty easy for quick tasks like this ok

```python
input_value = "3.6.5"
parts = input_value.split(".")
print(parts)
for part in parts:
    try:
        integer_value = int(part)
        print(f"{part} can be converted to an integer and its value is: {integer_value}")
    except ValueError:
         print(f"{part} is not an integer")
```
This will show you how each part of the string `3.6.5` can be handled and you can use this to process multiple inputs and create conditions based on what they are.
From here, you can build up some logic based on what your task actually is. If the task is to verify if the individual parts can be integers, then the code above is what you need but if the task is more nuanced and you need to verify if the input `3.6.5` as whole can be an integer, then what I did above is more on the verification of the parts and not the actual input itself. It really depends on what you want to verify

A lot of people in Zybooks courses run into trouble when dealing with string inputs and not understanding string processing is a common oversight I remember my first time dealing with this kind of issue I thought my computer was broken it wasn't its always the human in the end.

Now let's say you have some input like `1.2` and the objective is to test if it can be converted to an integer. A simple approach would be to test it before splitting it. Let's explore that.

```python
def is_convertible_to_int(input_str):
    try:
        int(float(input_str))
        return True
    except ValueError:
        return False

input_value = "1.2"
if is_convertible_to_int(input_value):
    print(f"{input_value} can be converted to an integer")
else:
     print(f"{input_value} can't be converted to an integer")
```

This function will first try to convert the text value to a floating-point and then to an integer this is a typical approach because some libraries can only parse strings to floating point and not to integer directly if that fails the input is not convertible to int and returns false. Now I know someone's gonna come here and complain about converting to float first well hey it works right and I have had this conversation 100 times with people over the years of me working in software development so I am pretty sure you will too.

Ok but let me get back to the core of the original problem we can confidently say that `3.6.5` is not going to be an integer on any traditional definition if the problem asks you to see if the input is an integer. It is a string and therefore not an integer. Now this looks simple and obvious to me now after working in the field for a while but I remember banging my head on these kinds of things back when I was first starting.

Now based on my experience working with Zybooks and similar coding platforms the trick is to carefully read what is being asked of you maybe they are asking about the individual parts as I mentioned earlier or maybe the conditions are less strict and you need to test if a certain condition is met. Zybooks is known for their specific input formats and it will be unlikely that you have to do complex numerical calculations if the input is `3.6.5` but you may have to do something with that string input and verify it meets certain conditions. If you had more details I could give you a more concrete solution.
Here's some of my go-to resources for stuff like this that have helped me become the amazing programmer I am today. I don't usually link to websites I hate that but I'll give you actual book titles and you can google them and find a PDF or buy them or whatever

*   **"Structure and Interpretation of Computer Programs"** by Harold Abelson and Gerald Jay Sussman this one's an oldie but a goodie it forces you to think about the fundamentals and really understand what's going on under the hood. It is a bit of a dense book but it will make you a better programmer
*   **"Code Complete"** by Steve McConnell yeah its a bit bulky but it talks about how to create better maintainable and readable software which is something I have found very important over the years if your code looks like a mess it probably will be so yeah this is a must-read if you are looking to be a better programmer.
*   **"Refactoring: Improving the Design of Existing Code"** by Martin Fowler this one helps me make cleaner code after I am done prototyping something so if you are like me and code first ask questions later this book is for you.

These books have guided me through similar types of issues before and helped my growth as a programmer I really think they can help you too if you are serious about programming.
Anyway hope this helped and good luck on your Zybooks journey oh and if you are running into integer problems maybe check your type annotations in python they are technically optional but sometimes a simple `:int` or `:float` can help you find errors early on. This is something I was doing wrong when I started I was so focused on the math and the logic I forgot that I needed to keep my variable types aligned or I would get unexpected issues so yeah keep that in mind.
