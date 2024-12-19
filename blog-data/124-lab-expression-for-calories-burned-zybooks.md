---
title: "1.24 lab expression for calories burned zybooks?"
date: "2024-12-13"
id: "124-lab-expression-for-calories-burned-zybooks"
---

Okay so you're looking at that 1.24 lab in Zybooks the calories burned thing I get it been there done that a million times it's one of those deceptively simple exercises that can throw you for a loop if you're not paying attention to the details

Let's break this down it's all about understanding the formula and how to translate that into code really it’s not rocket science more like high school physics mashed with some basic programming

First off the problem is basically giving you the rate of calories burned during different activities They give you a bunch of rates per minute for walking running swimming etc and then you gotta do some math and get the total calories burned given a duration for each activity They kinda make it look more complicated than it is but if you structure it well its pretty straightforward

I remember when I first tackled this kind of thing back in university Oh man I was struggling it was some intro to programming course and I was like the guy who hadn’t even touched a keyboard before that class I tried to hard code everything with just giant if-else statements because for some reason that seemed like a good idea at the time Needless to say it didn’t work well it was a mess of spaghetti code that took about a solid day to unravel I even asked the TA for help but all he did was say "it’s a bit messy maybe try to break down the problem" I was not happy I was more frustrated than a compiler trying to interpret my poorly written code

Anyway I learned my lesson the hard way breaking down the problem is key This means you gotta separate the data the input from the calculations you're doing with them If you do this right the whole thing is a lot easier so lets get to it

Basically what we need to do is get the exercise activity the time spent and then multiply the time by the activity rate to get the number of calories burned and then sum it all up If you wanna get fancy we can use functions to abstract away some of the details make it look a little cleaner

Here’s a basic python implementation that should get you going It's pretty basic but does the job

```python
def calculate_calories(activity, time_minutes):
    if activity == "walking":
        rate = 3.0
    elif activity == "running":
        rate = 10.0
    elif activity == "swimming":
       rate = 8.0
    elif activity == "biking":
       rate = 6.0
    else:
        rate = 0.0  # Or handle invalid activity differently
    return rate * time_minutes

# Example usage
total_calories = 0
total_calories += calculate_calories("walking", 30)
total_calories += calculate_calories("running", 15)
total_calories += calculate_calories("swimming", 20)

print(f"Total calories burned: {total_calories}")

```

This is okay it's functional I guess but lets make it a little more professional right

The thing with these kind of problems is if you start to have too many of these activity if else statements it is not very readable So it is better to use a dictionary it's a data structure that lets you store the activities and their rates in a more readable way.

Here is how I would do it a bit better using a dictionary It's a bit more scalable than the big if else if else mess.

```python
activity_rates = {
    "walking": 3.0,
    "running": 10.0,
    "swimming": 8.0,
    "biking": 6.0
}

def calculate_calories_better(activity, time_minutes):
   rate = activity_rates.get(activity, 0.0) # Gets the value of activity or defaults to zero if not found
   return rate * time_minutes

# Example usage
total_calories_better = 0
total_calories_better += calculate_calories_better("walking", 30)
total_calories_better += calculate_calories_better("running", 15)
total_calories_better += calculate_calories_better("swimming", 20)

print(f"Total calories burned: {total_calories_better}")
```
See that's much better We have a separate place for all our activity rates and the code to calculate it is a lot easier to read now We can even add new activities easily by just adding them to the dictionary and not changing our function its all about keeping your code organized you know It's the key to not ending up like my uni code where you have spaghetti that takes hours to unravel.

If you really want to scale this up for a lot of activities you can even load those values from a file like a CSV or use a database but that's probably overkill for this lab problem but its just a good practice to know how to scale your programs and use data sources.

Now let’s take this a little bit further If you are serious about coding you will always see that writing your own functions is the bread and butter of programming So let us use a class to wrap it up it's always a good way to organize related data and functions.

```python
class CalorieCalculator:
    def __init__(self):
        self.activity_rates = {
            "walking": 3.0,
            "running": 10.0,
            "swimming": 8.0,
            "biking": 6.0
        }

    def calculate_calories(self, activity, time_minutes):
       rate = self.activity_rates.get(activity, 0.0)
       return rate * time_minutes

# Example usage
calculator = CalorieCalculator()
total_calories_class = 0
total_calories_class += calculator.calculate_calories("walking", 30)
total_calories_class += calculator.calculate_calories("running", 15)
total_calories_class += calculator.calculate_calories("swimming", 20)

print(f"Total calories burned: {total_calories_class}")
```
Alright now we're using classes to our advantage This is basically a blueprint of how to calculate our calories and we can reuse it if we wanted to calculate for many users and just different activities. Now you're probably thinking wait this is becoming too complicated but trust me when you get bigger projects it helps a lot especially if you want to expand and add more features later.

You know I once spent 48 hours debugging a problem and after all the struggle I realized I was passing an int as a string to my calculation function It taught me the importance of proper variable types and naming conventions I never again made that mistake and it was a good lesson to learn I should have probably used type checking then hahaha funny how you learn it the hard way right.

So that's it really If you stick to this approach it should be pretty straightforward For the resources I would recommend checking out "Clean Code" by Robert C Martin it is a classic and teaches you the importance of organizing your code Also "Structure and Interpretation of Computer Programs" by Abelson and Sussman is also a good one to learn about how the fundamentals of programming work they are both classics and I would not go anywhere else they have it all. And of course practice this is key to improving I cannot overstate this enough practice is key.

Good luck on your lab and I hope this helps I've been there done that and I know it can get frustrating but remember just breakdown the problem into smaller pieces take it slow and you'll be fine.
