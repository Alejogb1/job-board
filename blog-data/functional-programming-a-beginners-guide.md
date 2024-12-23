---
title: "Functional Programming: A Beginner's Guide"
date: "2024-11-16"
id: "functional-programming-a-beginners-guide"
---

 dude so we're gonna geek out about this video right it's all about this wild thing called 'functional programming'  it's like this whole different way of thinking about code  the video basically tries to show you why it’s awesome and how it can make your life easier even if you're just starting out seriously it's like learning to ride a unicycle it looks weird at first but then you're zooming around town feeling like a total boss

so the setup is they're trying to build this thing a simple calculator app right but they do it in two ways the normal imperative way which is how most people start and then the functional way which is like the ninja level stuff  it’s all about pure functions immutability and things that sound scary but are actually kinda rad once you get it

one of the first things i noticed was this dude's super excited face when he talks about avoiding 'side effects'  remember that? he practically vibrates man  side effects are like when your function does something sneaky like changing a global variable or writing to a file without you even realizing it  it's a messy situation like leaving your socks on the floor it creates chaos in your code  you end up debugging for hours trying to figure out why something's not working  it's a nightmare trust me i've been there

another visual cue was how they showed the code flow  it was like this super clean diagram with arrows showing data moving  it looked way more organized than the spaghetti code i usually end up with it’s like comparing a perfectly organized sock drawer to a pile of laundry that just exploded.  it really emphasized how functional programming keeps everything tidy and predictable  you can practically follow the logic with your eyes closed

and then there was this moment when they explain recursion it was illustrated with those cool branching diagrams you know the kind where it breaks down a problem into smaller and smaller pieces  it was genius!  it totally clicked for me that recursion is just a function calling itself until it gets to the base case it's like those russian nesting dolls but with code

 so key idea number one pure functions this is the big one pure functions are like the zen masters of the code world they only depend on their inputs and they always produce the same output for the same input  no sneaky side effects no surprises  it's like a vending machine you put in your money and you get your snack no questions asked

here’s a simple example in python

```python
def add_numbers(x, y):
    return x + y

result = add_numbers(5, 3) # result will always be 8
print(result)
```

this is a pure function the output only depends on the inputs  it doesn't modify any external variables or do anything unexpected  it’s a total chill dude  compare this to something like:

```python
global_counter = 0

def increment_and_print():
  global global_counter
  global_counter +=1
  print(global_counter)

increment_and_print() #prints 1
increment_and_print() #prints 2
```

this is not a pure function because it relies on and changes a variable outside of its scope. it’s chaotic and unpredictable  you wouldn't want to build a rocket using this type of function

the second key idea is immutability basically it means you don't change things directly  you create new versions instead  imagine you have a list of names  instead of adding a new name to the existing list you create a whole new list with the new name added it’s like taking a photo of your sock drawer before and after cleaning it  you can see the change without messing up the original

```python
names = ["Alice", "Bob"]
new_names = names + ["Charlie"] #creates a new list
print(names) # Output: ["Alice", "Bob"]
print(new_names) # Output: ["Alice", "Bob", "Charlie"]
```

see how `names` remains unchanged? that's immutability in action  it's like having a superpower you can make changes without destroying the original data it’s amazing for avoiding bugs

the video also briefly touched on higher-order functions functions that take other functions as arguments or return them  it's like functions having functions as pets  this lets you do really cool stuff like map filter and reduce  it's like a toolbox full of mini-functions that let you manipulate your data in awesome ways. it’s like building with lego bricks but with functions

```python
numbers = [1, 2, 3, 4, 5]

#using map to double the numbers
doubled_numbers = list(map(lambda x: x * 2, numbers))
print(doubled_numbers) # Output: [2, 4, 6, 8, 10]

#using filter to get only even numbers
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers) # Output: [2, 4]
```

lambda functions are just tiny anonymous functions they’re super handy  in this example, they are used within map and filter to specify what operations should be done on each element of the list

the resolution? the video's main takeaway is that functional programming can make your code more readable maintainable and less prone to errors  it's not a magic bullet  it's a different way of thinking about problem-solving  it’s like learning a new martial art  it might be challenging at first but the benefits totally outweigh the initial learning curve it's a powerful tool once you get the hang of it so give it a shot you might just become a coding ninja yourself


i hope this helped man  let me know if you have any questions  we could grab some pizza and dig into this deeper some time  functional programming is a pretty deep rabbit hole but it’s a fun one  trust me.  now if you'll excuse me  i have some sock drawers to organize
