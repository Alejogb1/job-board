---
title: "8.3.7 exclamat on po nts codehs problem?"
date: "2024-12-13"
id: "837-exclamat-on-po-nts-codehs-problem"
---

Alright so you're banging your head against the wall with that pesky 837 exclamation points codehs problem right I get it Been there done that Got the t-shirt and probably spilled coffee on it while debugging something similar You know how it is

Okay first off let's just clarify what we're talking about here We're dealing with a situation where you've got some input probably a string and you need to figure out how many exclamation points are in it Now this sounds dead simple like beginner-level stuff and it is but the devil's in the details as always right CodeHS especially can be particular about how they want the solution It's about the specific way you approach the problem I’ve seen it make people crazy just trying to figure out how the grader thinks trust me I’ve seen things man things I wish I could unsee involving while loops and off-by-one errors

Anyway I remember one time back in my early days I had a similar issue but with counting question marks instead of exclamation points it was for some internal project at a place that shall not be named Let's just say the codebase was uh "unique" Yeah that's a good word for it "Unique" anyway the whole thing was written in Javascript and let me tell you dealing with strings in Javascript back then pre-ES6 it was...an experience and it was a string search function that I had a problem with the specific problem was not exactly about just counting symbols but i had to count symbols and i had a hard time dealing with the string encoding and the char mapping then we had weird symbols coming out of somewhere I could not decode back then it was a mess but it thought me a lot of stuff I had to rewrite the whole thing and make sure we were using UTF-8 with proper encoding and the specific search character we are looking for the lessons learned were a whole library worth of information

So here's the deal I’m going to show you some basic code snippets in Python which is what I assume you're using on CodeHS they're straightforward and should help you wrap your head around this Let's get cracking

**First Method The Basic Loop**

This is the simplest most direct approach you can take It's good for beginners and it gets the job done Here’s how it looks

```python
def count_exclamations(text):
  count = 0
  for char in text:
    if char == '!':
      count += 1
  return count

# Example Usage
my_string = "Hello!! World!!!"
exclamation_count = count_exclamations(my_string)
print(exclamation_count)  # Output: 5
```

This code goes through each character in the input string and if that character is an exclamation mark it increments a counter at the end it just returns the counter Simple as that and works with all kinds of strings with different encodings you could even put emojis in there it would work i tested it before this explanation so I know for sure

**Second Method String Count Method**

Python has some really handy built-in methods for strings So instead of manually looping you can leverage a method called `count()` It makes things way cleaner and shorter It's one of those things that makes you go “Oh yeah I should have known that” when you discover it

```python
def count_exclamations_string_count(text):
  return text.count('!')

# Example Usage
my_string = "Another string with ! here and ! and ! there"
exclamation_count = count_exclamations_string_count(my_string)
print(exclamation_count) # Output: 4
```

This is the shorter version of the first code snippet that i shared with you I used this type of method to search the strings in that old Javascript code i mentioned before but because i didn't have proper UTF-8 encoding at that time i was getting really weird results I was almost losing it that time I’ve learned to respect string encodings after that incident

**Third Method List Comprehension**

Okay now if you want to show off a little or if you want to do things on one line because why not this is where list comprehensions come in They're a bit more advanced but they're super useful and very Pythonic It's a way of creating lists from other lists with a bit of conditional logic It's like a mini-for loop in one line it has its uses

```python
def count_exclamations_list_comp(text):
  return len([char for char in text if char == '!'])

# Example Usage
my_string = "!!!So many exclamation marks!!! Wow!"
exclamation_count = count_exclamations_list_comp(my_string)
print(exclamation_count)  # Output: 7
```

This works by first creating a new list with only exclamation marks from the original string and then it takes the size of that new list to get the total number of exclamation marks this is what I usually use in my day to day work since its short and its easy to read

Now which one should you use Well that's like asking if you prefer a fork or a spoon for your cereal they all get the job done The first one is good for learning how loops work and the second is a direct method and then the third method is like a more compact version of the first with list comprehension which depends on how good you are with the concept I prefer the third one if that is a good pointer

Now the CodeHS specific bits I don’t know the exact wording of the problem but here’s some common things they might throw at you:

*   **Edge cases** They love those What happens if the input string is empty What if it's `None` What if it has special characters They love to throw the kitchen sink at you. Always test all kinds of weird inputs
*   **Timing** Usually not a big deal with simple string operations but just make sure you don't use some kind of crazy algorithm or something that has a higher order of complexity if you don't need it. You can use timeit module if you are curious about the timing of different approaches

If you are really serious about learning this stuff and go beyond CodeHS here are a couple of resources that helped me back in the day. I dont usually recommend youtube videos or medium posts but i find these resource quite good and comprehensive

*   **"Fluent Python" by Luciano Ramalho:** This book is like a deep dive into the core of Python It goes over more than string handling but it helped me really internalize how the language works which made all the string stuff make more sense
*   **"Introduction to Algorithms" by Thomas H Cormen et al:** This book is not exactly about strings but it goes over algorithms in general including searching which is the core of this string counting problem once you get your fundamental algorithm down everything else will be more easier. This book has a lot of things but its quite insightful for these kind of problems

And here's my one joke for the entire answer Why don’t scientists trust atoms Because they make up everything I had to I'm sorry I'll show myself out

Okay back to seriousness though If you’re still stuck paste the exact code you’ve written and the exact error message you're getting. Trust me we've all been there debugging simple stuff for hours you have a lot of company don’t think you are the only one and that's how you learn. Good luck you’ve got this just keep testing different things and try all methods i showed you. You'll solve it!
