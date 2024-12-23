---
title: "code check in code review process?"
date: "2024-12-13"
id: "code-check-in-code-review-process"
---

so code check in code review process right I've been down that rabbit hole more times than I care to remember Let me break it down for you from my perspective someone who's seen it all or at least a good chunk of it in the trenches

First thing's first code review isn't just some formality to tick off on a checklist It's a crucial part of the software development lifecycle that keeps things from turning into a total dumpster fire I'm talking from experience here I remember this one project back in 2010 yeah it was that long ago where we just pushed code straight to main without any reviews at all The result was pure chaos Bugs were everywhere the codebase was a tangled mess it was like trying to untangle a ball of yarn that a cat played with for a week We ended up spending more time fixing things than building new features It was a mess I tell you a total mess that I do not ever want to experience ever again that's a lesson learned the hard way

So yeah code reviews are definitely not optional In my opinion they're a vital necessity think of it as a sort of a quality control check it's a way to make sure that what you're putting out is up to standard that's readable that's maintainable that's actually working properly It's also an opportunity for knowledge sharing someone else might catch a subtle mistake that you missed or they might have a better way of doing things it's just plain good for the whole team's knowledge

Now when it comes to the actual process there are many ways to skin a cat or I mean many ways to approach code review personally I've found a few things that tend to work really well for me and for the teams that I've been a part of the first thing is keeping code reviews small and focused instead of cramming a week's worth of work into one massive pull request or merge request I've learned to break it down into smaller more manageable chunks That's easier on everyone really reviewers can digest the changes more effectively and its quicker to turnaround less context switching all around. I mean who likes a monstrous review anyways. You need to think of small commits and clear descriptions. This makes the review process much more streamlined.

Another key aspect is setting clear expectations before you even start the review for this what I've always done is using check lists and shared documents so that everyone is on the same page. We define coding standards and guidelines right in the team and then make sure that everyone agrees on the common ground. This avoids the endless back and forth about coding styles or nitpicking about whitespace or tabs versus spaces I mean seriously. It is very useful if the teams use some sort of linters and static analysis tools they just detect basic issues without even having a human look at them that is a huge benefit if you think about it.

As for what I look for when I am reviewing code I look for readability first is the code easy to understand can someone else pick it up in six months and figure out what it's doing If not its time to start doing some refactoring naming conventions are huge here please don't use meaningless names like x y or z It might seem obvious but you'd be surprised at how often this crops up. So meaningful variable names that are explicit. I also look for potential bugs or errors have you considered all the edge cases is there some exception that you have forgotten to handle I look for code duplication if code is written twice or three times we should be refactoring it into a common function or a class it's not only saving you time but also maintainability is easier. Also have you used the libraries or frameworks correctly is there any dependency issues or incorrect imports it's all part of the whole process. Last but not least tests! I always like to see unit test and maybe integration tests if needed coverage is a plus It shows that you have thought about how the code is supposed to work. And how to maintain it in the future.

Now let me show you some examples of code and review process that I had in the past that are good and bad examples:

```python
# Example of a bad review (lack of context and detail)
# Reviewer: "This code is bad needs refactoring"
# Author: "Why?"
# Reviewer: "Just does"

def calculate_area(length, width):
    return length * width
```

This is an example that happens way too often someone just saying it is bad or not good without any context this means that you will just waste time back and forth with the person that did the code it is a waste of time of the whole team and it shows that this reviewer is not even paying attention to the code

A more productive approach would be like this:
```python
#Example of a good code review
#Reviewer:"Nice work on calculating the area although I think it should handle invalid values like negative ones or non number values how about we add some handling for that"
def calculate_area(length, width):
    if not isinstance(length,(int,float)) or not isinstance(width,(int,float)):
         raise ValueError("Length and width should be numbers")
    if length < 0 or width < 0:
         raise ValueError("Length and width cannot be negative")
    return length * width
```
In this example reviewer is providing specific feedback that is actionable by the developer this way we actually have an outcome from the review that will improve the code base. Also the reviewer is not only saying that it is wrong but also explaining how the code should handle the incorrect cases

One more example let's say we are dealing with databases
```python
# Example of bad code without proper sanitization
def get_user_by_id(user_id):
    query=f"SELECT * FROM users where id={user_id}"
    cursor.execute(query)
    result=cursor.fetchone()
    return result
```
This is a huge no no This code is very vulnerable to SQL injection attacks what if a malicious user passes an id that contains SQL code? The database would be compromised it's a recipe for disaster. We always need to sanitize the inputs we take from the user

Here is how you should be doing this :

```python
# Example of good code with sanitized inputs
def get_user_by_id(user_id):
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query,(user_id,))
    result= cursor.fetchone()
    return result
```
Here you can see we're using parameterization to prevent SQL injection this approach keeps your code safe and also makes it more readable.

Now regarding resources if you want to go deep into this topic there are few books that I recommend. "Code Complete" by Steve McConnell is a classic that covers just about everything in software development including code reviews. It's a bit old but still very relevant. "Refactoring" by Martin Fowler is a must read if you want to learn how to improve your code quality. And if you're interested in the actual coding standards and stuff you should check out the official documents for each language or framework. For example the PEP8 for python if you use python that is or Google's Java style guide if that's your cup of tea. Or the coding standards for C++. Its best to always stick to the standards that are already existing to not cause even more inconsistencies.

Oh one time I was reviewing this massive pull request and the author decided to name all the variables after characters from their favorite show it took me an hour just to figure out what the code was doing It was a bit like trying to understand what my cat is trying to tell me when it meows at 3am. It was a long long night let me tell you but I never did that again the person was properly advised that naming stuff like that is not very productive. The moral of the story? Always aim for clarity always.

In conclusion code reviews are not just some annoying hurdle they are very important aspect of software development they improve quality they teach you new things and make the team work better they prevent bugs and overall they help you produce better products.
Remember to keep reviews small to define coding standards that are clear to you and everyone and you have to give detailed feedback that is actionable
