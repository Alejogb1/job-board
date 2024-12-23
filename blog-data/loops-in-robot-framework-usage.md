---
title: "loops in robot framework usage?"
date: "2024-12-13"
id: "loops-in-robot-framework-usage"
---

 so loops in Robot Framework yeah I’ve been there done that more times than I care to admit It’s like the bread and butter of test automation once you get past the initial "hello world" scripts So lets talk loops in RF from my perspective

First off lets get one thing straight there's no one-size-fits-all way to loop in Robot Framework it's not like some languages where you have a single for loop and that's it In RF you are basically looking at variations on top of the ‘For’ keyword and how you manage data iteration through variables or lists that's where the real control lives

I've personally seen countless people struggle with this mostly because they try to make it more complex than it needs to be or they are coming from programming languages where loops are way more straightforward and the syntaxes are a bit different The trick here is to understand how robot handles variables which are strings in essence and how you can manipulate them inside a loop structure

I remember this one time I was working on this massive UI testing suite for an e-commerce site It was a nightmare we had to test product listings under different categories with various filters and price ranges and if there was no loop implementation I would probably still be writing the tests till now the initial tests were like 500 lines long I kid you not and half of that were doing the same thing with small variations We had to click category filter search type keywords check results a total nightmare to maintain

We were wasting so much time copy-pasting code so i said enough was enough it was time to clean this mess and implement proper loops We started with the basics a simple for loop to iterate through a list

```robotframework
*** Test Cases ***
Example Simple Loop
    @{categories}=   Create List  Electronics Books Clothing
    FOR    ${category}    IN    @{categories}
        Log    Category: ${category}
        # Here you'd put the testing logic for each category
        # For instance go to the page check some elements
    END
```
This is pretty basic right Just creating a list assigning it to the variable @{categories} and iterating through it with ‘FOR’ keyword in RF The output if you run it would just show all the categories being logged to the console this is where you would start building the logic that is supposed to go in the loop itself In my case it was browsing to product category pages doing all the usual checks confirming product list filtering worked etc etc

Now what if you are not dealing with a static list? What if you need to loop through a range of numbers or something like that? This is another frequent situation I had on another testing project a while back we had to generate test user accounts and we needed to do it iteratively based on a given number we ended up with something similar to this:

```robotframework
*** Test Cases ***
Example Range Loop
    ${number_of_users}=  Set Variable 5
    FOR    ${i}    IN RANGE   1  ${number_of_users}
        ${username}=    Catenate    SEPARATOR    user     ${i}
        Log   Creating user ${username}
        # here we'd actually create the user with an API call
        # or something like that
    END
```
See this uses ‘IN RANGE’ keyword that is a Robot Framework way to loop through a range of numbers note that it starts from 1 and goes until the number provided so from 1 to 5 in this case the variable ${i} is automatically incremented by 1 with each iteration inside the loop The important point here is to understand that ‘IN RANGE’ will generate a sequence from the provided start to the end number but the variable ${i} will increment by a static default increment value this might not be enough for more complex iteration logic so keep that in mind

Now lets talk about nested loops this one is very important and its a source of many errors for newcomers but also people like me I've made my fair share of mistakes with nested loops too I once messed up the nested loop structure and ended up with the test running for like 45 minutes before catching my mistake because it was running through every combination of categories and filters but not in the way it was expected it’s one of those moments I still remember and laugh a little bit you know debugging at 3 AM because I missed a small indentation is part of the game

Here’s a quick example of nested loops and i think that is the most practical use case when writing automation tests:

```robotframework
*** Test Cases ***
Example Nested Loop
    @{categories}=   Create List  Electronics Books Clothing
    @{filters}=  Create List  Price Date Relevance
    FOR    ${category}    IN    @{categories}
        Log   Category: ${category}
        FOR  ${filter}  IN  @{filters}
            Log     Filter: ${filter}
            # test logic for each category and filter combination
        END
    END
```

Simple nested loop right the outer loop goes through all categories and for each category the inner loop goes through all filter options and you can nest even more loops it is a matter of organizing your thought process and the problem at hand

Remember to be careful with nested loops in real world scenarios because they can quickly explode the test execution times especially when your lists are large so always optimize your test data and your test architecture

One crucial thing is not just the looping itself but also how you use those variables inside the loop Robot Framework treats variables in a special way remember that variables are just strings and how you construct strings inside loops matters a lot when you are generating values programmatically for your test cases You might need to use ‘Catenate’ keyword or other string manipulation keywords inside the loop to dynamically create file names or ids for instance and that is where you start seeing the real power of RF with test automation

For resources I’d recommend checking out the official Robot Framework user guide first its probably the best place to start then if you really want to dig deeper there are some good books on test automation with python and robot framework try searching for them online some authors cover the topic more in depth But honestly just playing around with different ways to loop and checking the documentation as you go is the best way to learn I’ve learned most of it by doing and banging my head against the wall and making my fair share of mistakes along the way so do not worry if you mess something up its part of the journey for every one of us

So yeah thats about it from my experience looping in Robot Framework is very important once you pass the basics its just about knowing the keywords and variables and how to use them to solve your actual testing problem There are no magical shortcuts its basically practice practice practice and more practice until it clicks
