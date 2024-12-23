---
title: "what is the logical not in prolog?"
date: "2024-12-13"
id: "what-is-the-logical-not-in-prolog"
---

 so you want to know about the logical not in Prolog right I got you I've wrestled with this beast plenty of times let me break it down for you using my real life experiences no fluffy stuff

See Prolog is kinda different from your usual imperative languages think C Python Java the not is not a simple flip of a boolean it's more like a check for something not being provable if I recall I was debugging some fuzzy logic AI project back in the day and my code kept giving me weird results I traced it to how I was handling negations and it turned out I wasn't using prolog not correctly it was a real head scratcher at first

So in Prolog the logical not is implemented using the `\+` operator its prefix like not so `\+ some_predicate` means "it is not provable that `some_predicate` is true" it doesn’t mean that the predicate is false per se just that prolog can't establish it as true with the current facts and rules you've provided remember Prolog is all about inference if something cannot be inferred or proven it’s considered as not true

 let's get into it with the code examples first imagine you have a simple database of facts:

```prolog
likes(john, pizza).
likes(mary, sushi).
```

Now if you query `likes(john, pizza)` prolog would return true yeah obvious but what if you asked `likes(john, sushi)` prolog would return false because it cannot prove that john likes sushi see the absence of evidence of course doesn't equate to proof of the contrary that is another concept

Now let's see the not operator in action

```prolog
?- \+ likes(john, sushi).
true.

?- \+ likes(john, pizza).
false.
```

So `\+ likes(john, sushi)` is true because prolog can't prove john likes sushi it returns true not because john dislikes sushi but because it cannot be deduced from the knowledge base the other query is false because we already know john likes pizza.

Now here is a trickier case with variables

```prolog
?- \+ likes(john, Food).
false.
```

The result is false because Prolog finds at least one value for Food that makes `likes(john, Food)` true in this case it was pizza so not of true is false but here it is another example

```prolog
?- \+ likes(peter, Food).
true.
```

That is correct prolog return true because the fact `likes(peter, _)` is not in the knowledge base remember that a variable will make the entire statement succeed if it can be resolved or not when used with the `\+` operator

This can lead to some surprising behavior if you are used to imperative languages like the following:

```prolog
different_food(X,Y) :-
    likes(X, Food1),
    likes(Y,Food2),
    \+ Food1 = Food2.

?- different_food(john,mary).
true
```

Here we are defining a predicate that checks if two people like different foods this seems right the first person likes some food the second person likes another food and these foods are different this is correct because john likes pizza and mary likes sushi therefore `different_food(john, mary)` is true
But take a look at the following which is going to fail

```prolog
not_liking_same(X,Y) :-
    \+ (likes(X, Food), likes(Y,Food)).
```

this will not return what you expect if we try to call it `not_liking_same(john,mary)` it will return `false` even if they like different food this is because we are checking if there is some food that both like in this case it does not exist but not being able to find the same food is not equivalent to both liking different foods the operator `\+` in this context means that no food exists that they both like that is different

If that last one got your brain spinning a bit don't worry it got mine too the first time I messed with it I spent way too long debugging code and it turns out it was just a simple case of using not wrongly I felt like a total noob on that day but hey we learn by doing right and sometimes scratching our heads a bit

The key takeaway is that `\+` is negation as failure it means "I cannot prove this". This isn't the same as a logical not in traditional terms so you need to think in terms of Prolog's inference and proof rather than boolean true or false this is quite a big difference between the languages and can lead to some interesting results that we saw

The tricky cases are with unbound variables if a variable is unbound Prolog will try to unify it and the not operator will fail if the unification succeeds even if we think about it in logical terms like both the variables might not be unified in reality when we evaluate the `not` we have already bound it and the other variable to some values so it can find something that is true and that is why the result will return false

This tripped me up on so many of my initial prolog implementations one of them was implementing a solver for a puzzle and the other one was a constraint solver this lead me to discover that I needed to rethink how negation worked and it is not really about logical negation but negation based on the proof procedure

For deepening your knowledge on this stuff I wouldn’t recommend relying solely on online tutorials especially on this topic because some of them are either wrong or they are not precise and might mislead you check these instead I would strongly advise checking the classic "Programming in Prolog" by Clocksin and Mellish and if you want to go deeper maybe look into "The Art of Prolog" by Sterling and Shapiro these two will give you a good base on Prolog and this particular case so you don't need to stumble into the same issues I had they are very precise and well explained which will give you some solid foundation on the subject

And finally just a small note about this not operator it’s like that friend who’s always technically right but can sometimes give you unexpected results but with a clear understanding of how it works and how to use it it can become a valuable tool in your Prolog toolkit.

One thing I learned the hard way and I think it is an important point for you is to test your code often don't assume that things will work as you expect especially when you are using the not operator it can easily bite you in the rear so always test and use a debugger to check what is happening before you go crazy trying to figure out why your code is not doing what you want this tip was probably the one that has saved me the most debugging time of all the others i know this from experience like that time I debugged code for 5 hours just to find that I forgot a dot in prolog that was a fun one and i learned that prolog is unforgiving when it comes to syntax this kind of thing happens to everyone even me and I am not going to lie

So there you go a deep dive into the logical not in Prolog straight from my real world trials and tribulations hopefully this helps you avoid some headaches I had in the past if you need more help just ask I'm here to assist you with your prolog quests.
