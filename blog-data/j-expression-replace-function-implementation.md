---
title: "j expression replace function implementation?"
date: "2024-12-13"
id: "j-expression-replace-function-implementation"
---

 so j expression replace function implementation right yeah I've been there a few times this is a classic problem and honestly anyone who's dipped their toes in the j programming language for more than a weekend has probably had to wrestle with this one I remember when I first started using j I was trying to do some data wrangling thing involving text files it was a mess I had this huge block of text and I needed to swap out all instances of one particular string for another and it felt like i was in the wild west at first but i got it going so here’s how i usually approach it

so lets get the core concept its not about replacing directly its about finding and building new string thats how it works in j it’s not like some python string replace it uses more of a functional approach we’re going to use the power of j's array processing to locate the occurrences and piece things back together using a mix of indexing and concatenation

Let’s start with a simple case replacing one single word in a string with another

```j
replaceSingle=: 3 : 0
    text =. x
    old =. y {::0
    new =. y {::1
    pos =. (old = text) i. 1
    if. 0 = pos do.
        text
    else.
        (pos {. text) , new , (pos + #old) }. text
    end.
)
```

 what we have here  is a simple verb `replaceSingle` it takes two arguments the string to work on x and the pair `(old new)` as argument y
first we get the string we going to modify and assign it to `text` then we take the `old` and the `new` string from the `y` argument the `{::` part is just indexing the `y` argument then we find the position of the old string in text using `i. 1` which finds the first occurrence of `old` inside text which will return the position of the substring where it starts or if it does not exist it will give us 0 which is perfect for checking for its existence

after that we check with an if statement if the position `pos` is 0 if it is we return the original text because it means the string we wanted to replace does not even exist in text
if the position is not 0 that means that we do have our `old` string in `text` so we construct a new string we select characters from the beginning of the text till our position `pos` we use `{. text` to do that then we append our `new` string and after that we append the rest of the string from text that means from `pos + #old` where # is the length of `old` to the end of the string text by using the `}. text` part at the end
This verb works if you only want to replace the first instance of a word but what if you want to replace all of them well lets go to our second example

```j
replaceAll=: 3 : 0
    text =. x
    old =. y {::0
    new =. y {::1
    positions =. (old = text) i. 0
    if. 0 = #positions do.
        text
    else.
        result =. ''
        i=. 0
        while. i < #positions do.
           start =. positions { i
            result =. result , (start {. text) , new
            text =. (start + #old) }. text
            positions =. positions - (start + #old) # 0
            i =. i + 1
        end.
        result , text
    end.
)
```

 this one is a bit more meaty in this verb we also get `text` and `old` and `new` in the same way as the previous verb `replaceSingle` but the difference starts here we are going to search for all the occurrences of `old` in the string `text` and return the indices in the `positions` varible for example if text is  `abcabcabc` and old is `abc` then the positions will be `0 3 6`  and we check to see if there are any occurrences at all if there is no occurrence we return the original text

If there are positions it gets more interesting and complicated we start building our result in result variable first and initialize a counter at 0 we start looping through every position we get from positions in the start variable we then add the part of the text from the beginning to the position start to the result variable and add our `new` string now the interesting thing we replace the current text variable by the part of text starting after the `old` string and we update our positions variable by subtracting the starting position and the length of the old string form all the other positions this makes our loop go through all the occurrences
now that we have a good grasp of the core functionality lets spice things up with a real problem
Lets say we have a string containing several placeholder that we need to replace
```j
replacePlaceholders=: 3 : 0
    text =. x
    placeholders =. y
    result =. text
    for_placeholder. placeholders do.
        result =. result replaceAll placeholder
    end.
    result
)
```

This one is much simpler we have our `text` that is to be modified and a 2 dimension array `placeholders` that contains pairs of `old` `new` strings for example `(‘name’ ‘john’) , (‘lastname’ ‘doe’)`  we then loop over all the `placeholders` we modify the text using `replaceAll` function we just implemented using the current element of the placeholders which is a pair of old and new strings

And now the joke this reminds me of a time when i was debugging some code with someone and we had this massive string manipulation problem and he looks at me and says is j even real and I just had to laugh at that point

Now for some real talk lets say you want to get really good with this kind of stuff for more advanced string stuff and not use these verbs I recomend these resources I used them myself to get a grip with how J works and it can help you with your specific problem:

First up “J for C programmers” its an oldie but a goldie the guy is a genius he explains the core concepts with such detail you will never forget the concepts or the syntax of the language
then there is "Concrete Abstractions An Introduction to J" it’s more of a modern book very useful to learn some of the more functional aspects of j it dives deep into the ideas of functional programming which is what j is all about
then finally I also found this paper that talks about array oriented approach to string manipulation it was a academic paper but it had a profound impact on how I write my code
These should give you solid foundation

Anyways that's how I do it hope it helps remember J is a different beast than your average imperative language and that is why you gotta approach it in a more functional kind of way but once you wrap your head around it string manipulation becomes a breeze with the powerful primitives it provides
