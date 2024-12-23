---
title: "regex length of string validation?"
date: "2024-12-13"
id: "regex-length-of-string-validation"
---

 so regex string length validation right been there done that seen the t-shirt and probably written the library for it somewhere in my dusty hard drive of code I mean it sounds simple enough right but the devil is always in the details as they say I’ve had my share of hair pulling moments with this kind of thing back in the day when I was still learning the ropes of web dev in like 2008 so lets dive in

First off the problem is pretty straightforward you need to check if a string has a specific length or if it falls within a given length range using regular expressions and while regex isn’t always the first tool that comes to mind for length validations especially when you have direct access to the string length property it can still be helpful for more complex input validations or when you are stuck working in environments where you are constrained to using regex for input processing like I was once forced to do when working on some old legacy perl script that used regex for almost everything under the sun (and I mean everything)

The simplest case of all is checking for an exact string length in that case you would go for the following regex

```regex
^.{10}$
```

Simple right this regex anchors the pattern to the start and end of the string using the caret `^` and the dollar sign `$` symbols then it matches exactly ten characters using the dot `.` which means any character and the quantifier `{10}` if you want exactly 20 you would change 10 for 20 and so on so on so forth etc etc This means the string must have exactly ten characters to be a valid match not more not less

Now things get slightly more complicated when you want to validate a range of lengths you want a minimum and a maximum say at least 5 chars and at most 15 chars then you would need to use quantifiers with ranges like this

```regex
^.{5,15}$
```

This regex is also anchored at the start and end and matches any character but the quantifier here `{5,15}` allows for any string that has between 5 and 15 characters inclusive and that is what you want most of the time for basic validation purposes I actually had to fix a similar bug once where the backend was not checking for proper length and it was causing some weird database errors because a long string was getting truncated there and causing all sort of data inconsistencies issues which was not fun to debug at all because the logs were very not helpful to say the least and it took me almost a day to pinpoint the exact reason for all that mess. It was indeed a bad day

 so far so good you can check for an exact length or a length range but what about more complex scenarios what if you need to ensure that the string is of a specific length or falls within a length range but only contains certain characters well that is when you combine character classes and length quantifiers you want at most 10 characters but only a-z you would do something like this

```regex
^[a-z]{1,10}$
```

This will allow strings between 1 and 10 characters long and those characters must be lower-case letters from a to z I once had to do a regex like that to validate a user id in an old system I was maintaining for a small startup it was really not that complex but it was interesting how they tried to enforce a rule for user ids with so many constraints that almost no one remembered what the regex did at the end I had to spend almost an hour deciphering what that regex actually did and then rewrite it in a more maintainable way that I would not have to worry later about what it meant. I bet you had similar experiences right? We all have our stories I guess that is life in tech

Now for the more experienced of you I know what you are thinking are there any other things that I should consider when using regex for length validation in my apps. Yes there are, mostly edge cases you might encounter

One of them is that regex and unicode don’t always play well together sometimes it does and sometimes it does not It depends on the regex engine that you are using but sometimes it might not work as expected if you have unicode chars that are made out of multiple code points. So keep an eye for those or you will have headaches later trust me on this one because I had to deal with that kind of issues multiple times with different regex engines It is a pain I know and I have had to debug that for a good amount of time so be aware of the unicode traps. I mean the first time I saw this I had to go to the unicode consortium to better understand what the hell was happening and trust me it is not as simple as it might seem on the surface. So yeah be careful with unicode chars

Also keep in mind that for simple length checks using regex might be overkill in many cases It will generally be easier to just use the length method or property of the string class in whatever language that you are using most languages will have one and they are very easy to use so dont make it too complicated I mean sometimes it is good to not reinvent the wheel unless you really have to otherwise you would be wasting precious time to build something that is already built in every single language that you are going to use and no need to complicate things too much for something as simple as a length validation.

Another thing to keep in mind is that regex engines are not always the same. There are minor differences depending on the regex engine you are using which could lead to unexpected behavior so make sure you know what regex engine your language or framework is using to avoid surprises down the road I mean when you are under pressure and you have some deadline and your code is not working the last thing you want to debug is the regex engine difference. Seriously there are differences and they are a pain to debug you will thank me later. It is also a good practice to always test your regex expressions in a online tool before you use them in real production and that will save you a lot of trouble.

Also keep in mind about performance using complex regexes specially in a loop can be very slow in terms of performance so always try to use a simpler regex and if it is possible try to avoid the use of regex and rely on the language string length capabilities when they are enough to solve the problem I mean I have seen some nasty regexes that could kill a server if used incorrectly and you would not want to be the guy who wrote that code so always check the performance issues that your regex is doing with some profiling tools that every major framework and language has.

Speaking of which what resources can you use to study regex in depth well I can recommend “Mastering Regular Expressions” by Jeffrey Friedl its an amazing book that goes to the very roots of the regex topic and it is super useful but if you want to understand how regex is implemented in different regex engines I would suggest you look into papers related to Thompson NFA and DFA automata to understand how regex matching is actually done under the hood. It is super interesting stuff

And yeah I think that more or less covers it I think you have all the important stuff now with all of my past painful experiences included in the mix to get a better understanding of how this thing works. Let me know if you have other questions I will be glad to help if I can so yeah thats all folks have fun coding and remember the best regex is the one you dont have to write

Oh and one last thing why did the regex go to therapy because it had too many match issues haha sorry couldn't resist
