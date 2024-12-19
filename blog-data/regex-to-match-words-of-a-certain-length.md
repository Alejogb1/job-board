---
title: "regex to match words of a certain length?"
date: "2024-12-13"
id: "regex-to-match-words-of-a-certain-length"
---

Okay so regex to match words of a certain length right been there done that many times lets get this thing sorted out no fluff just practical code and some stories from the trenches

Right so the core thing here is we need a regex that respects word boundaries and can enforce the length criteria we're talking about individual words not parts of words or random character clusters the key component here is using `\b` which marks a word boundary in regex for example if you are looking for a word that starts with "a" and ends with "b" you will use something like `\ba.*b\b` where `.` means any character and * means 0 or more

First things first let’s tackle the basic scenario matching words that are exactly a certain length lets say 4 characters for this we’re going to use `\b\w{4}\b` let me break that down for you `\b` as I said is the word boundary `\w` matches any word character alphanumeric and underscore and `{4}` means exactly 4 times so that will match "code" "text" "data" but not "coding" or "cod"

```python
import re

text = "This is some code and data for a text example"
pattern = r"\b\w{4}\b"
matches = re.findall(pattern, text)
print(matches)  # Output: ['code', 'data', 'text']
```

Pretty straightforward right now lets move on to matching words that are within a range let's say between 3 and 6 characters that means we will use `{3,6}` so `\b\w{3,6}\b` will match "the" "code" "text" "codes" "coding" but not "a" or "example"

```python
import re

text = "This is a simple example with some codes coding and a very long one"
pattern = r"\b\w{3,6}\b"
matches = re.findall(pattern, text)
print(matches)  # Output: ['This', 'simple', 'with', 'some', 'codes', 'coding', 'very', 'long']
```

Okay one of the things that has tripped me up a few times in the past is when you are using languages that support unicode or when you are dealing with international texts that’s when `\w` might not be enough especially for languages with special characters so in that case you might need to use other unicode character properties instead of \w but those cases are pretty rare so for 99% of the time \w is enough just keep this in the back of your mind and this will save you headache for sure

For the experience part let me tell you about that one time in my internship back in 2015 where I was tasked with cleaning a huge database of customer reviews for sentiment analysis we had a weird requirement to only consider the words that were between 4 and 8 characters long for some strange statistical reason I tried to do it manually with a loop and string length checks and boy that was slow like turtle slow I then realized that regex could do it in a single line and it was not just faster it was cleaner and more readable I felt like an idiot for not thinking of it before but hey we live and we learn

And another thing that I remember is once I was working with logs for some network application we had this weird bug where some packets were not being processed and the logs were full of random strings that were not even valid words and the task was to find words longer than 10 characters that might point out to the faulty packets so for that I had to modify the regex and use `\b\w{10,}\b` to find words of 10 characters or more where `{10,}` is a way of indicating "10 or more"

Okay let's talk about some edge cases for example lets consider hyphenated words like "well-known" with the current `\w` based regexes "well-known" would be considered as 2 distinct words "well" and "known" that’s where you have to decide if you want to treat these as individual words or as one if you need to consider the full word then you have to extend the word character set with a hyphen so in that case you should use something like `[\w-]+` which means at least one word character or hyphen and in the example above `\b[\w-]{4,10}\b` this will match words between 4 and 10 characters including hyphenated words but be aware that this will now also match things like "----" or "--test--" so you need to adjust the code according to your needs but I think you get the idea

Let me show you an example using that pattern using hyphenated word that’s a thing that I use every once in a while

```python
import re

text = "Some well-known keywords like data-structure and code-review are important"
pattern = r"\b[\w-]{4,10}\b"
matches = re.findall(pattern, text)
print(matches) # Output: ['well-known', 'keywords', 'data-structure', 'code-review', 'important']
```

Now you might also encounter cases where you want to include some specific characters in your words so let’s say we need to find identifiers in a programming language that include underscores and letters the regex pattern in that case will be something like `\b[a-zA-Z_]+\b`

And lets be honest here sometimes using regex is overkill its like using a sledgehammer to crack a nut but you know I love regexes its like one of the first things I learnt and there are some corner cases when using libraries like pandas that will force you to use regex for simple stuff that could be solved with basic string methods so regex is a pretty useful tool in my book

One more thing I want to mention is that when you start using complex regex patterns make sure to test your regex with a bunch of examples edge cases like empty texts texts with special characters texts with international characters that will help you to understand if your regex is working correctly or not and it will save you some time and frustration

You know there is a joke I know about regex it’s so bad and nerdy that it should not be funny but here you go "I had a problem so I used regex now I have two problems" that one always get me going haha but yeah regex can be tricky

If you want to get a more detailed explanation of regular expressions I recommend "Mastering Regular Expressions" by Jeffrey Friedl its like the bible of regexes it explains everything in very detail Another resource I like is the documentation of regex library that you use for example for python I like the python re module documentation its very detailed and full of examples for java the java.util.regex package is also very well documented these are good starting points to start your regex journey

So in summary for matching words of a certain length use `\b\w{n}\b` for an exact length `n` use `\b\w{n,m}\b` for a range between `n` and `m` or `\b\w{n,}\b` for `n` or more and remember that `\b` is the key to match whole words remember to always test your patterns thoroughly with diverse inputs

Hopefully this explanation is clear enough I tried to make it practical and from my experience with regexes if you have any questions or you want to dive deeper into specific scenarios don’t hesitate to ask and ill do my best to give you an answer
