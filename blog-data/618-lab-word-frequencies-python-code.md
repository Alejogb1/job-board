---
title: "6.18 lab word frequencies python code?"
date: "2024-12-13"
id: "618-lab-word-frequencies-python-code"
---

Okay so you're tackling the classic word frequency problem with Python right I've been there believe me So 618 lab you say I guess that means you're in some sort of intro CS course Maybe its Data Structures or something anyway I know this problem inside and out it's a rite of passage for anyone getting into programming particularly with text analysis

Let's break it down I've seen countless variations of this over the years starting from my own early struggles with C++ before I even touched Python good times or rather not so good at the time

So the basic premise is simple you have some text and you need to count how many times each word appears Its sounds easy and it is but you can get into the weeds with it really fast

First off you will get some text or a path to the text file Let us get that sorted out first

```python
def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
       print(f"Error: An error occurred while reading the file: {e}")
       return None
```

This is pretty standard a function to read text from a file it includes error handling which is often forgotten when we are in a rush

I remember one time I was working on a project for a university class and I skipped this error handling It took me three hours to figure out why the program was crashing It turned out I misspelled the file name and the program was not able to read it The error was quite a cryptic one too Lesson learned always add error checks it saves so much time in the long run

Now that we have a way to read in the text the next step is where it all happens text processing I use this to ensure all text is in lowercase and only contains alphanumeric characters

```python
import re

def preprocess_text(text):
    if text is None:
      return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

```

I am using here the regular expression library to remove non-alphanumeric characters so things like punctuation marks exclamation points commas question marks are gone

Why all this effort? Well because "The" and "the" should be counted as the same word and without removing punctuations we could end up with different tokens because a word with a comma is not the same word without one

Now that is done we need to actually calculate how many time each word appears this can be done easily with a dictionary or a counter object which is even better

```python
from collections import Counter

def calculate_word_frequencies(text):
    if not text:
      return {}
    words = text.split()
    return Counter(words)
```

It first splits the text into individual words using spaces as delimiters and the Counter object does exactly what is expected it counts each word and creates a dictionary of keys (words) and values (frequencies)

I once had a project where I tried to do this without Counter and ended up writing my own loop to count and then spent so much time debugging because there was a missing edge case in my loop that created errors when words did not exist it was a nightmare Counter made my life way simpler

So now putting it all together in a function

```python

def word_frequency_analyzer(file_path):
    text = read_text_from_file(file_path)
    if text is None:
        return None
    processed_text = preprocess_text(text)
    word_frequencies = calculate_word_frequencies(processed_text)
    return word_frequencies

```

This ties everything together you input a file path the program reads the text process it and counts the word frequencies as a dictionary

I once forgot to put in the if text is None return None in the word frequency analyzer and it resulted in my program crashing on an empty text document and it was the most frustrating debug session I have ever done so do not make my mistake

Now on the technical side if you want to learn more about the theory behind all this I suggest reading "Speech and Language Processing" by Daniel Jurafsky and James H. Martin it is an excellent book for those looking to understand more the basics of natural language processing and it covers many things regarding the topic of tokenization and word analysis including all the edge cases you will want to know about

Also for more on algorithm analysis and to have a more solid understanding of the Big O analysis of why this works you should try the introduction to algorithms book by Cormen it is pretty good for understanding the complexity of the code I wrote above

Also a note to you if you are working with really large files you might need to think about using generators instead of loading the whole file into memory it is a good exercise in optimizing your code but for this exercise that code should work fine unless you are trying to analyze the whole internet of text documents at once

Also about that joke I promised why did the programmer quit his job because he did not get arrays hehe

Anyways back to coding you now have a working solution for your word frequency problem. It is not the most fancy code out there but it gets the job done. I hope this helped you get a better understanding of the process and maybe you learned something about my own coding mistakes so do not make the same ones I did. Remember error checking and always reading the official documentation it is there for a reason
