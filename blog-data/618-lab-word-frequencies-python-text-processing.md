---
title: "6.18 lab word frequencies python text processing?"
date: "2024-12-13"
id: "618-lab-word-frequencies-python-text-processing"
---

Okay so word frequencies in Python right classic I've banged my head against this wall more times than I care to admit Seriously this is like a rite of passage for anyone getting into text processing and data analysis in Python I remember my early days probably around python 2.7 days or so back when you had to worry about unicode like it was a bomb waiting to go off I was working on a project involving parsing tons of legal documents think court transcripts and laws and the like My boss at the time a real stickler for clean data was breathing down my neck about getting accurate term frequencies He had some weird obsession with being able to spot certain phrases and keywords like "reasonable doubt" and I had to deliver like yesterday I made all the rookie mistakes you can imagine trying to brute force it with simple loops and regexes it was a total mess Performance was trash and I swear I had some race conditions hiding in there somewhere It made me want to quit and take up knitting seriously

The key is you need to think more like a machine than you do a human that's the secret for text processing you want to use libraries that are designed for this stuff Instead of reinventing the wheel let's get to the code.

First things first let's talk about the basics You’re going to need to do a few things no matter how fancy your approach gets You need to read in the text clean it up a bit and then count the words and then you will be set to do all sorts of stuff with those frequencies but before you get to that you should make sure that what you read is a string and not bytes or it will make your code break that was a tough one to debug back in the day

Here is some basic code that I hope can solve your immediate problem:

```python
import string
from collections import Counter

def get_word_frequencies_basic(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    word_counts = Counter(words)
    return word_counts

# Example usage
example_text = "This is a test. This is only a test!"
frequencies = get_word_frequencies_basic(example_text)
print(frequencies)
```

Okay so what's happening here First I am importing some modules that are useful we have string to help deal with punctuation and `Counter` is literally built for what we are trying to do. The function `get_word_frequencies_basic` takes our text it lowercases everything so we don't treat "The" and "the" as different words it cleans out the punctuation that's the next part. Then it splits the string into individual words. Finally the `Counter` will go through that list of words and create a dictionary like object with the frequencies. Easy right that should handle your basic needs but let's be honest that's the bare minimum

Now what if you need to deal with big files not just a couple of sentences What if you have a massive text file of say a million lines or more then the first approach is going to become incredibly slow for those cases You will have to start thinking about how memory works and optimize this thing to become fast and efficient for large files that's where generators become your friend let's see an example of that

```python
import string
from collections import Counter

def word_generator(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.lower()
            line = "".join([char for char in line if char not in string.punctuation])
            for word in line.split():
                yield word

def get_word_frequencies_generator(file_path):
    word_counts = Counter(word_generator(file_path))
    return word_counts


# Example usage
file_path = 'large_text.txt'  # Replace with your large text file
# To test without a big file just create a text file called large_text.txt and write some text
# into it
frequencies = get_word_frequencies_generator(file_path)
print(frequencies.most_common(10)) # Show top 10
```

Okay this one is different here I introduce `word_generator` and the `yield` keyword that's how we create generators this guy will read the text line by line not all at once it cleans it up as it goes and yields each word one at a time the main function `get_word_frequencies_generator` uses the generator to feed the words into the `Counter` This is way more efficient for larger datasets because it doesn't load the entire file into memory at once. it keeps it on disk and only fetches the current line It's not magic but for large datasets its close to magic

Let's talk about edge cases you need to handle things like weird characters maybe you're dealing with text with emojis or non-standard punctuation in this case the `string.punctuation` might not catch all of it so you can clean it with regular expressions its a tool you must use or be able to use in all your text processing tasks here is an example

```python
import re
from collections import Counter

def get_word_frequencies_regex(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Use regex to remove non-alphanumeric characters
    words = text.split()
    word_counts = Counter(words)
    return word_counts

# Example usage
example_text = "This is a test! 😉 with some extra things. Like numbers 123."
frequencies = get_word_frequencies_regex(example_text)
print(frequencies)
```

Alright this one looks different right I replaced the `string.punctuation` part with a regex. The line `re.sub(r'[^a-z\s]', '', text)` uses a regular expression to keep only lowercase letters and spaces in the text. everything else is removed this gives you a more robust way to handle all kinds of input including weird emojis unicode symbols and whatnot which you may encounter on text from the wild.

Now what about more complex scenarios? you will want to be familiar with nltk specifically its tokenizers if you want to do advanced processing you can use it to deal with things like stemming lemmatization and stop words but that is beyond this question for the purpose of this I am just covering the basics

One more thing keep in mind that you should also be paying attention to performance these things can get slow when you are processing gigabytes of data if you are planning to be doing that then think about optimizing your code for maximum performance and you can do that by profiling the code and finding the bottlenecks and optimizing them

For resources there are some amazing things out there. The "Natural Language Processing with Python" book by Steven Bird, Ewan Klein, and Edward Loper is a great one to get a strong grasp on all this text processing. Also look into academic papers specifically those that describe state of the art algorithms for text processing it's an excellent way to learn how the professionals are tackling the problem. There are a lot of blog posts out there as well some more reliable than others but I find the best way to learn is to just try stuff out and see what breaks and learn from those experiences. Remember you have to fail a lot to get good at these things so don't be afraid to experiment and make mistakes it's all part of the learning process and besides at least it will be a funny story to tell later on remember my boss he is probably still reading transcripts at this hour.

So that's the gist of it It might seem like a lot but its all just building on a few basic concepts. Start simple and build your way up its the same in almost everything in life. And remember don't try to over complicate it. The best code is always the simplest code it's a principle I've seen the benefits of many many times in my years. Now get out there and count some words.
