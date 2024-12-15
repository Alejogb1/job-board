---
title: "Why is Plot Tweets Length not clear why the length has high values?"
date: "2024-12-15"
id: "why-is-plot-tweets-length-not-clear-why-the-length-has-high-values"
---

hello there,

i see you're having a bit of a head-scratcher with tweet lengths and plotting them. yeah, i've been down that rabbit hole myself, plenty of times. it's one of those things that looks simple on the surface, but when you actually get into it, the devil is in the details – or rather, the encoding and counting algorithms. i'll try and break it down for you based on my experience.

first off, what seems like just a string of characters isn't always as straightforward as that. there's this thing called unicode. tweets, unlike older sms messages, need to be able to handle a whole world of characters, emojis, symbols etc. this means they're stored internally using unicode, specifically utf-8 which is a variable-width character encoding. this means one single character in the human readable sense, can actually take up one or multiple bytes internally. so, when you see a tweet like "hello world! 😎" there are more bytes than just the letters. the smiling face with sunglasses is one character in terms of how you see it but in utf-8 it's four bytes. this is where a lot of the length "confusion" comes from. what looks like 14 chars may really be way more bytes under the hood.

when you plot tweet lengths, if you are measuring the raw byte count (or perhaps a naive character count where you just iterate through the string), you will likely see higher values than expected if the tweets contain a large number of these multi-byte characters, emojis, or some of those less used unicode glyphs that can span multiple bytes. that's the core of the problem. it's not so much that the tweet *is* long in the sense of a simple ascii count, it's that it's longer in bytes, or longer in terms of logical characters if your count is not utf-8 aware.

i remember once, ages ago, i was working on a sentiment analysis project. we were collecting a large stream of tweets, and i was using python with its standard string length function `len()`. i was just blindly using it as a proxy to what we considered “tweet size”. i started noticing that the histograms of “tweet lengths” looked completely skewed. i was expecting most tweets to be in the 100-200 characters range, but i was seeing a lot of values going way beyond that into the 250+ range. i initially thought there was some encoding error on my end, or that something was corrupting the tweet text data itself. turned out the problem was this naive length function not being aware of the bytes used by some unicode characters. i just wasn't counting correctly because i was counting codepoints and not bytes. it was a silly mistake but it also took me some good hours of debugging to find.

the fix for that, in python, is using the `encode` method to convert the string to a sequence of bytes, then taking the length of that. here is a simple way to do it.

```python
def get_tweet_byte_length(tweet_text):
    return len(tweet_text.encode('utf-8'))

#example
tweet = "this is a test with éàöü and 😎!"
byte_length = get_tweet_byte_length(tweet)
print(f"the tweet '{tweet}' has {byte_length} bytes")
```

here, we force the tweet text into a utf-8 byte string, then `len()` gives us the number of bytes. that is usually the right metric if what you want is the internal representation size. if you are more interested in user perceived length you need to consider how things are counted when displayed.

now, sometimes, twitter itself might do some weird character substitution or normalization. it's not very common these days as they have improved but in the past, there was some url shortning in their system and that would affect the actual content. if you're collecting tweets from their api, there's some data scrubbing that's probably happening behind the scenes as well. this can also contribute to some differences between what you see and what you think you're measuring. i once encountered a strange issue where some characters would transform when going through the twitter api, especially when dealing with very specific unicode ranges. the twitter library would try to be helpful with character normalisation and sometimes some edge cases could cause unexpected changes on length in bytes. after that i always double checked everything.

also, when displaying tweets, especially in web pages or apps, there could be some extra markup added, or special characters rendered, that are not part of the "core" tweet. so, if you are not measuring the original text and are instead taking it from a rendered view it could have all sorts of things added by the renderer, not by the text itself.

there are cases where, for example, you need to consider diacritics. these are the small glyphs added to letters such as in spanish or german éàöü. sometimes, a character with diacritics is actually represented as two or more code points, meaning they could be counted differently based on your counting method (or programming language library).

that said, a more "human-centric" length count might focus on *grapheme clusters*. a grapheme cluster is what a user thinks of as a single character unit, it's the "visual" character. so, if you have a character with a diacritic or an emoji, the whole cluster can be considered as a single logical unit even though internally it might be stored as multiple code points or bytes. to get this right you might need a unicode library and some understanding of grapheme boundaries. you can explore the `regex` library in python which provides full unicode support.

here's how you might use it to count grapheme clusters.

```python
import regex

def get_tweet_grapheme_length(tweet_text):
  return len(regex.findall(r'\X', tweet_text))

tweet = "this is another test with éàöü and 😎! plus some Ż"
grapheme_length = get_tweet_grapheme_length(tweet)
print(f"the tweet '{tweet}' has {grapheme_length} grapheme clusters")
```

the `\X` regex pattern in `regex.findall()` matches extended grapheme clusters which provides you with a count closer to what the human eye would perceive as the length of the text. this gives you a more sensible user-focused length count of characters.

one more thing: if your plot uses length bins that are too wide, then the variability of the counts might seem large. for instance, if you have bins like 0-50, 50-100 and 100-250, then the later bins could contain very different byte lengths and contribute to a large variance. also the axis itself should reflect if you are measuring bytes or grapheme clusters.

here's a quick example using `matplotlib` just to put this all together:

```python
import matplotlib.pyplot as plt
import regex

def get_tweet_byte_length(tweet_text):
    return len(tweet_text.encode('utf-8'))

def get_tweet_grapheme_length(tweet_text):
  return len(regex.findall(r'\X', tweet_text))

tweets = [
    "short tweet",
    "a slightly longer tweet with some éàöü.",
    "a very long tweet with emoji 😎😎😎😎😎 and more characters to really test things out like Ż or ؜",
    "another one, this should also have a normal length",
    "one last test with a lot of 🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪🤪"
]

byte_lengths = [get_tweet_byte_length(tweet) for tweet in tweets]
grapheme_lengths = [get_tweet_grapheme_length(tweet) for tweet in tweets]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(byte_lengths, bins=5)
plt.title("byte lengths")
plt.xlabel("length in bytes")
plt.ylabel("frequency")

plt.subplot(1, 2, 2)
plt.hist(grapheme_lengths, bins=5)
plt.title("grapheme cluster lengths")
plt.xlabel("length in grapheme clusters")
plt.ylabel("frequency")

plt.tight_layout()
plt.show()
```

this example shows you two histograms on the same figure, one showing byte lengths and another showing grapheme cluster lengths. this should highlight the differences you'll observe when using one or the other. the "length" of text can be subjective, depending what you want to measure and what units you want to use. i hope you find this helpful, i had to create a custom function once that counted the amount of times a person used the 'facepalm' emoji in a text analysis project and the naive string methods didn't work.

if you want more information you could take a look at the unicode standard itself, it is available online. there's a book i remember, something like "unicode explained" or "programming with unicode", i can't remember the exact title now, that was very helpful. reading the relevant parts of the official unicode documentation, available at their site, will give you all the details about code points, encodings, and normalization forms if you want to delve deeper into the subject. you can also take a look at the documentation for the `regex` library as well as the `string` and `codecs` python standard libraries for the details about encoding and string processing. it's always useful to have that in your mind if you are dealing with real text in your projects. hope this sorts it out for you, let me know how it goes.
