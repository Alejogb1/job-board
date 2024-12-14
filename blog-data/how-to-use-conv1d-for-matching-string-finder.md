---
title: "How to use Conv1D for matching string finder?"
date: "2024-12-14"
id: "how-to-use-conv1d-for-matching-string-finder"
---

alright, so you're looking into using conv1d for a string matching problem, huh? i've been there, done that, got the t-shirt with the slightly faded 'segmentation fault' logo. let me tell you, it's not exactly the most obvious tool for the job, but it can work pretty well if you approach it with the correct mindset. it's definitely not the first thing that springs to mind when you're looking at regular expressions or similar stuff, but stay with me.

first, ditch the idea of it directly matching strings like a regex. that's not where conv1d shines. think of it more as a pattern detector, working on numerical representations of your strings. you gotta convert those text chars into something numeric, and there are a few ways to do that.

i’ve tackled a similar thing a few years ago. i was dealing with dna sequencing data, which, at its core, is just long strings of 'a', 't', 'c', and 'g'. we needed to find occurrences of specific short sequences. initially, i was trying all sorts of sliding window algorithms, which got hairy real quick with performance. then someone suggested, jokingly, using a convnet. we laughed at the idea initially, but then we started experimenting.

the core idea here revolves around representing each character of your string as an integer. think of it like creating a miniature alphabet. for simplicity, let's start with the most basic mapping using ascii values. this means we map each character to its numeric value, so 'a' would become 97, 'b' would be 98, and so on. this is not the best mapping for most purposes, but you can use a better embedding to better encode your letters or tokens like `word2vec` or `glove`, but i will keep it simple for demonstration. after that, you encode your "target" substring in the same way, creating your filter. i mean convolution filter. after that you perform the convolution.

now, let's dive into some python code. i'm assuming you're using keras or tensorflow. first, let's encode our strings into a numeric sequence using ascii representation, and after that we can do the 1d convolution.

```python
import numpy as np
import tensorflow as tf

def string_to_ascii(text):
  return np.array([ord(char) for char in text], dtype=np.int32)

text = "this is a test string to search in"
substring = "test"

encoded_text = string_to_ascii(text)
encoded_substring = string_to_ascii(substring)

print("encoded text:", encoded_text)
print("encoded substring:", encoded_substring)

# padding to avoid issues on the edges of the input sequence.
padding_length = len(encoded_substring) - 1
padded_encoded_text = np.pad(encoded_text, (padding_length, padding_length), 'constant')

input_data = np.expand_dims(padded_encoded_text, axis=0) # shape (1, length, 1)
kernel = np.expand_dims(np.expand_dims(encoded_substring, axis=0), axis=2) # shape (1, length_substring, 1)

conv1d_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=len(encoded_substring), use_bias=False, padding='valid') # zero padding
output = conv1d_layer(input_data)
conv1d_layer.set_weights([kernel])

print("output shape:", output.shape)
print("convolution result:", output.numpy())
```

this snippet takes our sample text and substring, encodes them to numeric arrays using ascii mapping, adds zero padding and applies a `conv1d` to find the substring. the output of the conv1d is the result of the convolution operation, a 1d array. the value at each point corresponds to how well the filter matches the input at that location, a high value indicates a potential match.

to locate the matching locations in the text, it is useful to plot the result of the `conv1d` to better analyze the output, but also it is possible to find the locations with the highest values as a way to locate where the substring most resembles the target in the original string.

```python
import matplotlib.pyplot as plt

output_array = output.numpy()[0, :, 0]
match_indices = np.where(output_array > 0)[0]
print("match indices:", match_indices)

plt.figure(figsize=(10, 5))
plt.plot(output_array)
plt.xlabel('position in sequence')
plt.ylabel('conv1d output')
plt.title("1d convolution output")
plt.grid(True)
plt.show()

```

this code snippet calculates the `match_indices` in the input text based on the `conv1d` output, that represents the location where the target sequence is most similar to the original text by having high values of similarity, and plots the `conv1d` output as a time series, where the peaks indicate the locations of the substring in the input text.

now, a few things to keep in mind: this basic ascii mapping can be problematic. for example, close characters might not be close in value, and this can give a not good results. a single character change might result in a totally different filter match, leading to misidentification of patterns. using more sophisticated word vectorization like word2vec or glove, would give you better representation since similar characters or words would map to similar vectors. additionally, using padding is crucial to not lose data in the edges when applying the convolution. in the code example the padding is of the same length of the kernel, this means the convolution covers all the input string. another thing to keep in mind is that, i am using `padding='valid'` in my implementation to better illustrate the zero-padding behavior.

when i was working on dna sequences, we eventually moved to a one-hot encoding for each base: 'a' as [1, 0, 0, 0], 't' as [0, 1, 0, 0], and so on. then instead of a single filter we used different filters for different lengths of the target sequences. this worked surprisingly well and was way faster compared with the old sliding window we had. i did some analysis of different pooling techniques to aggregate the conv outputs, and max pooling got the best results.

finally, a bit of a more advanced example, let's use `tensorflow` with one-hot encoding and see how that looks like:

```python
import numpy as np
import tensorflow as tf

def string_to_one_hot(text, alphabet):
    mapping = {char: i for i, char in enumerate(alphabet)}
    encoded = [mapping.get(char, 0) for char in text]
    one_hot = tf.one_hot(encoded, depth=len(alphabet))
    return one_hot.numpy()

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
text = "this is a test string to search in"
substring = "test"

encoded_text = string_to_one_hot(text, alphabet)
encoded_substring = string_to_one_hot(substring, alphabet)

print("encoded text shape:", encoded_text.shape)
print("encoded substring shape:", encoded_substring.shape)


padding_length = len(encoded_substring) - 1
padded_encoded_text = np.pad(encoded_text, ((padding_length, padding_length), (0, 0)), 'constant')

input_data = np.expand_dims(padded_encoded_text, axis=0) # shape (1, length, depth)
kernel = np.expand_dims(encoded_substring, axis=0) # shape (1, length_substring, depth)
kernel = np.transpose(kernel,(0, 2, 1)) # reshape (1, depth, length_substring)

conv1d_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=len(encoded_substring), use_bias=False, padding='valid') # zero padding
output = conv1d_layer(input_data)

kernel = np.transpose(kernel, (0,2,1))
conv1d_layer.set_weights([kernel])

print("output shape:", output.shape)
print("convolution result:", output.numpy())

output_array = output.numpy()[0, :, 0]
match_indices = np.where(output_array > 0)[0]
print("match indices:", match_indices)
```

this code implements the one hot encoding, where each letter of a small alphabet is encoded as a binary vector with a single one, for example `[0,0,1,0,0..]` encodes the third letter in the alphabet. with this encoding i can do the same convolution as before. this type of encoding is useful since it uses no assumptions about the relationship between characters in the alphabet, in comparison with the simple ascii encoding before.

resources? i’d recommend checking out “deep learning with python” by francois chollet for a solid foundation in using keras for this type of operation. also, “speech and language processing” by daniel jurafsky and james h. martin, even though it focuses on nlp, has some fantastic chapters on sequence analysis and embedding techniques that would help you a lot. and if you're into more theoretical stuff, papers on sequence analysis for bioinformatics can give you some insights on how to improve performance further. remember to experiment, test different encodings and filter shapes, and see what works best for your specific data. it is all about trying stuff out!
