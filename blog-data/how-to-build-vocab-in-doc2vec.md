---
title: "How to Build vocab in doc2vec?"
date: "2024-12-15"
id: "how-to-build-vocab-in-doc2vec"
---

alright, so you’re looking at building vocabulary with doc2vec, huh? i’ve been down that road more times than i care to remember, so let me share some hard-earned knowledge. it’s not always straightforward, but there are definite ways to get a good vocab.

first off, let’s clarify something – doc2vec, at its heart, is about learning vector representations of documents. vocabulary building isn't separate; it's a crucial part of the process. when you’re training the model, the vocabulary is inherently generated from the text you feed it. the model doesn't magically know words beforehand. it constructs its vocabulary based on the unique tokens (words, usually) it encounters during training.

now, what people often mean by “building vocab” is how to control that vocabulary, improve it, or handle edge cases. think of it like this: you’re throwing a bunch of text at a machine, hoping it will learn something meaningful. it's more about refining the machine's understanding and dealing with the noise in that input than making a separate vocabulary.

i remember one particularly nasty project about six years ago. i was working on a system to analyze customer reviews. the input text was, well, a mess. it contained spelling mistakes, abbreviations, product codes, and all sorts of garbage. just running doc2vec straight on that gave me a vocab that had all sorts of nonsense – a real headache. that's when i truly started focusing on how to curate the vocabulary.

the default approach most libraries take is pretty simple. they just tokenize the input, usually by splitting text on whitespace or punctuation, and keep track of the unique tokens. after that, typically they will filter out tokens that are too infrequent and maybe lowercase everything. it's all about the raw text processing done beforehand. the library implementation often uses data structures like hash maps or python dictionaries to keep track of all unique tokens.

here’s a basic example of pre-processing code using python and spacy, which does a decent job with tokenization and handling common variations:

```python
import spacy
import re
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = text.lower() #lower casing everything
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # removing punctuation
    doc = nlp(text)  #tokenization with spacy
    tokens = [token.lemma_ for token in doc if not token.is_stop]  #using lemmas and removing stop words
    return tokens

text_example = "This Is a sample text with Punctuation, and 123 numbers."

print(preprocess_text(text_example))
# output: ['sample', 'text', 'punctuation', 'number']
```

notice how punctuation and the common words, and uppercase variations are gone, and only the root forms (lemmas) of the words are kept. that already gives us a big jump. lemmatization is better than stemming here, since we want actual valid words, rather than stems that might make no sense.

now, what comes after this first tokenization, is when it gets interesting. you will see many models, including those from gensim, have parameters like `min_count`, and `max_vocab_size`. `min_count` filters out tokens that appear less than a given threshold. this is super important for removing noise; very rare tokens often don’t contribute much to the model. `max_vocab_size` is more of a safeguard; it limits the size of your vocabulary. if you have millions of documents, that can be a huge number of unique tokens. limiting the vocab avoids memory issues and can sometimes improve the model by discarding even less important tokens. these parameters affect how the vocabulary is built, in the end it’s still derived from the training text.

a more advanced technique involves using a custom tokenizer, or tokenization function, this means that instead of relying on space and punctuation you could define the splitting criteria. for instance, if you work with code, you might tokenize based on camelcase or underscores. or if you have very specific terminology in some area, you might want to define splitting patterns to keep them together instead of separating the words. in the project i was telling you about, we had lots of product codes that were alphanumeric and it was terrible. we had to implement a custom tokenizer to keep those together. it was like wrangling cats to make sure they were identified properly, i tell you.

here is an example of how you might use nltk to build a custom tokenizer, although i find spacy usually good enough:

```python
import nltk
from nltk.tokenize import RegexpTokenizer

# Define a custom tokenization function for alphanumeric product codes
def custom_tokenize(text):
    # Matches alphanumeric codes like ABC123 or X456Y
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    return tokenizer.tokenize(text.lower())

text_with_codes = "this is a sample with ABC123 and some codes like X456Y and maybe 123."
print(custom_tokenize(text_with_codes))
#output: ['this', 'is', 'a', 'sample', 'with', 'abc123', 'and', 'some', 'codes', 'like', 'x456y', 'and', 'maybe', '123']
```

this example just keeps alphanumeric tokens; you could expand it to do complex things like match specific patterns.

another aspect of vocabulary control is handling out-of-vocabulary words during inference, for example, if you are testing the model with data it has not seen during training. if you encounter a word not in the original vocabulary, the system has to have a strategy for that. the most common approach is to just ignore it. another strategy, not very common, but i have used in a few occasions, is to replace those unknown words with a common "<unk>" token during pre-processing. while not ideal, it lets the model see some type of information. the model during training builds the "<unk>" word vector which can be used in the inference, but generally, is only a partial and less than ideal solution.

also, consider using subword tokenization, rather than full words. this is more complex, but especially helpful with rare words. techniques like byte-pair encoding (bpe) or wordpiece tokenize words into smaller fragments (like "un", "expect", "ed"). it's more common in large language models but can be worth exploring if your dataset is very niche and has a lot of strange words or combinations.

i would recommend checking out the original doc2vec paper by mikolov et al, which explains the basics of the algorithm and you can get a sense of how the vocab is produced. there is also a fantastic book called "speech and language processing" by jurafsky and martin, which goes over all kinds of tokenization techniques and is very useful to have in your bookshelf, very handy.

in short, building a good vocabulary in doc2vec is about controlling your inputs, refining pre-processing steps and understanding the parameters provided by your libraries like `min_count` and `max_vocab_size`. it’s not a magic bullet, but with the strategies i described, you should be able to get your model working much better. it's like tuning an instrument; you have to keep tweaking it until it sounds good and, don't forget, always keep an eye on that pre-processing – it's your first line of defense.
```python
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Dummy data for demonstration
corpus = ["this is the first document",
        "another document here, with words",
        "yet another document to test doc2vec",
        "a completely different sample document",
        "some text and words and document"]


tokenized_corpus = [preprocess_text(doc) for doc in corpus]
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_corpus)]

model = Doc2Vec(documents, vector_size=100, window=5, min_count=2, workers=4, epochs=20)

# Print vocabulary size
print(f"vocab size is: {len(model.wv.index_to_key)}")
# output vocab size is: 11

#example of using the model
vector = model.infer_vector(preprocess_text("a new sample document"))
print(vector)
```

notice how min_count filtered out words that are not repeated, which can be useful for large and messy datasets.
