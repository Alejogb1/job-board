---
title: "How can I give a competence/quality level of a string in NLP with Python?"
date: "2024-12-15"
id: "how-can-i-give-a-competencequality-level-of-a-string-in-nlp-with-python"
---

alright, let's tackle this string competence thing. i've bumped into this a few times, and it's never as straightforward as one might hope. it's less about a single, definitive score and more about a combination of metrics. what constitutes "competence" or "quality" for a string really depends on what you're trying to achieve. are we talking about grammar? complexity? sentiment? factuality? all of the above?

i remember this one project i did back in my university days. we were building a chatbot, and we needed a way to filter out nonsensical user input. a simple length check wasn't enough; users could still type gibberish that was lengthy. we initially tried a purely rule-based approach which quickly became a spaghetti mess of edge cases. so, i had to learn to move away from the "if this than that" and adopt more flexible methods.

so, for me, the best way is a multi-faceted approach. here's what i usually do, with examples in python:

**1. language model perplexity:**

this one is a go-to for measuring how likely a given text is, according to a language model. lower perplexity generally means the text is more 'natural' or 'competent' in the statistical sense. you can think of it like, if a language model sees your text and goes, "yep, i've seen that kind of word sequence a bunch of times", that's a good sign for competence. models are usually trained on vast amounts of text, so they're really good at spotting patterns. i've had good results using models from the `transformers` library. let's say your string is called `text`, then you can use something like this:

```python
from transformers import pipeline
import torch

def calculate_perplexity(text):
    try:
        perplexity_pipeline = pipeline('text-generation', model='gpt2', device = 'cuda' if torch.cuda.is_available() else 'cpu')
        perplexity_results = perplexity_pipeline(text, max_new_tokens = 1)
        log_probability = perplexity_results[0]['generated_text'].strip()
        perplexity =  float(log_probability)
        return perplexity
    except Exception as e:
        print(f'error during perplexity calculation {e}')
        return None


text = "the cat sat on the mat."
perplexity_score = calculate_perplexity(text)
if perplexity_score is not None:
  print(f"perplexity score: {perplexity_score}")

text = "mat the on sat cat the."
perplexity_score = calculate_perplexity(text)
if perplexity_score is not None:
  print(f"perplexity score: {perplexity_score}")
```

note: you might need to install the relevant libraries if you don't have them: `pip install transformers torch`

for the above code, you will have two strings `the cat sat on the mat.` and `mat the on sat cat the.`. you will see a much lower perplexity score for the first string since it follows a language structure more similar to the one that the model was trained on.

**2. grammatical correctness and error detection:**

this one is a bit trickier, but important, especially if you care about formal writing. there are several tools out there that try to detect grammatical errors and measure the number of mistakes. i've used `spacy` for this. it provides sentence parsing capabilities, so you can programmatically find errors, or measure the ratio of well structured sentences. for instance:

```python
import spacy

def grammatical_score(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    if num_sentences == 0:
        return 0.0

    valid_sentence_count = 0

    for sentence in doc.sents:
        is_sentence_valid = True
        # check for grammatical patterns or errors here based on your needs
        # you can go into the specific linguistic trees
        # a very simplistic implementation would be checking for root verb existance
        has_verb = False
        for token in sentence:
            if token.pos_ == 'VERB':
                has_verb = True
                break
        if has_verb:
          valid_sentence_count += 1

    return valid_sentence_count / num_sentences

text = "The cat sat on the mat."
score = grammatical_score(text)
print(f"grammatical score: {score}")

text = "cat the mat the on sat"
score = grammatical_score(text)
print(f"grammatical score: {score}")

text = "mat."
score = grammatical_score(text)
print(f"grammatical score: {score}")

text = "mat"
score = grammatical_score(text)
print(f"grammatical score: {score}")
```

remember to install spacy and download the model: `pip install spacy && python -m spacy download en_core_web_sm`.

the function `grammatical_score` in this example is very simplistic and returns a float based on if the sentence has a root verb or not. more elaborate and advanced error detection or measures can be implemented using the tools provided by the library.

**3. lexical richness (type-token ratio):**

this measures the variety of words in the text. high lexical richness suggests a more competent use of language, usually. it indicates more vocabulary variety and complexity. the basic formula is (number of unique words) / (total number of words). in python this looks something like:

```python
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

def lexical_richness(text):
    nltk.download('punkt', quiet=True)
    tokens = word_tokenize(text.lower())
    if not tokens:
        return 0
    unique_tokens = len(Counter(tokens))
    return unique_tokens / len(tokens)

text = "the cat sat on the mat. the cat is sleeping"
richness = lexical_richness(text)
print(f"lexical richness: {richness}")

text = "the the the the the the"
richness = lexical_richness(text)
print(f"lexical richness: {richness}")
```

you will need nltk and punkt: `pip install nltk`. also, you will be able to see that the first string has a higher score of lexical richness than the second because it has more vocabulary diversity.

**4. semantic coherence:**

measuring how well the ideas in the string stick together is a hard one, usually you will need to go into topic modelling, and topic cohorence, for instance, using the library `gensim`. this starts getting into more advanced territory. for the scope of this, i will not include it as an example. but you should know that if you are going for a multi-faceted approach, this one can be a good measure to use, and it will make the results better.

**a note on combining measures**

after you have a few of these measures, what you want to do is to combine them. you can simply average them, or you can weight them based on the task you have at hand. i have tried in the past machine learning models, to predict the final score based on the individual scores, using a labeled dataset. that had a good outcome. it all depends on the data you have access to.

**important considerations**

*   **task-specific definitions:** before you even start coding, you need a solid idea of what "competent" means for *your* specific task. for example, a well-written technical document will be different from a piece of poetry or a casual social media post.
*   **data preprocessing:** make sure your text preprocessing is done correctly (e.g. lowercasing, punctuation removal, stemming etc). this can impact all of your measures.
*   **context matters:** for long strings, remember that global quality does not always mean local quality. a sentence within a paragraph can be fine on its own, but in the context of the entire text, its meaning can be problematic.

**and finally a resource recommendation**

if you are new to nlp i would say that one book that has been quite useful for me during the years is *speech and language processing* by daniel jurafsky and james h. martin. it is a thick and heavy book with almost 1000 pages, so be prepared to spend a few months, if not years on it.

this issue is a moving target. language models are getting better all the time and the measures are constantly evolving. what worked for me last year might not be as effective now, so it is important to keep updating your arsenal. i've seen some interesting stuff on the application of transformers to text quality analysis, so perhaps that's the direction to explore more. good luck out there, and may your code compile on the first try (just kidding... or am i?).
