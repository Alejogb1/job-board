---
title: "How can I improve bigram extraction from a DataFrame using spaCy's named entities?"
date: "2024-12-23"
id: "how-can-i-improve-bigram-extraction-from-a-dataframe-using-spacys-named-entities"
---

Okay, let's tackle this. I’ve seen this particular challenge pop up quite a few times over the years, usually when someone is moving from basic text analysis to more nuanced feature engineering, specifically for natural language processing tasks. Bigram extraction, in isolation, isn't particularly complex, but incorporating named entity recognition (NER) from spaCy adds a dimension that requires careful consideration. My experiences have shown me that naively applying bigram extraction after spaCy's processing often doesn’t yield the most insightful results. The typical problem arises because straightforward token-based bigrams might break up named entities, losing contextual meaning. We need a solution that’s aware of these entities.

The core issue is that standard bigram approaches operate on tokenized text, treating individual words as the fundamental unit. SpaCy’s NER, on the other hand, groups tokens into named entities such as people, organizations, or locations. If we blindly generate bigrams from the token stream, we can split meaningful entities, thereby introducing noise and reducing the informational value of our features. For example, "New York City" might be split into "New York" and "York City," discarding the key notion that "New York City" represents a single entity.

My preferred approach, and one I've consistently seen produce better results, is to create bigrams that either fully encapsulate a named entity or are generated between such entities and the tokens surrounding them. This essentially means we're treating named entities as if they're single, albeit multi-token, units within the context of bigram generation.

Here’s how this could be implemented, and I’ll give you a few code snippets to demonstrate:

**Example 1: Basic Token-Based Bigrams vs Entity-Aware Bigrams**

Let's assume we have a DataFrame with a column named 'text'.

```python
import pandas as pd
import spacy
from nltk import ngrams

nlp = spacy.load("en_core_web_sm")

def basic_bigrams(text):
    tokens = [token.text for token in nlp(text)]
    return list(ngrams(tokens, 2))

def entity_aware_bigrams(text):
    doc = nlp(text)
    bigrams = []
    entities = [ent.text for ent in doc.ents]
    tokens = [token.text for token in doc]
    
    i = 0
    while i < len(tokens) - 1:
        current_token = tokens[i]
        next_token = tokens[i+1]
        
        # Check if current token or next is part of an entity
        current_in_entity = any(current_token in entity for entity in entities)
        next_in_entity = any(next_token in entity for entity in entities)

        # If the current token isn't part of an entity, make a bigram and move to next
        if not current_in_entity:
            bigrams.append((current_token, next_token))
            i += 1
        
        else:
            # Find and advance past the entity, creating entity-aware bigram
            for entity in entities:
                if current_token in entity:
                  entity_tokens = entity.split(' ')
                  if entity_tokens[0] == current_token:
                    if i + len(entity_tokens) <= len(tokens) :
                        
                      if i + len(entity_tokens)  < len(tokens):
                           bigrams.append((entity, tokens[i + len(entity_tokens)]))
                      else:
                        bigrams.append((entity, 'END'))
                        
                      i += len(entity_tokens)
                    break
                  
            else: i+=1

    return bigrams

# Sample DataFrame
data = {'text': ["Apple is planning a new product launch in New York City.",
                "Google's headquarters are located in Mountain View.",
                "The meeting was attended by John Doe and Jane Smith.",
                "This new book is great. I can read all day",
                "Amazon is working on a new project in Seattle"]}
df = pd.DataFrame(data)

# Apply functions
df['basic_bigrams'] = df['text'].apply(basic_bigrams)
df['entity_bigrams'] = df['text'].apply(entity_aware_bigrams)

print(df[['text','basic_bigrams', 'entity_bigrams']])
```

In this example, `basic_bigrams` creates bigrams without any entity awareness. `entity_aware_bigrams` attempts to avoid splitting up entities by treating them as single bigram units. The output should show that the basic bigram implementation has split some named entities and the entity-aware version maintains the entity integrity. Note, in this implementation I've included 'END' as a marker for cases where entities are at the end of text. This might be modified depending on specific needs.

**Example 2: More sophisticated bigram generation with pre-defined entities.**

A more robust method would involve processing the text once with spaCy to extract the entities, then using that entity information during the bigram generation:

```python
import pandas as pd
import spacy
from nltk import ngrams

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    entities = [(ent.start_char, ent.end_char, ent.text) for ent in doc.ents]
    tokens = [token.text for token in doc]
    return tokens, entities

def entity_aware_bigrams_v2(tokens, entities):
    bigrams = []
    entity_spans = [(start, end) for start, end, _ in entities]
    
    i = 0
    while i < len(tokens) - 1:
        current_token_start = sum(len(t) + 1 for t in tokens[:i])
        current_token_end = current_token_start + len(tokens[i])
        
        next_token_start = sum(len(t) + 1 for t in tokens[:i+1])
        next_token_end = next_token_start + len(tokens[i+1])

        current_is_entity = False
        current_entity_text = None

        next_is_entity = False
        next_entity_text = None
        
        # Check if current token is part of an entity
        for start, end, entity in entities:
            if start <= current_token_start < end:
               current_is_entity = True
               current_entity_text = entity
               break

        # Check if next token is part of an entity
        for start, end, entity in entities:
            if start <= next_token_start < end:
              next_is_entity = True
              next_entity_text = entity
              break
          
        if not current_is_entity:
          bigrams.append((tokens[i], tokens[i+1]))
          i+=1

        else:
           if not next_is_entity:
             bigrams.append((current_entity_text, tokens[i+1]))
             i += len(current_entity_text.split(" "))
           else:
             
             
             if next_token_start > current_token_end:
                 bigrams.append((current_entity_text,next_entity_text))
                 i += len(current_entity_text.split(" "))
             else:
              
               i+=1

    return bigrams


# Sample DataFrame
data = {'text': ["Apple is planning a new product launch in New York City.",
                "Google's headquarters are located in Mountain View.",
                "The meeting was attended by John Doe and Jane Smith.",
                "This new book is great. I can read all day",
                "Amazon is working on a new project in Seattle"]}
df = pd.DataFrame(data)

# Apply functions
df[['tokens','entities']] = df['text'].apply(lambda x: pd.Series(preprocess_text(x)))
df['entity_bigrams_v2'] = df.apply(lambda row: entity_aware_bigrams_v2(row['tokens'], row['entities']), axis=1)
print(df[['text','entity_bigrams_v2']])
```

Here, `preprocess_text` handles spaCy processing once, and returns tokens and entity information. Then `entity_aware_bigrams_v2` uses the spans to intelligently skip the tokens forming the entity during bigram formation, this implementation uses the character offsets to determine the length and position of the tokens within the text and therefore, where they are in relation to an identified named entity. This method avoids the repeated NLP processing and should be faster for larger datasets.

**Example 3: Using a dedicated library for n-gram with pre-processing**.

While the custom implementation is useful to understand the mechanics, you can also use existing libraries designed to handle entity awareness better, or preprocess the text accordingly and then apply n-gram functions in libraries like Scikit Learn. For example, modifying the text to insert an undercore (_) to create a single token out of the entities.

```python
import pandas as pd
import spacy
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")

def preprocess_text_v2(text):
  doc = nlp(text)
  new_tokens = []
  
  i=0
  while i < len(doc):
      ent_start_char = None
      for ent in doc.ents:
        if ent.start_char == doc[i].idx:
            ent_start_char = ent.start_char
            new_tokens.append(ent.text.replace(' ', '_'))
            i += len(ent.text.split(" "))
            break

      if ent_start_char is None:
         new_tokens.append(doc[i].text)
         i += 1

  return " ".join(new_tokens)


# Sample DataFrame
data = {'text': ["Apple is planning a new product launch in New York City.",
                "Google's headquarters are located in Mountain View.",
                "The meeting was attended by John Doe and Jane Smith.",
                "This new book is great. I can read all day",
                "Amazon is working on a new project in Seattle"]}
df = pd.DataFrame(data)

# Apply functions
df['processed_text'] = df['text'].apply(preprocess_text_v2)


vectorizer = CountVectorizer(ngram_range=(2, 2))
df['bigrams_sklearn'] = df['processed_text'].apply(lambda text: vectorizer.fit_transform([text]).toarray())
df['bigrams_feature_names_sklearn'] = df['processed_text'].apply(lambda text: vectorizer.fit(text.split()).get_feature_names_out())

print(df[['text','bigrams_feature_names_sklearn']])

```

In the code, `preprocess_text_v2` replaces spaces within named entities with underscores, therefore, the named entities will be treated as one single token by the CountVectorizer function. Then, `CountVectorizer` from scikit-learn is used which generates bigrams. This demonstrates a more direct way of leveraging existing libraries for text feature engineering tasks.

**Further Reading**

For a deeper understanding of NER and text processing:
* **"Speech and Language Processing" by Daniel Jurafsky and James H. Martin:** This book provides an in-depth look at fundamental concepts of natural language processing including text processing and named entity recognition.
* **"Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper:** This book offers a more hands-on approach to NLP, covering practical techniques in Python with code examples using NLTK, but the general ideas can be applied in similar libraries.

**Final Thoughts**

The best strategy often depends on the specific characteristics of your data and the goal of your analysis. The important part is to think critically about how each step of your processing pipeline affects your final features. In my experience, combining a solid understanding of NLP techniques with a clear vision of what you're trying to accomplish has consistently been more valuable than relying solely on off-the-shelf solutions. Start with the conceptual approach, then tailor it to meet the specific demands of the data set. Good luck, and happy coding.
