---
title: "How can I improve the accuracy of NER on Spacy for a tag that is not following one format?"
date: "2024-12-23"
id: "how-can-i-improve-the-accuracy-of-ner-on-spacy-for-a-tag-that-is-not-following-one-format"
---

Alright,  It's a problem I've certainly bumped into more than a few times. You're dealing with a named entity recognition (NER) challenge using spaCy, and the hurdle is that your specific entity tag isn't conforming to a predictable pattern. This isn't uncommon; real-world data rarely fits neatly into pre-defined boxes. It's the kind of issue that often requires a bit of fine-tuning beyond just relying on out-of-the-box models.

I remember a project a few years back where I was working on processing legal documents. We had to identify specific clauses related to contract terminations, which weren't always worded consistently, nor were they marked with obvious keywords. The standard spaCy models did a decent job on the more structured parts, but for these termination clauses, the accuracy was consistently low.

So, where do we begin? The key here is understanding that spaCy's models are trained on specific data. If your tag doesn’t align with the patterns the model was trained on, you'll see lower accuracy. We can improve this situation by primarily focusing on two complementary strategies: refining the training data and augmenting our pipeline components.

First, let's talk about the data. The foundation of any good NER model is a robust dataset that represents the nuances of your specific use case. For your tag that isn't following one format, the solution often involves creating more examples that reflect this variability. Think about it: the model learns through exposure. If you're only showing it examples of your entity that look one way, it's going to struggle when it encounters a variation. In my legal document project, we painstakingly reviewed hundreds of documents, annotating all the different ways these termination clauses appeared. We found that they could be phrased as ‘upon breach of this agreement’, ‘if the party fails to meet the obligations outlined’ and countless similar variations, all meaning the same thing but not always obviously so for a naive algorithm. This meant a significant manual effort, but it was the crucial foundation for improving accuracy. You might use annotation tools like prodigy (by the creators of spacy) to ease the process.

Now, let's look at augmenting our model through code and strategy. We'll use three examples to illustrate the main points.

**Example 1: Pattern Matching**

Directly extending spaCy's pipeline using pattern matching can be very effective. If your entity sometimes appears alongside consistent tokens, we can create a series of `Matcher` objects to identify these contextual clues.

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm") # Or any other spaCy model
matcher = Matcher(nlp.vocab)

# Add patterns to the matcher
pattern1 = [{"LOWER": "terminates"}, {"POS": "ADP"}, {"POS": "DET"}, {"POS": "NOUN"}]
pattern2 = [{"LOWER": "cancellation"}, {"POS": "OF"}, {"POS":"DET"},{"POS": "NOUN"}]
pattern3 = [{"LOWER": "cease"}, {"LOWER": "to"}, {"LOWER": "be"}, {"POS":"NOUN"}]

matcher.add("TERMINATION_CLAUSE", [pattern1, pattern2, pattern3]) #Added multiple patterns

def add_termination_entity(doc):
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        doc.ents = list(doc.ents) + [span.as_doc().ents[0].as_span(doc)] #ensure the span is aligned to the correct doc
    return doc

nlp.add_pipe(add_termination_entity)

# Process text using the enhanced pipeline
text = "The agreement terminates upon notice. There was also a cancellation of the policy. The services will cease to be available."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```
In this example, I define several pattern objects for common terminological language surrounding terminations. In addition to matching tokens, you can also use lemmatization, regular expression matches, or even custom functions within your patterns to capture very specific rules for your entity.
In this approach, we are adding a new pipeline component. The original NER model will still run and identify it's tags as it normally would, but after this process, we augment the document's entities with our patterns.

**Example 2: Training a Custom NER Component**

If pattern matching alone isn't cutting it, the next step is to train a new NER model, or a component, specifically for your unique entity tag. This involves creating your own training data and fine-tuning spaCy. It's a more computationally expensive approach but often provides higher accuracy for complex, context-dependent entities.

```python
import spacy
import random

def train_custom_ner(train_data, model_name='custom_ner'):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities", []):
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()
    for itn in range(20):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            nlp.update([text], [annotations], sgd=optimizer, losses=losses)
        print(f"Losses at iteration {itn}:", losses)
    nlp.to_disk(model_name)
    return model_name

# Example training data
TRAIN_DATA = [
    ("This contract is terminated if party A does not uphold", {"entities": [(17, 31, "TERMINATION_CLAUSE")]}),
    ("Breach of terms lead to immediate termination of service", {"entities": [(28, 44, "TERMINATION_CLAUSE")]}),
    ("The contract will be cancelled by A", {"entities": [(17, 29, "TERMINATION_CLAUSE")]}),
    ("A is no longer bound by contract obligations", {"entities": [(0, 1, "TERMINATION_CLAUSE")]}),

]

model_name = train_custom_ner(TRAIN_DATA)
nlp = spacy.load(model_name)


text = "This contract is terminated if party A does not uphold. Breach of terms lead to immediate termination of service. A is no longer bound by contract obligations, and the contract will be cancelled by A"
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)

```
Here we're creating a custom NER model from scratch. The `TRAIN_DATA` includes text samples and their annotated entities with the "TERMINATION_CLAUSE" label. This allows spaCy to learn the specific nuances of your tag from your training data. Note, this example is for demonstration purposes, ideally you'd want more training examples. A practical approach would be to also start from a base model (such as 'en_core_web_sm') and train only the ner component.

**Example 3: Data Augmentation using Rules and Patterns**

Since data is key to accuracy, we can implement rules and patterns to augment our existing training data. In the example below, we are going to create rules to generate synthetic examples of our 'TERMINATION_CLAUSE' to increase our training data.

```python
import spacy
import random
from spacy.matcher import Matcher
from spacy.tokens import Doc

def generate_synthetic_data(train_data, nlp):
    matcher = Matcher(nlp.vocab)

    def is_termination_entity(doc):
        return any(ent.label_ == "TERMINATION_CLAUSE" for ent in doc.ents)

    def _generate_variations(doc, start, end):
      variations = []

      verb = doc[start].text
      prepositional_phrase = "due to a violation" #this is a stand-in for a more complex variation
      variation1 = Doc(nlp.vocab, words=[verb,"will","not", "be", "enforceable", prepositional_phrase])
      variation2 = Doc(nlp.vocab, words=[verb,"was", "immediately", "nullified", prepositional_phrase])
      variations.extend([variation1,variation2])

      return variations

    patterns=[
        [{"POS":"VERB"}, {"LOWER":"by", "OP":"?"}, {"POS":"DET", "OP":"?"}, {"POS":"NOUN"}],
        [{"LOWER": "breach"}, {"POS": "ADP"},{"POS":"NOUN", "OP":"?"}]
    ]
    matcher.add("TERMINATION_TRIGGER", patterns)

    new_data = []
    for text, annotations in train_data:
        doc = nlp(text)
        for match_id, start, end in matcher(doc):
            if is_termination_entity(doc):
                variations = _generate_variations(doc, start, end)
                for variation in variations:
                   new_data.append((variation.text, {"entities": [(0,len(variation.text), "TERMINATION_CLAUSE")]}))

    return new_data


nlp = spacy.load("en_core_web_sm")
TRAIN_DATA = [
    ("This contract is terminated if party A does not uphold", {"entities": [(17, 31, "TERMINATION_CLAUSE")]}),
    ("Breach of terms lead to immediate termination of service", {"entities": [(28, 44, "TERMINATION_CLAUSE")]}),
    ("The contract will be cancelled by A", {"entities": [(17, 29, "TERMINATION_CLAUSE")]}),
    ("A is no longer bound by contract obligations", {"entities": [(0, 1, "TERMINATION_CLAUSE")]}),

]

synthetic_data = generate_synthetic_data(TRAIN_DATA, nlp)
combined_data = TRAIN_DATA + synthetic_data
print(len(combined_data))

# Train the custom NER model with the extended data, as in the example before
```

Here, we use pattern matching to find instances that may lead to termination, like a "breach of contract" or a "terminated" verb in some context and use it to generate synthetic samples. This process augments our training data with more examples. In a real scenario, the `_generate_variations` method would be much more elaborate using linguistic rules and transformations to mimic the possible expressions of your entities. The benefit is that you are using existing patterns to create new, labeled examples which will improve the model's robustness. This helps when it is difficult to acquire additional annotated real-world examples.

For further understanding, you should explore the following resources:

*   **The spaCy documentation:** It's the definitive source for everything spaCy-related, including fine-tuning and training procedures: `https://spacy.io/usage/training`.
*  **"Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper:** A classic resource on NLP, providing a strong foundation for understanding the fundamentals behind spaCy's techniques.
* **"Speech and Language Processing" by Daniel Jurafsky and James H. Martin:** A very comprehensive resource on computational linguistics that covers many concepts underlying spaCy's models, including sequence models.
* **"Deep Learning for Natural Language Processing" by Jason Brownlee:** This is a more modern take on the matter which deals with the deep learning algorithms at the core of many contemporary NLP tools.

In my experience, a combination of these techniques often works best. Start with pattern matching to address the easier cases. Then, prepare a good training dataset and fine-tune a new model or pipeline component. Consider augmenting the data with generative techniques, which can enhance both model accuracy and its robustness. Remember that iterative improvement is key: evaluate, refine, and repeat. It’s an ongoing process, but the result is usually a much more precise and valuable NER system for real-world applications.
