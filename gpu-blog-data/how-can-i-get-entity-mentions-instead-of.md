---
title: "How can I get entity mentions instead of tokens in PyTorch NER?"
date: "2025-01-30"
id: "how-can-i-get-entity-mentions-instead-of"
---
The core issue when working with Named Entity Recognition (NER) in PyTorch often lies in the output granularity; a model typically predicts tags for each *token*, not for the higher-level *entities* formed by those tokens. This requires post-processing of the model’s output to reconstruct the entities from the per-token tags. In my experience, a frequent source of confusion arises from assuming the model provides these directly. I've seen many implementations fail because they handle only isolated tags, and not contiguous multi-token entities.

I'll break down how to bridge this gap. First, the model outputs a sequence of predicted tags, corresponding one-to-one with input tokens. These tags often use the BIO (Beginning, Inside, Outside) or IOB (Inside, Outside, Beginning) annotation scheme. The "B-" prefix indicates the start of an entity, "I-" marks an inside token of that entity, and "O" means that a token is not part of any entity. For example, in "New York is a city", a BIO tag sequence might be: "B-LOC", "I-LOC", "O", "O", "O". To get the "New York" entity, we need to analyze the sequence for contiguous tagged segments.

Let’s consider the practical conversion of this tagged sequence to a list of entity mentions. The fundamental algorithm iterates through the sequence. When a "B-" tag is encountered, we initiate an entity. We then keep adding the corresponding tokens to the entity's span until a non-“I-” tag with same entity type is seen or sequence ends. The token indices are critical for reconstructing the actual entity text. The algorithm must handle cases where no "B-" is present (no entities are found), or when multiple entities are present sequentially. In such scenarios, nested or overlapping entity detection also becomes an important challenge, however, this response shall focus on non-overlapping entity extraction.

Here is a code snippet illustrating a basic implementation:

```python
def extract_entities(tokens, tags):
    entities = []
    current_entity = None

    for i, tag in enumerate(tags):
      if tag.startswith("B-"):
        if current_entity:
          entities.append(current_entity)
        entity_type = tag[2:]
        current_entity = {
            "type": entity_type,
            "start": i,
            "end": i+1,
            "text": tokens[i:i+1]
        }
      elif tag.startswith("I-"):
          if current_entity and tag[2:] == current_entity["type"]:
              current_entity["end"] = i+1
              current_entity["text"].append(tokens[i])
          else: # Handle badly formed tag sequences.
            if current_entity:
              entities.append(current_entity)
            current_entity = None
      elif current_entity:
        entities.append(current_entity)
        current_entity = None
    if current_entity:
       entities.append(current_entity)

    for ent in entities:
       ent["text"] = " ".join(ent["text"])
    return entities

tokens = ["New", "York", "is", "a", "city", "."]
tags = ["B-LOC", "I-LOC", "O", "O", "O", "O"]

entities = extract_entities(tokens, tags)
print(entities)
# Output: [{'type': 'LOC', 'start': 0, 'end': 2, 'text': 'New York'}]
```

This first example showcases the core logic. I loop through the tokens and tags. When I see a "B-" tag, I start a new entity.  "I-" tags append to the current entity. A non "I-", or "B-" tag signifies the end of an existing entity. Note that we also handles malformed tag sequences that might occur if model predictions are not perfect. The entity dictionary keeps track of its type, start and end token indices, and the text derived from the tokens. The final output shows an entity dictionary with the extracted information.

Now let's consider a more complex scenario with multiple entities:

```python
tokens = ["Apple", "Inc.", "is", "based", "in", "Cupertino", ",", "California", "."]
tags = ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "O", "B-LOC", "O"]
entities = extract_entities(tokens, tags)
print(entities)
# Output: [{'type': 'ORG', 'start': 0, 'end': 2, 'text': 'Apple Inc.'}, {'type': 'LOC', 'start': 5, 'end': 6, 'text': 'Cupertino'}, {'type': 'LOC', 'start': 7, 'end': 8, 'text': 'California'}]
```

In this second example, the sequence includes multiple entities. The algorithm correctly identifies "Apple Inc." as an organization and both "Cupertino" and "California" as locations. The code handles the transitions between entities effectively. The logic remains consistent in correctly identifying each entity type, location and corresponding tokens. I have frequently encountered such multi-entity scenarios in practice, making this kind of robust extraction method crucial.

Finally, let's illustrate how to handle a case with no entities being found:

```python
tokens = ["This", "is", "a", "sentence", "with", "no", "entities", "."]
tags = ["O", "O", "O", "O", "O", "O", "O", "O"]
entities = extract_entities(tokens, tags)
print(entities)
# Output: []
```
The third example demonstrates the case where no entities are found in the sequence. The function correctly returns an empty list, as expected. This is an important aspect to manage in NER system development, as non-entity sentences are commonly encountered.

To further enhance this extraction process, several avenues exist. First, it may be useful to modify the function to include confidence scores of entity identification, if model provides those. Second, error handling should be comprehensive, including edge cases of malformed input sequences. Third, performance can be improved by using more efficient data structures and algorithms, especially for longer documents.

To learn more about these concepts, I recommend exploring the following resources.  First, I suggest reading the documentation and tutorials of your particular NER model. The model documentation often provides practical guidance on how to interpret output, including the specifics of its tokenization and tagging scheme. Second, I recommend looking into academic papers about named entity recognition, as the field has a robust history of algorithm development.  Third, study various implementations of entity extraction across various libraries. Understanding the code behind different NER packages will give you new insights on how each function manages the output and converts it to something usable. Finally, experimentation is crucial, so construct test cases that are similar to the dataset you are working on, to ensure a robust implementation.
