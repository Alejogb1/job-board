---
title: "How can I configure Bot Framework LUIS to ignore specific words?"
date: "2024-12-23"
id: "how-can-i-configure-bot-framework-luis-to-ignore-specific-words"
---

Okay, let's tackle the challenge of having LUIS, the Language Understanding intelligent service from the Bot Framework, selectively ignore certain words. This is a common requirement, and frankly, I’ve dealt with it myself several times over the years, usually in scenarios involving noisy conversational data or when dealing with very specific domain lexicons. The good news is that LUIS offers several mechanisms that, while not a single ‘ignore word’ flag, can effectively accomplish this behavior through a combination of careful model design and feature engineering. It's less about explicitly telling LUIS "ignore this," and more about shaping how it interprets your users' input.

Firstly, let's get one misconception out of the way: LUIS doesn't inherently support an 'ignore words' list like a basic spam filter. Instead, its understanding is based on probabilistic modeling of the entire sentence, considering all words in context. So, direct exclusion is not the approach; rather, we strategically guide LUIS to focus on the *relevant* parts of the utterance.

The first and arguably most straightforward technique is to use phrase lists. These aren't about *ignoring* words outright, but instead, about grouping them into a conceptual "chunk" that helps LUIS generalize. If, for instance, you frequently see phrases like "please", "can you", "would you", "i want" before the core request, you don’t want LUIS to see each of these as equally important for intent recognition. By defining a phrase list that includes these conversational formalities (e.g., “polite_phrases” with the values “please,” “can you,” “would you,” “i want”), you are effectively indicating that they should be treated as a single feature rather than individual words. This means LUIS will pay more attention to the remaining, more semantically valuable words for determining the intent. While this doesn’t strictly ignore the words, it lowers their relative impact on intent and entities, which often achieves the desired outcome.

Here's a simple snippet demonstrating this via the LUIS authoring api. (Note: You will need the luissdk for python):

```python
from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from msrest.authentication import CognitiveServicesCredentials

# Your LUIS API key and endpoint
luis_key = "YOUR_LUIS_AUTHORING_KEY"
luis_endpoint = "YOUR_LUIS_ENDPOINT"

# Your LUIS app id
app_id = "YOUR_LUIS_APP_ID"

# Your app version
version_id = "YOUR_VERSION_ID"


credentials = CognitiveServicesCredentials(luis_key)
client = LUISAuthoringClient(endpoint=luis_endpoint, credentials=credentials)

# Create a phrase list
phrase_list_create_object = {
    "name": "polite_phrases",
    "phrases": ["please", "can you", "would you", "i want", "i'd like"],
    "is_exchangeable": False
}

phrase_list_id = client.features.add_phrase_list(app_id, version_id, phrase_list_create_object)

print(f"Phrase list created with id: {phrase_list_id}")
```

In this code, we're using the LUIS SDK to create a phrase list called "polite\_phrases." The `is_exchangeable=False` means the phrases are added as literal values. If it were true, LUIS would also consider variations of the provided phrases. After creating the phrase list, you need to train the model for the change to take effect. Note, however, this doesn't completely erase influence.

Another effective method is to use patterns. Patterns provide a way to define templates that match specific structures in utterances. You can use this to effectively “skip” over certain parts of the user input. For instance, if you know that users often prefix a numerical value with "the number is," and this phrase isn't important for extracting the numerical value as an entity, you can define a pattern that captures the number while effectively ignoring the prefix.

Consider the following pattern example: `the number is {number}`. This tells LUIS to match any utterance following that format and to primarily treat the `{number}` portion as the entity. The words before it are part of the matching pattern but aren't directly used as strong signals for intent. LUIS focuses on the {number} part for entity extraction, effectively disregarding the surrounding "the number is" phrase for the most part. This is more sophisticated than phrase lists, giving you contextual pattern matching.

Here's how to add a pattern:

```python
pattern_object = {
    "pattern": "the number is {number}",
    "intent": "GetNumber"  # Ensure you have a 'GetNumber' intent or choose an appropriate one
}


pattern_id = client.patterns.add_pattern(app_id, version_id, pattern_object)

print(f"Pattern created with ID: {pattern_id}")

```
Here, a pattern is created to recognize utterances that match the given template and extracts the number, associating it with "GetNumber" intent and effectively minimizing impact of the "the number is" phrase on intent.

A third approach, though a bit more advanced, is to use prebuilt entities and roles. While not directly about ignoring words, it allows you to be specific about what entities you care about and how they should be interpreted. For example, instead of training a custom entity to identify dates, leverage the pre-built `datetimeV2` entity type from LUIS. By focusing on the pre-built entity you are telling LUIS where the relevant semantic information resides and the rest becomes noise in comparison. You can even assign roles to these entities, further refining how your bot understands the context. This, alongside active learning, is where the iterative nature of LUIS model creation comes into play. By observing where the model falls short, you fine-tune and adjust accordingly.

Imagine that in your use-case, "tomorrow at 5pm" is an important piece of info, while 'remind me' that might commonly precede the date is not. You would leverage the built in datetimeV2 type and assign a role of say 'event_time'. This allows you to capture the specific date, and let the surrounding words naturally become less important to the intent.

Let's use python to add a role to a datetimeV2 entity:

```python
entity_type = "datetimeV2"
role_name = "event_time"

# fetch existing entities from our app, you might need to iterate to find the right one, this will be a single element list
entities = client.entities.list_entities(app_id, version_id)

# Assuming the datetimeV2 type exists, if not it would require additional calls
for entity in entities:
  if entity.type_id == entity_type:
      role_create_object = {
          "name": role_name
      }
      role_id = client.entities.add_entity_role(app_id, version_id, entity.id, role_create_object)
      print(f"Role created with id: {role_id}")
      break
```

This code snippet shows the basics of adding a role to existing pre-built entity. This would require a more complex setup in real life with additional error handling etc, but helps illustrate the functionality.

In practice, you'll likely need to combine these techniques, iterating based on your bot's performance. A good place to start would be with a combination of the phrase lists and patterns. I highly recommend looking into the official Microsoft documentation on LUIS, specifically the sections on phrase lists, patterns, prebuilt entities and roles. Also consider "Speech and Language Processing" by Jurafsky and Martin, as it provides foundational knowledge on text processing and natural language modeling, offering deeper understanding on how these techniques work behind the scenes. Lastly, if you wish to dive deep on specific models and algorithms consider work by Christopher Manning as a solid foundation, as well as recent work in transformers and large language models which significantly influences how modern NLP works.

The key takeaway here is that you rarely 'ignore' words in the strictest sense. You train LUIS to focus on the relevant information, which effectively diminishes the impact of other words in the utterance. The methods I described are not magic bullets; continuous training, observation, and tweaking are crucial for achieving optimal performance. Your model is always in development, not a finished product. It’s a continuous process of refinement.
