---
title: "How to write a custom Rasa component to preprocess text in a pipeline?"
date: "2024-12-23"
id: "how-to-write-a-custom-rasa-component-to-preprocess-text-in-a-pipeline"
---

,  Preprocessing text in a Rasa pipeline using a custom component is something I've spent a decent amount of time with over the years, particularly when dealing with niche linguistic quirks or very domain-specific terminology. It's often the case that the standard components aren't quite enough and a tailored approach is necessary. Here's how I usually go about crafting such a component, and a few things I’ve learned along the way.

The core idea is to create a class that adheres to the `Component` interface defined by Rasa. This interface provides the necessary methods for integrating your code seamlessly into the existing pipeline. Importantly, the component needs to handle the `train`, `process`, and, depending on your use case, the `persist` and `load` methods. Let’s focus on training and processing first, as these are usually the most crucial.

Imagine, for instance, that we're building a bot for a highly technical support environment where users often use specific identifiers or error codes that need to be standardized or masked before they’re passed to downstream components. I recall a project where we were handling customer logs; these logs frequently contained internal server names and timestamps that needed to be removed. Our NLU model performed much better when trained on cleaner, more structured data.

So, let's break it down:

First, let’s define the structure of our class, keeping it focused on the specific goal of text preprocessing. Here's how a barebones example of a component would look in Python:

```python
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.recipes.default_recipe import DefaultRecipe

@DefaultRecipe.register(
    [DefaultRecipe.ComponentType.MESSAGE_PREPROCESSOR],
    is_trainable=False  # Indicates this component doesn't require training
)
class CustomPreprocessor(GraphComponent):
    def __init__(self):
        pass

    @classmethod
    def create(
        cls,
        config: dict,
        model_storage: Resource,
        resource_config: dict,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls()

    def process(self, messages, **kwargs):
        for message in messages:
            # Implement your preprocessing logic here
            processed_text = self._preprocess_text(message.get("text"))
            message.set("text", processed_text)
        return messages

    def _preprocess_text(self, text):
        # Placeholder for actual preprocessing logic
        return text.lower()
```
This very basic snippet provides the template: It uses Rasa’s new `GraphComponent` API. I’ve added a `create` class method which is used to instantiate your component based on the configuration. The crucial part, of course, is the `process` method. It iterates through the messages and calls a helper method `_preprocess_text` in this example. The example preprocessing logic converts text to lowercase, however, it’s a placeholder; you'll need to add your specific text manipulation logic here. This initial setup doesn’t involve any model training, indicated by the `@DefaultRecipe.register` call and the `is_trainable=False` setting, making it ideal for operations like text normalization or sanitization.

Now, let's consider a more complex example involving regular expressions to remove the identifiers that I mentioned previously:

```python
import re
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.recipes.default_recipe import DefaultRecipe


@DefaultRecipe.register(
    [DefaultRecipe.ComponentType.MESSAGE_PREPROCESSOR],
    is_trainable=False
)
class IdentifierRemover(GraphComponent):
    def __init__(self, identifier_pattern):
        self.identifier_pattern = re.compile(identifier_pattern)

    @classmethod
    def create(
        cls,
        config: dict,
        model_storage: Resource,
        resource_config: dict,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config.get("identifier_pattern"))

    def process(self, messages, **kwargs):
        for message in messages:
            processed_text = self._remove_identifiers(message.get("text"))
            message.set("text", processed_text)
        return messages

    def _remove_identifiers(self, text):
        return self.identifier_pattern.sub("", text)

```

Here, I’ve added a constructor to accept an `identifier_pattern`. This pattern gets compiled into a regular expression and used within `_remove_identifiers` to replace matching patterns with an empty string, effectively removing them from the text. The configuration for this would, in the `config.yml` file of the Rasa project, look something like this:

```yaml
pipeline:
  - name: IdentifierRemover
    identifier_pattern: "[A-Z]{3}-\d{5}" # Example pattern for identifiers
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: DIETClassifier
```

This config places our `IdentifierRemover` component as the first preprocessing step, ensuring that identifiers are removed before tokenization and feature extraction take place. This is crucial as it impacts how the model perceives the input and can directly impact its performance.

Finally, let’s think about a scenario where we need to handle acronyms. Often in technical domains, people use acronyms that might not be readily understood by standard models. I recall working on a system for medical records where many acronyms for medical procedures were common. The following code snippet demonstrates how you could expand specific acronyms based on a dictionary:

```python
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.recipes.default_recipe import DefaultRecipe

@DefaultRecipe.register(
    [DefaultRecipe.ComponentType.MESSAGE_PREPROCESSOR],
    is_trainable=False
)
class AcronymExpander(GraphComponent):

    def __init__(self, acronyms):
        self.acronyms = acronyms

    @classmethod
    def create(
        cls,
        config: dict,
        model_storage: Resource,
        resource_config: dict,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config.get("acronyms", {}))

    def process(self, messages, **kwargs):
       for message in messages:
            processed_text = self._expand_acronyms(message.get("text"))
            message.set("text", processed_text)
       return messages

    def _expand_acronyms(self, text):
        words = text.split()
        expanded_words = [self.acronyms.get(word, word) for word in words]
        return " ".join(expanded_words)
```

In this example, the `AcronymExpander` class is initialized with a dictionary that maps acronyms to their expanded forms. During processing, the `_expand_acronyms` function simply looks up each word in the dictionary. If it's present, the expanded form replaces it; otherwise, the word remains as is. This approach can improve model understanding and could be configured as follows:
```yaml
pipeline:
  - name: AcronymExpander
    acronyms:
      "MRI": "Magnetic Resonance Imaging"
      "CT": "Computed Tomography"
      "ECG": "Electrocardiogram"
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: DIETClassifier
```

For a deeper dive into advanced text processing, I recommend the book “Speech and Language Processing” by Daniel Jurafsky and James H. Martin, it’s a cornerstone for anyone working in NLP. It provides a comprehensive guide to understanding and implementing text processing techniques. Additionally, papers detailing specific NLP tasks such as text normalization from conferences like ACL (Association for Computational Linguistics) or EMNLP (Empirical Methods in Natural Language Processing) could be immensely valuable.

When creating your component, think about its placement in your pipeline. The ordering is important. If you were, for instance, using spacy, the component will have it's own preprocessing stage, that should typically happen after your custom stage. Make sure that what you process in your component is the input that the next component requires. Also, bear in mind that while these custom components are incredibly useful, always try to leverage existing components within Rasa first. A thorough examination of Rasa’s documentation often reveals components that can accomplish the same goal with minimal configuration. Remember, custom components add to the codebase complexity, so only use them when you can’t accomplish the same thing another way.

In conclusion, writing custom Rasa components for text preprocessing involves understanding the `Component` interface, strategically placing the component in your pipeline, and thoroughly testing the implementation. It's a powerful way to tailor Rasa to unique linguistic requirements and can often provide a meaningful boost in model performance. Always test your component thoroughly by inspecting the text before and after the processing. It is easy to make mistakes, especially with regular expressions, and your model will certainly suffer if that happens.
