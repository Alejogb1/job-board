---
title: "How do I write a custom Rasa component to preprocess text in a pipeline?"
date: "2024-12-23"
id: "how-do-i-write-a-custom-rasa-component-to-preprocess-text-in-a-pipeline"
---

, let's unpack this. I've certainly spent my share of hours crafting custom components for Rasa, especially around the intricacies of text preprocessing. It’s a common requirement when you're moving beyond the standard English model or dealing with noisy, non-standard text. So, rather than launching into a textbook definition, I'll frame this around a situation I encountered some years back. I was working on a multilingual chatbot, and the standard Rasa pipeline components just weren't cutting it for some of the languages. The text needed specific cleaning before being fed into the more complex intent classifiers and entity extractors. This led to a deep dive into custom component creation, and hopefully, that experience can be useful to you here.

Essentially, a custom component in Rasa is a piece of code that you slot into the processing pipeline to achieve specific transformations on the user input. The pipeline's sequence dictates when and how your component affects the flow, be it before intent classification, after, or even in between entity extraction steps. When it comes to preprocessing text, the crucial thing is to design your component to manipulate the `message.text` attribute, which is where the raw user input resides.

The beauty of custom components in Rasa is their flexibility. They allow for fine-grained control over processing, which is essential when dealing with different languages or text structures that the off-the-shelf components might not handle efficiently. Your component can incorporate regular expressions, stemming algorithms, lemmatization techniques, or even call external APIs to enrich the text with contextual information.

Now, let’s delve into the actual coding. We're talking about a class inheriting from Rasa’s `Component` class, and you’ll primarily be implementing a few key methods, with `process` being the most vital. This `process` method is where you implement your text transformation.

**Example 1: Basic Text Lowercasing**

Let’s start with something straightforward: lowercasing. While Rasa offers pre-built components to achieve this, it’s a good starting point to illustrate the structure:

```python
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.recipes.default_recipe import DefaultRecipe

@DefaultRecipe.register(component_name="lowercase_component")
class LowercaseComponent(GraphComponent):
    @classmethod
    def create(
        cls,
        config: dict,
        model_storage: Resource,
        resource_config: Resource,
        execution_context: ExecutionContext,
    ) -> "LowercaseComponent":
      return cls()

    def process(self, messages, **kwargs):
        for message in messages:
            if message.get("text"):
                message.text = message.text.lower()
        return messages
```

In this example, the `create` method is simple; no specific initialization needed. The `process` method iterates through the list of `messages`, and for each message with a text field, it converts the content to lowercase. Notice how it's directly modifying the `message.text` attribute, this is critical. To use this component, you would add it to your `config.yml`:

```yaml
pipeline:
  - name: "lowercase_component"
  - name: "WhitespaceTokenizer"
  - name: "RegexFeaturizer"
  - name: "LexicalSyntacticFeaturizer"
  - name: "CountVectorsFeaturizer"
  - name: "DIETClassifier"
  - name: "EntitySynonymMapper"
  - name: "ResponseSelector"
```

**Example 2: Removing Punctuation and Special Characters**

Next up, let's tackle removing punctuation and special characters. This can make intent recognition more robust. We will utilise the `string` and `re` libraries.

```python
import string
import re

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.recipes.default_recipe import DefaultRecipe

@DefaultRecipe.register(component_name="punctuation_removal_component")
class PunctuationRemovalComponent(GraphComponent):
  @classmethod
  def create(
      cls,
      config: dict,
      model_storage: Resource,
      resource_config: Resource,
      execution_context: ExecutionContext,
  ) -> "PunctuationRemovalComponent":
    return cls()

  def process(self, messages, **kwargs):
    punctuation_chars = string.punctuation
    for message in messages:
      if message.get("text"):
        message.text = re.sub(rf"[{re.escape(punctuation_chars)}]", "", message.text)
    return messages
```

Here we use `string.punctuation` to easily get a list of standard punctuation marks, which are then used within a regular expression pattern to remove any matched chars. Again, modifying `message.text`. Add "punctuation_removal_component" to your `config.yml` as per the first example to enable this.

**Example 3: Handling Language-Specific Text Processing**

Now for something a bit more involved: imagine we need specific processing for a language like Spanish, perhaps removing accent marks before further steps. We will utilise the `unicodedata` library.

```python
import unicodedata

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.recipes.default_recipe import DefaultRecipe

@DefaultRecipe.register(component_name="accent_removal_component")
class AccentRemovalComponent(GraphComponent):
  @classmethod
  def create(
      cls,
      config: dict,
      model_storage: Resource,
      resource_config: Resource,
      execution_context: ExecutionContext,
  ) -> "AccentRemovalComponent":
    return cls()

  def process(self, messages, **kwargs):
    for message in messages:
      if message.get("text"):
        normalized_text = unicodedata.normalize("NFKD", message.text)
        message.text = "".join(char for char in normalized_text if not unicodedata.combining(char))
    return messages
```

In this case we utilise the `unicodedata` library to "decompose" any accented character into its base form and its combining accent char. We then discard the accent char, giving us the desired base form. The same idea as before applies, we are still directly modifying `message.text`. Once again, you'll need to add "accent_removal_component" to your `config.yml`.

It's crucial to consider how different preprocessing steps interact with each other. The order in which you arrange these components in your `config.yml` pipeline is essential. Lowercasing, punctuation removal, and accent removal typically come before tokenization and featurization. This is what I observed when I was building the multilingual chatbot; getting the order wrong significantly impacted the overall quality of intent and entity predictions.

Beyond the examples, here are a few resources that I've found invaluable when building custom components, I would highly recommend delving into these:

*   **"Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper:** This is an excellent foundational book covering the fundamental concepts and algorithms in NLP. It's a must-read if you're planning to do more than basic preprocessing.
*   **"Speech and Language Processing" by Daniel Jurafsky and James H. Martin:** A comprehensive textbook that goes into great detail about various NLP techniques. Although broader than the specific task of preprocessing, it offers a holistic understanding of the field, helping you grasp how preprocessing choices affect downstream tasks.
*  **Rasa documentation itself:** The official Rasa documentation provides comprehensive guidelines for creating custom components. It covers the necessary interfaces and best practices for integration into the Rasa framework. The Rasa forums are also an excellent place to look for examples and to discuss specific challenges with other users.

Creating custom components for text preprocessing in Rasa is a powerful technique, but it requires a methodical approach. Start small, experiment, and iterate. Be sure to thoroughly test each component in isolation and as part of the entire pipeline to observe any unintended interactions. Remember, clean and well-structured text is the foundation for accurate and reliable chatbot interactions.
