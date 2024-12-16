---
title: "How to write a custom Rasa component for text preprocessing?"
date: "2024-12-16"
id: "how-to-write-a-custom-rasa-component-for-text-preprocessing"
---

Okay, let's unpack this. Building custom components for Rasa is a task I’ve found myself revisiting many times over the years, especially when dealing with very specific text preprocessing needs that the standard pipelines don’t quite cover. It’s where the rubber meets the road, so to speak, in making your conversational AI truly tailored to your domain.

The core challenge, as I see it, is creating a modular, maintainable piece of code that fits seamlessly within Rasa's framework. We’re not just stringing together arbitrary functions; we're building a component that understands the Rasa input format, transforms it as needed, and then passes it along to the next stage of the pipeline. I've personally dealt with issues ranging from handling nuanced entity variations in product names to implementing highly specific stemming algorithms—each requiring a bespoke component.

First, let's talk structure. A Rasa component is essentially a python class that inherits from `rasa.nlu.components.Component`. This class must implement several methods, the most crucial of which are `__init__`, `train`, `process`, and `persist`. Optionally, you might also need `load` to restore previously trained components.

Here’s a breakdown of each:

*   `__init__`: This is where you set up any internal state that your component will require, as well as load any external resources like dictionaries or pre-trained models. For example, if you're implementing a custom lemmatizer, this is where you'd load your lemmatization model.
*   `train(training_data: TrainingData, config: RasaNLUModelConfig, **kwargs) -> None`: This is where the component learns or adapts to the training data. If your text preprocessing step is static, it might be an empty method, but if you are doing something that learns parameters, like a domain specific tokenizer, this is where it would happen.
*   `process(message: Message, **kwargs) -> None`: This is where the magic happens. You receive a `Message` object containing the input text and related metadata. Your component then modifies the message object (adding attributes to its `data` dictionary, like tokenized words or new entities).
*   `persist(file_name: Text, model_dir: Text) -> None`: Saves any state you want to retain between runs to a file on disk. This method is crucial to restore learned components or ensure that static resources are readily accessible to subsequent pipeline components.
*   `load(meta: Dict[Text, Any], model_dir: Text) -> Optional[Component]`: (Optional). This is where the persisted component's information is loaded back into memory.

Now, let’s get into some practical examples.

**Example 1: A simple case normalization component**

This component will convert the entire input text to lowercase, useful for ignoring case differences in further processing. Note that this particular task is very common, but illustrates the approach.

```python
from typing import Any, Dict, Text, Optional, List
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.message import Message


class CaseNormalizer(Component):
    name = "case_normalizer"
    provides = ["text"]
    requires = []  #No other component needed
    defaults = {}

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)

    def train(self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs) -> None:
        pass  #Nothing to train

    def process(self, message: Message, **kwargs) -> None:
        text = message.get("text")
        if text:
            message.set("text", text.lower())


    def persist(self, file_name: Text, model_dir: Text) -> None:
        pass  #Nothing to persist

    @classmethod
    def load(cls, meta: Dict[Text, Any], model_dir: Text) -> Optional[Component]:
       return cls()
```

In this simple example, our `CaseNormalizer` component only modifies the `text` attribute of the message, converting it to lowercase. This is straightforward but establishes the fundamental structure.

**Example 2: A component for removing punctuation**

Here, we will remove all punctuation characters using regular expressions. While NLTK and other libraries provide ready-made functions for punctuation removal, implementing it this way demonstrates how you could integrate custom logic.

```python
import re
from typing import Any, Dict, Text, Optional, List
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.message import Message


class PunctuationRemover(Component):
    name = "punctuation_remover"
    provides = ["text"]
    requires = []
    defaults = {}

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)
        self.punctuation_pattern = re.compile(r"[^\w\s]")

    def train(self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs) -> None:
        pass # Nothing to train

    def process(self, message: Message, **kwargs) -> None:
        text = message.get("text")
        if text:
            message.set("text", self.punctuation_pattern.sub('', text))

    def persist(self, file_name: Text, model_dir: Text) -> None:
        pass

    @classmethod
    def load(cls, meta: Dict[Text, Any], model_dir: Text) -> Optional[Component]:
        return cls()

```

Here, we initialize the regex pattern in `__init__` to remove any character that is not a word character (`\w`) or whitespace (`\s`). The `process` method applies this pattern to remove punctuation from the input message text.

**Example 3: A component for custom entity mapping**

This component showcases more practical use-case of mapping custom entity names. Suppose you have product name variations (e.g., "Pro 13", "Pro 13-inch", "13 inch Pro") which you’d want to normalize into a single entity value "MacBook Pro 13". This is a common issue I’ve faced when dealing with user inputs that are less structured.

```python
from typing import Any, Dict, Text, Optional, List
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.message import Message

class CustomEntityMapper(Component):
    name = "custom_entity_mapper"
    provides = ["entities"]
    requires = []
    defaults = {
        "entity_map": {
            "Pro 13": "MacBook Pro 13",
            "Pro 13-inch": "MacBook Pro 13",
            "13 inch Pro": "MacBook Pro 13",
            "Air 13": "MacBook Air 13"

            # etc.. you would add more custom mapping.
        }
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)
        self.entity_map = component_config.get("entity_map", self.defaults["entity_map"])


    def train(self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs) -> None:
        pass #Nothing to train

    def process(self, message: Message, **kwargs) -> None:
      entities = message.get("entities")
      if entities:
          for entity in entities:
              if entity.get("entity") == "product": # Assuming "product" entity was extracted elsewhere
                entity_value = entity.get("value")
                if entity_value in self.entity_map:
                  entity["value"] = self.entity_map[entity_value]


    def persist(self, file_name: Text, model_dir: Text) -> None:
         pass #Nothing to persist

    @classmethod
    def load(cls, meta: Dict[Text, Any], model_dir: Text) -> Optional[Component]:
        return cls(meta)
```

In this more involved example, the component iterates through the extracted entities, and if an entity is identified as a "product", it checks if its `value` matches one of the defined keys in the `entity_map`. If it matches, the `value` is replaced with the canonical name.

For further reference, I highly recommend reading the official Rasa documentation thoroughly and especially examining the structure of existing components in Rasa’s GitHub repository. There are also good resources available, including "Speech and Language Processing" by Daniel Jurafsky and James H. Martin, for deep understanding about natural language processing. The book “Foundations of Statistical Natural Language Processing” by Christopher D. Manning and Hinrich Schütze provides a more statistical view to these topics. These resources should deepen your understanding and aid you in developing advanced and tailored text preprocessing components. Also, keep an eye out for papers on specific text transformation techniques (e.g., stemming algorithms or specialized tokenizers) that fit your exact use case. This way, you can find a deeper context for your implementation details. Remember to ensure you test your custom components thoroughly before deploying them into a production environment, or you may find that your bot is misinterpreting crucial pieces of text input from users.
