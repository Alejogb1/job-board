---
title: "How can I use .spacy training data in Python, now that it's no longer in JSON format?"
date: "2024-12-23"
id: "how-can-i-use-spacy-training-data-in-python-now-that-its-no-longer-in-json-format"
---

, let's tackle this head-on. I remember dealing with a similar transition myself a few years back when spaCy’s data format changed. Initially, it can seem like a hurdle, but it's actually a more streamlined and efficient approach once you understand the underlying principles. The shift away from the older json format was, in my opinion, a necessary evolution to handle the complexity and sheer volume of data involved in more intricate natural language processing tasks. Instead of being confined to a text-based, sometimes verbose, structure like json, we now work directly with spaCy's binary format, typically stored as `.spacy` files. This binary format is optimized for faster loading and processing, particularly beneficial when you’re dealing with large datasets.

The core idea remains the same though: we are still working with a collection of *example* objects for training, which describe the desired linguistic annotations for given textual inputs. The format change mainly impacts how we load, save, and potentially generate these examples. When I started, the json to .spacy migration required some adjustments in my workflow, but here's how I now typically handle the situation using Python:

Firstly, let's address the loading part. The primary function to load `.spacy` data is `spacy.tokens.DocBin.from_disk()`. This function reads a `.spacy` file from your filesystem and returns a `DocBin` object, which is essentially a container for your training `Doc` objects. Each `Doc` contains all the annotation data needed for training spaCy models. This binary format, in contrast to parsing json each time, provides improved read speeds and reduced memory consumption, which is especially important with larger models and datasets.

Here's a simple code snippet to demonstrate:

```python
import spacy
from spacy.tokens import DocBin

def load_spacy_data(filepath):
    """Loads a .spacy file from disk and returns the DocBin."""
    try:
        doc_bin = DocBin().from_disk(filepath)
        print(f"Successfully loaded data from: {filepath}")
        return doc_bin
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
      print(f"Error loading .spacy file: {e}")
      return None

# Example usage:
file_path = "my_training_data.spacy" # Replace with the actual path to your .spacy file
loaded_data = load_spacy_data(file_path)
if loaded_data:
  print(f"Number of documents: {len(loaded_data)}")
```

This function neatly encapsulates the process of loading the data, includes basic error handling for file not found issues, and will return a doc_bin object, or None if there is an issue. You could expand the error handling here to be more granular if your context requires it.

Now, once you have this `DocBin`, you need to convert it into an iterable of `Example` objects to work effectively with spaCy’s training pipeline. Example objects are pairings of a `Doc` and its corresponding annotations used to learn during training. This step is crucial, as spaCy’s `Trainer` class utilizes `Example` objects during the update cycle.

Here’s how you typically transform the loaded `DocBin` to `Example` objects:

```python
from spacy.training import Example

def create_examples(doc_bin, nlp):
  """Converts a DocBin to a list of Example objects."""
  examples = []
  for doc in doc_bin.get_docs(nlp.vocab):
    example = Example.from_dict(doc, {"words": [t.text for t in doc],
                                      "spaces": [t.whitespace_ for t in doc]})
    for token in doc:
        example.reference.set_token_attr("POS", [token.pos_ for token in doc]) #added POS tags
        if token.ent_type_:
             example.reference.set_ents([token.ent_type_ for token in doc]) #added entity tags
    examples.append(example)
  return examples

# Assuming we loaded the data and have a language model (nlp):
if loaded_data:
  nlp = spacy.blank("en") # This is for demonstration, use your model here
  examples = create_examples(loaded_data, nlp)
  print(f"Created {len(examples)} training examples.")
```

In this function, we iterate through the `Doc` objects in the `DocBin`. Each doc gets converted to an Example object, with the tokens and spaces extracted and added to the initial doc reference to be used during training, as well as pos and entity tags, if available. It's a relatively straightforward process, though you may want to customize the attribute handling inside the loop, such as the entity tags, depending on your specifics tasks.

Now, let’s say you want to programmatically create the `.spacy` files instead of relying solely on the `convert` utility from spaCy. This is useful when you are generating data on the fly, perhaps as a result of some data annotation activity or some intermediate step in your pipeline. The key is using the `DocBin`’s `add()` method followed by the `to_disk()` method. Here's a brief illustrative example:

```python
import spacy
from spacy.tokens import DocBin
from spacy.training import Example

def create_and_save_spacy_data(texts, filepath):
    """Creates a DocBin from a list of texts and saves to disk."""
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    for text in texts:
        doc = nlp(text)
        example = Example.from_dict(doc, {"words": [t.text for t in doc],
                                      "spaces": [t.whitespace_ for t in doc]})
        for token in doc:
            example.reference.set_token_attr("POS", ["NOUN" for token in doc]) #added POS tags
            example.reference.set_ents(["ORG" for token in doc]) #added entity tags, all "ORG" for demonstration
        doc_bin.add(example.reference)
    doc_bin.to_disk(filepath)
    print(f"Saved data to: {filepath}")


# Example usage:
training_texts = ["This is sentence one.", "Another sentence here."]
output_filepath = "generated_training_data.spacy"
create_and_save_spacy_data(training_texts, output_filepath)
```

Here, we generate spaCy doc objects from text, then add these to a `DocBin`, saving them into a `.spacy` file. Notice that I've added some basic annotations for demonstration purposes only, setting all the POS tags to "NOUN" and the entity tags to "ORG" - naturally you would need to adapt this to your annotation needs.

These snippets offer a practical overview of handling the `.spacy` format, covering loading, conversion to `Example` objects, and programmatically creating these files. This new format, while initially seeming like an obstacle, is now my preferred way of handling training data because it’s more efficient in handling large datasets, while also making our workflow more portable.

For deeper understanding, I recommend checking out "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper. While not directly focused on spaCy’s formats, it will strengthen your foundational knowledge. Also, spaCy's official documentation, particularly sections on training and data handling is essential and it's continually updated with the latest practices. In addition, the "Applied Text Analysis with Python" by Benjamin Bengfort, Rebecca Bilbro and Tony Ojeda provides an effective understanding of practical text processing, which includes discussions of training data formats in a professional context. These resources, combined with practice and experience, will help you navigate any data format changes that may come in the future.
