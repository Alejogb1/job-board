---
title: "How can I resolve Spacy v3.x NER training errors related to finding an optimal move for supervision?"
date: "2025-01-30"
id: "how-can-i-resolve-spacy-v3x-ner-training"
---
In SpaCy v3.x, Named Entity Recognition (NER) model training failures often stem from issues in the way the model learns to align predicted entity spans with the ground truth annotations during backpropagation; specifically, a suboptimal 'move' is calculated, disrupting learning. I've encountered this problem extensively in my work fine-tuning SpaCy models for medical text processing, particularly with the challenges posed by nuanced entity boundaries. The issue isn't necessarily that the model is failing to *predict*, but rather that the *error* it is receiving isn't reflective of the true deviation from the desired outcome, leading to unstable and ultimately poor learning.

The core of the problem lies in how the `spacy-transformers` pipeline component, which underpins much of SpaCy's NER training in v3.x, calculates the loss function. It attempts to find the best "move" to adjust the predicted spans towards the correct annotated spans. This "move" determines which token spans to adjust, whether to extend, shrink, or move the predicted boundary. When the model struggles to find a consistent, optimal move, the loss signal fluctuates wildly, which means gradient updates don't reliably improve the model. This often results in validation loss not decreasing, or worse, plateauing at a relatively high value, and ultimately a model that performs poorly. This manifests not only as incorrect entity classifications, but also often as an inability to even detect the entities at all, or producing far too many, or overlapping, low-confidence predictions. These issues are exacerbated by inadequate training data, especially small datasets or those with ambiguous boundary annotations.

Here are a few techniques I've found effective in resolving such issues, along with illustrating code examples:

**1. Augmenting the Training Data:**

Insufficient training examples, especially edge cases with tricky boundary definition, can significantly hinder the model's ability to generalize. The solution is often not to gather a ton more raw text, but to transform your existing data intelligently using well thought-out augmentation strategies. I've found success with data augmentation methods specifically designed to target boundary definition. This is preferable to random word shuffling which can remove the meaningful structure.

```python
import spacy
from spacy.training import Example
from spacy.util import filter_spans
from typing import List, Tuple

def augment_boundary_example(example: Example, augment_percent: float = 0.2) -> List[Example]:
    """Augments an example by slightly adjusting entity boundaries."""

    augmented_examples = []
    doc = example.reference
    entities = example.reference.ents
    if not entities:
      return [example]

    for ent in entities:
        # Attempt to extend boundaries with probability
        if len(ent) > 1 and (random.random() < augment_percent):
            # try to extend from front or back
            front_or_back = random.choice([0,1])
            if front_or_back == 0: # Front
              if ent.start > 0:
                new_start = ent.start - 1
                new_end = ent.end
              else:
                new_start = ent.start
                new_end = ent.end + 1 if ent.end < len(doc) else ent.end
            else: #Back
               new_start = ent.start
               new_end = ent.end + 1 if ent.end < len(doc) else ent.end

            new_entities = [x for x in entities if x != ent]
            new_ent = doc.char_span(doc[new_start].idx, doc[new_end -1].idx + len(doc[new_end-1]), label=ent.label_)

            if new_ent is not None:
              new_entities.append(new_ent)
              new_entities = filter_spans(new_entities)
              example_new = example.copy()
              example_new.reference.ents = new_entities
              augmented_examples.append(example_new)

    return augmented_examples


nlp = spacy.blank("en") # Load base model if needed

def augment_training_data(train_data: List[Tuple[str, dict]], nlp, augment_percent: float = 0.2) -> List[Tuple[str, dict]]:
    augmented_train_data = []
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        new_examples = augment_boundary_example(example, augment_percent)

        #append original if none is created
        if not new_examples:
           augmented_train_data.append(example.to_tuple())
        else:
          for aug_ex in new_examples:
            augmented_train_data.append(aug_ex.to_tuple())
    return augmented_train_data


if __name__ == "__main__":
   import random
   train_data = [
        ("The patient took 500mg of ibuprofen.", {"entities": [(17, 26, "DRUG")]}),
        ("He had a headache with mild fever.", {"entities": [(10, 18, "SYMPTOM")]})
    ]
    augmented_data = augment_training_data(train_data, nlp)
    for text, annots in augmented_data:
        print(text)
        print(annots)
```

This code first defines a function `augment_boundary_example` that, given an example, tries to add boundary examples by slightly extending the entities using random probability. Then the `augment_training_data` function applies `augment_boundary_example` to all examples within your dataset. Notice that I specifically consider not just adding new annotations but adjusting existing boundary edges. The output of this script shows the slightly altered entities in the augmented examples. I've found this technique particularly helpful when dealing with named entities that often have optional modifiers, for example, where the training data might label "mild headache" but not "headache," in which case, my model struggles to detect "headache" on its own.

**2. Adjusting the Span Prediction Loss:**

In SpaCy, the loss calculation involves the calculation of a span-based loss which is directly involved in calculating the error in each forward pass. If the loss is not calculated effectively, this can directly affect the model optimization step. We can control the loss more directly by influencing the parameters passed to the pipeline.

```python
import spacy
from spacy.util import decaying

def create_nlp_config():
  """Function that builds config."""
  config = {
      "nlp": {"lang": "en", "pipeline": ["transformer", "ner"]},
      "components": {
          "transformer": {
              "model": {"name": "bert-base-uncased"}
          },
          "ner":{
                "model": {
                 "@architectures": "spacy.TransitionBasedParser.v2",
                 "moves": None,
                 "update_with_oracle_cut_size": 2,
                 "scorer": {"@scorers":"spacy.ner_scorer.v1"},
                 "nO": 2, # This will determine the number of outputs (e.g. entity categories)

                  },
               "config": {
                  "learn_tokens": True,
                  "use_gold_spans": True, # Very Important to use the gold standard
                  "spans_key": "sc",
               }
           }
      },
      "training": {
        "seed": 0,
        "dropout": 0.2,
      "batcher": {
            "@batchers": "spacy.batch_by_padded.v1",
            "discard_oversize": True,
            "buffer": 1000,
            "max_words": 1000,
          },
         "optimizer": {
            "@optimizers": "spacy.Adam.v1",
            "learn_rate": 0.00002,
            "beta1": 0.9,
            "beta2": 0.999,
            "use_averages": True,
            "L2": 0.01,
            "grad_clip": 0.0,
            },
        "max_epochs": 2000,
        "patience": 50,
        "eval_frequency": 100,
        "checkpoints": "checkpoints",
        "dev_corpus": "corpus/dev.spacy",
        "train_corpus": "corpus/train.spacy"
      },
  }
  return config

if __name__ == "__main__":
    import os, json
    config = create_nlp_config()
    cfg_path = "config.cfg"

    with open(cfg_path, "w") as f:
      json.dump(config, f)

    os.system(f"python -m spacy train {cfg_path} --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy")

```

This code block focuses on adjusting the training configuration file to better guide the loss calculation. Crucially, within the NER component's `config` section, I set `use_gold_spans` to `True`, meaning that we're explicitly giving the system the correct token spans and forcing the learning to optimize around it, rather than letting the model have free reign. This significantly increases the stability of training, and is an important step when facing a problem such as this, especially when dealing with subtle or overlapping entities. Additionally, in `training.optimizer` section, I recommend a fine-tuned learning rate and L2 penalty, as these parameters can significantly influence the model optimization. It is crucial to save this configuration in a `.cfg` file and use `spacy train` to run the training using this configuration.

**3. Adjusting the move_size parameter:**

The root of the 'optimal move' problem lies in how SpaCy calculates the best 'move' during each backpropagation step. This move determines how the model adjusts the predicted spans toward the gold annotations. SpaCy uses a greedy strategy when adjusting token spans. Sometimes this strategy might be too restrictive or too generous based on the data. The size of the span adjustments are controlled by the `update_with_oracle_cut_size` parameter.

```python
import spacy
from spacy.util import decaying

def create_nlp_config_2():
  """Function that builds config."""
  config = {
      "nlp": {"lang": "en", "pipeline": ["transformer", "ner"]},
      "components": {
          "transformer": {
              "model": {"name": "bert-base-uncased"}
          },
          "ner":{
                "model": {
                 "@architectures": "spacy.TransitionBasedParser.v2",
                 "moves": None,
                 "update_with_oracle_cut_size": 4, # <--------- THIS HAS BEEN CHANGED
                 "scorer": {"@scorers":"spacy.ner_scorer.v1"},
                 "nO": 2, # This will determine the number of outputs (e.g. entity categories)

                  },
               "config": {
                  "learn_tokens": True,
                  "use_gold_spans": True, # Very Important to use the gold standard
                  "spans_key": "sc",
               }
           }
      },
      "training": {
        "seed": 0,
        "dropout": 0.2,
      "batcher": {
            "@batchers": "spacy.batch_by_padded.v1",
            "discard_oversize": True,
            "buffer": 1000,
            "max_words": 1000,
          },
         "optimizer": {
            "@optimizers": "spacy.Adam.v1",
            "learn_rate": 0.00002,
            "beta1": 0.9,
            "beta2": 0.999,
            "use_averages": True,
            "L2": 0.01,
            "grad_clip": 0.0,
            },
        "max_epochs": 2000,
        "patience": 50,
        "eval_frequency": 100,
        "checkpoints": "checkpoints",
        "dev_corpus": "corpus/dev.spacy",
        "train_corpus": "corpus/train.spacy"
      },
  }
  return config

if __name__ == "__main__":
    import os, json
    config = create_nlp_config_2()
    cfg_path = "config2.cfg"

    with open(cfg_path, "w") as f:
      json.dump(config, f)

    os.system(f"python -m spacy train {cfg_path} --output ./output2 --paths.train ./train.spacy --paths.dev ./dev.spacy")
```

This code block demonstrates changing a key parameter to influence the 'optimal move' calculation: `update_with_oracle_cut_size`. I've changed this from 2 to 4. When this parameter is small, the greedy strategy works on small adjustments to the span, which may make it difficult to train if large corrections are needed. On the other hand, if the parameter is large, the greedy strategy might over-adjust. This number typically should be small, and not too large, as overly generous moves can lead to incorrect and unstable predictions. The correct adjustment size should be found experimentally as it depends greatly on the size of your entities, the complexity of your dataset and the specific text being processed. I have found the need to experiment with this value a few times to find the optimum.

In addition to the above strategies, it is always worthwhile to ensure your data annotation standards are consistent and clear; review your annotations if problems persist. When troubleshooting similar challenges, reviewing the SpaCy documentation for the `spacy-transformers` component and associated training parameters is essential. Additionally, examining the source code of the `TransitionBasedParser` can provide insights into the mechanics of loss calculations. Publications related to sequence labeling and specifically on how to optimize models for long span sequences can also prove very useful when designing an optimal training pipeline. These resources are more helpful than vague general recommendations that often fall short when dealing with nuanced problems such as these.
