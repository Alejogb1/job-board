---
title: "How do I extract transition and emission scores from a Flair sequence tagger's CRF layer?"
date: "2024-12-23"
id: "how-do-i-extract-transition-and-emission-scores-from-a-flair-sequence-taggers-crf-layer"
---

Okay, let's talk about accessing those hidden CRF parameters within a flair sequence tagger. It's not always straightforward, but definitely achievable, and I've been in the weeds with this particular issue myself, more times than I'd like to recall. Several projects required me to not only use flair for sequence tagging but also to understand the inner workings of the conditional random field (crf) layer for advanced analysis and, let's be honest, some debugging.

The challenge essentially boils down to the fact that the `flair` library, while providing a high-level api, doesn't directly expose the crf parameters as public attributes. So, instead of attempting to 'pull them out,' we'll need to access the internals of the model and its components. This isn’t usually something you'd encounter day to day when using the model, but once you move to a deeper understanding, it becomes quite necessary.

First, a bit about *why* you might want these scores. The transition scores represent the probability (after passing through the activation function) of transitioning *from* one tag *to* another. Emission scores, on the other hand, are the probabilities (after passing through the activation function) of a particular word having a specific tag. These scores, combined through the crf, determine the final tag sequence probabilities, making understanding them crucial for interpreting why a certain sequence was chosen. For example, I had a project where we were trying to identify bias in a medical text tagging model. To analyze this, I needed the exact transition probabilities to see how the model was favoring certain tag sequences irrespective of the actual medical information.

Here's how we approach this. The key is to understand that the crf in flair is encapsulated within the `SequenceTagger` class. This class, during initialization, uses a `Model` which often contains a `CRF` layer object. That `CRF` object contains the transition parameters we are seeking. So, we navigate through these objects.

Let’s start with the emission scores. Remember, these aren't explicitly outputted; rather, they're computed as a result of the interaction between your word embeddings, any intermediate layers (like a lstm), and your linear layer that generates tag scores. To see the effect before the crf is applied, you have to run an input sequence through the model up to the linear layer. Here's some code demonstrating this:

```python
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings

# Load the pre-trained model. I'm just using a smaller model for the demo
# but feel free to swap this for larger models like 'ner-large' or similar.
tagger = SequenceTagger.load('ner-fast')

# Example sentence
sentence = Sentence("George went to New York.")

# Embed the sentence - note you should have embeddings matching the model
# In this case flair uses glove embedding by default with the 'ner-fast' tagger.
tagger.embed(sentence)

# Get the hidden states for the sentence using the forward method
hidden_states = tagger.model.forward(sentence.get_tensor(), sentence)

# Access the scores and then detach from computational graph.
scores = hidden_states.detach()

print(scores)
print(scores.shape)

#The output is usually of shape (batch size, sequence length, tag_size)
#where batch size is usually 1 if you provide just one sentence.
#Sequence length is the number of tokens.
#Tag_size is number of tags in your tag set.

# For example, you could print the emission score for "George" with respect
# to the 'PERSON' tag if it's included in the tagset.
# The index of a tag is defined by the vocabulary of the tagger itself
# usually it starts at index 0 for '<START>' and ends at the index N for '<STOP>'.
#  We'd have to work out 'PERSON' tag index by accessing the model vocab.
# For demonstration let's assume that the 'PERSON' tag is at index 3.
# So, assuming you have that information, you would get the emission for
# the word "George" and tag "PERSON" as follows:

# We can retrieve the word position with the sentence object.
word_idx = sentence.get_token_index(sentence[0]) #index of "George" token = 0

emission_score_george = scores[0, word_idx, 3].item()

print(f"Emission score for 'George' and tag 'PERSON': {emission_score_george:.4f}")
```

In this snippet, we first load a flair sequence tagger. We then feed it a sentence. After embedding it, we invoke the `model.forward` method. This method takes the sentence's embedding tensor and the original sentence. We detach the scores to remove it from the computation graph. The output is the pre-crf scores, and these are your emission scores for each word against each tag. You can then access individual scores based on the word and tag position. Note that the model’s vocabulary contains the information about the tag indices, you have to access the tagger.tag_dictionary object.

Now, let’s discuss those transition scores. They live inside the `CRF` layer itself. Again, it's not a direct attribute, but we can access it. The `CRF` layer has a `transitions` attribute which is a torch parameter containing the raw transition scores. We can get this through a similar process:

```python
import torch
from flair.models import SequenceTagger

# Load the pre-trained model (same as before)
tagger = SequenceTagger.load('ner-fast')

# Access the crf layer
crf_layer = tagger.model.crf

# Access the transition tensor.
transition_scores = crf_layer.transitions

print(transition_scores)
print(transition_scores.shape)

#The output of transition_scores is a tensor with size
#(number of tags, number of tags), i.e., a matrix

# For example, to get the transition score from the tag <START> to the tag <PERSON>
# we need to find the indexes of these tags.

# Let's assume that the '<START>' tag is the tag at index 0 and the 'PERSON' tag is the tag at index 3 (again from model's dictionary).

start_tag_idx = 0
person_tag_idx = 3
transition_start_to_person = transition_scores[start_tag_idx, person_tag_idx].item()

print(f"Transition score from '<START>' to 'PERSON': {transition_start_to_person:.4f}")

# Similarly, to get transition from 'PERSON' to the tag '<STOP>'.
# Let's assume that the '<STOP>' tag is at the last index of our model tags which we can obtain from the
# tag dictionary. Let's also assume that there are 10 tags in the model, so the stop tag will be at 9.

stop_tag_idx = 9
transition_person_to_stop = transition_scores[person_tag_idx, stop_tag_idx].item()

print(f"Transition score from 'PERSON' to '<STOP>': {transition_person_to_stop:.4f}")
```

Here, we obtain the tagger again. Then, we access its crf layer using `tagger.model.crf`. Finally, we can get the transition scores directly from the `transitions` attribute. This is a matrix, where the row index corresponds to the 'from' tag, and the column index corresponds to the 'to' tag. Again, you'd use the tagger's vocabulary to map indices to their respective tags to make use of this matrix.

One crucial note is that these transition scores are before the softmax activation during training, and are log probabilities at inference time, which you can also see in the `flair/nn/crf.py` file (specifically in the `_viterbi_decode` method) of the library. The activation function that would usually be applied to these raw scores can be found in the pytorch documentation as well. However, the raw scores, like the ones shown, are typically what one analyzes since they are not scaled to a probability distribution.

Finally, here's an example showing a different model structure, and how we access the embeddings within this model. It also demonstrates the need to check the structure of a model before attempting to access its internals.

```python
import torch
from flair.models import SequenceTagger
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

# Example using a different model structure with transformer embeddings.
tagger = SequenceTagger.load('ner')

# Example Sentence
sentence = Sentence("The cat sat on the mat.")

# Embed the sentence
tagger.embed(sentence)

# Check if model is sequence tagger
if hasattr(tagger, 'model'):
    print("This is a sequence tagger model.")
    model = tagger.model
    if hasattr(model, 'embeddings'):
        print("The sequence tagger has word embeddings.")
        embeddings = model.embeddings
        if isinstance(embeddings, TransformerWordEmbeddings):
             print("We are using Transformer Embeddings")
             embed_tensor = embeddings.forward(sentence)
             print(embed_tensor.shape)
        else:
            print("We are not using Transformer embeddings, we need to find the correct way to obtain embeddings.")

        if hasattr(model, 'crf'):
             print("We have a CRF layer. Transition matrix available to inspect")
             crf_layer = model.crf
             transition_scores = crf_layer.transitions
             print(transition_scores.shape)
    else:
        print("No embedding available directly")
else:
    print("This is not a sequence tagger model.")


# Remember, this is a generalized exploration, each model can be unique.
```

This snippet demonstrates that we have to handle cases where the embedding structure is different. For instance, it shows how to access the word embeddings from transformer-based models. This highlights the need to inspect the model’s internal structure, which can be different based on the specific flair model loaded. It also checks the type of embeddings, ensuring that we are applying the correct logic to retrieve the actual embeddings.

To delve deeper into these topics, I'd recommend checking out "Speech and Language Processing" by Daniel Jurafsky and James H. Martin, particularly the chapters on conditional random fields. Also, for a more detailed treatment of how transformers work, "Attention is All You Need," by Vaswani et al, is crucial to understand how embedding systems are made. The official PyTorch documentation, especially sections regarding neural networks and recurrent layers will be useful as well. Finally, exploring the flair source code on GitHub for the specific versions used for your model can help you understand how the various layers connect internally. In my experience, working through the core code provides invaluable insight into this kind of operation.

Accessing the crf layer parameters isn't a frequent task, but understanding *how* it is done provides valuable insight when facing tricky scenarios. These methods have been extremely helpful in projects where deeper introspection of the model was needed. Remember to always verify the specific details of the model you're using, as architectures can vary widely.
