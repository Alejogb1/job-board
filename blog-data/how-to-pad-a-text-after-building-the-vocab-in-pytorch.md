---
title: "How to pad a text after building the vocab in pytorch?"
date: "2024-12-14"
id: "how-to-pad-a-text-after-building-the-vocab-in-pytorch"
---

alright, so you're dealing with text padding after creating your vocabulary in pytorch, huh? i've been there, more times than i'd like to remember. it's a classic gotcha when you're starting to get serious with sequence data. let me share what i've learned over the years, and a couple of things that have saved my bacon (and my precious gpu time).

basically, you've built your vocabulary – you've turned all your words or tokens into unique integer ids. that's great. but when you're feeding sequences to your neural network, they need to be the same length. you can't just throw in variable-length sequences, at least not without some extra work (like using masking which is another beast for another time). that’s where padding comes in. we add extra tokens—usually a special token representing "padding"—to the end of shorter sequences until they match the length of the longest sequence in your batch.

i remember this one time, back when i was a fresh-faced grad student, i was building a simple text classifier. i had all my data ready, i had this beautifully crafted vocabulary, and then… nothing worked. the gradients were exploding, my loss was all over the place. it was like trying to build a house on quicksand. i spent a solid day, maybe more (the caffeinated haze of those days makes it hard to be precise), staring at my code, debugging the model, re-reading papers. it turned out i had completely overlooked padding. my sequences were all different lengths, and pytorch didn't like that one bit, so that was the root cause of the non convergence. after adding proper padding, things started behaving sanely, and i finally got to sleep. lessons learned. never underestimate the power of data preparation.

so, how do we do this in pytorch? it's pretty straightforward. let me show you the basic steps:

first, you'll have your sequences as a list of lists of integers. each inner list is a sequence of token ids.

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_sequences(sequences, pad_token):
    """
    pads a list of integer sequences to the max length of the sequence with the pad_token
    Args:
        sequences (list[list[int]]): list of sequences
        pad_token (int): the padding token id

    Returns:
       torch.Tensor: a tensor representing the padded sequences
    """
    tensor_sequences = [torch.tensor(seq) for seq in sequences]
    padded_sequences = pad_sequence(tensor_sequences, batch_first=True, padding_value=pad_token)
    return padded_sequences

# example usage:
sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
pad_token = 0  # lets assume 0 is our pad token
padded_tensor = pad_sequences(sequences, pad_token)
print(padded_tensor)

```

here, `pad_sequence` from `torch.nn.utils.rnn` is your friend. it takes a list of tensors (you have to convert the lists of integers to tensors first), and a padding token id. `batch_first=true` ensures that the batch dimension (the different sequences) comes first. the output is a tensor that has the shape `(batch_size, max_sequence_length)`. shorter sequences are padded with the specified `pad_token` until reaching the length of the longest sequence in the batch.

a lot of times, you'll be dealing with text that's already tokenized, but just in case, i'll throw in how you can do it from scratch using the `torchtext` library:

```python
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

def tokenize_text(text_data, tokenizer_type="basic_english"):
    """
    Tokenizes a list of strings into integer sequences
    Args:
        text_data (list[str]): list of strings to be tokenized
        tokenizer_type (str): type of tokenizer to use from torchtext

    Returns:
        tuple(list[list[int]], torchtext.vocab.Vocab): a tuple of the list of sequences and the vocab object

    """
    tokenizer = get_tokenizer(tokenizer_type)
    tokenized_data = [tokenizer(item) for item in text_data]
    return tokenized_data

def build_vocab(tokenized_data):
    """
    Builds a vocabulary object from a list of lists of tokens

    Args:
        tokenized_data (list[list[str]]): List of tokens sequences

    Returns:
        torchtext.vocab.Vocab: vocab object

    """
    vocab = build_vocab_from_iterator(tokenized_data, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def text_to_ids(tokenized_data, vocab):
    """
    Turns tokenized sequences into integer sequences

    Args:
        tokenized_data (list[list[str]]): List of tokens sequences
        vocab (torchtext.vocab.Vocab): vocab object

    Returns:
        list[list[int]]: list of integer sequences

    """
    ids = [[vocab[token] for token in item ] for item in tokenized_data]
    return ids


# example usage
text_data = ["this is a sentence", "another one", "short"]
tokenized_data = tokenize_text(text_data)
vocab = build_vocab(tokenized_data)
id_sequences = text_to_ids(tokenized_data,vocab)
pad_token = vocab["<pad>"]
padded_tensor = pad_sequences(id_sequences, pad_token)
print(padded_tensor)
```

here, i've broken it down into steps. we use `get_tokenizer` from `torchtext` to tokenize our text (we use a basic tokenizer here, but you can select others), then we build a vocab using `build_vocab_from_iterator` and the tokenized list. we add a `<unk>` and `<pad>` special tokens which we need. then we map each sequence to it's integer ids using our vocabulary.  finally, we use the same `pad_sequence` as before. this is how we achieve padding after building the vocab. the output should be a pytorch tensor.

one thing i have learned over the years is that how and when do you do it also impacts your training. sometimes, depending on the problem, you might not need it per batch but per dataset. and it will also depend if the padding is done pre-training or if it is dynamic on the fly per batch, which could also improve model training. but that depends entirely on the use case. let's do another example of dynamically padding sequences.

```python
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import random
class CustomDataset(Dataset):
    def __init__(self, data, vocab, pad_token):
        self.data = data
        self.vocab = vocab
        self.pad_token = pad_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        tensor_batch = [torch.tensor(seq) for seq in batch]
        padded_batch = pad_sequence(tensor_batch, batch_first=True, padding_value=self.pad_token)
        return padded_batch

def text_to_ids(tokenized_data, vocab):
    """
    Turns tokenized sequences into integer sequences

    Args:
        tokenized_data (list[list[str]]): List of tokens sequences
        vocab (torchtext.vocab.Vocab): vocab object

    Returns:
        list[list[int]]: list of integer sequences

    """
    ids = [[vocab[token] for token in item ] for item in tokenized_data]
    return ids

def tokenize_text(text_data, tokenizer_type="basic_english"):
    """
    Tokenizes a list of strings into integer sequences
    Args:
        text_data (list[str]): list of strings to be tokenized
        tokenizer_type (str): type of tokenizer to use from torchtext

    Returns:
        tuple(list[list[int]], torchtext.vocab.Vocab): a tuple of the list of sequences and the vocab object

    """
    tokenizer = get_tokenizer(tokenizer_type)
    tokenized_data = [tokenizer(item) for item in text_data]
    return tokenized_data

def build_vocab(tokenized_data):
    """
    Builds a vocabulary object from a list of lists of tokens

    Args:
        tokenized_data (list[list[str]]): List of tokens sequences

    Returns:
        torchtext.vocab.Vocab: vocab object

    """
    vocab = build_vocab_from_iterator(tokenized_data, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab



text_data = ["this is a sentence", "another one", "short one", "a very long one with lots of words here"]
tokenized_data = tokenize_text(text_data)
vocab = build_vocab(tokenized_data)
id_sequences = text_to_ids(tokenized_data, vocab)
pad_token = vocab["<pad>"]
dataset = CustomDataset(id_sequences,vocab,pad_token)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
for batch in dataloader:
    print(batch)
```
here we create a custom dataset and override the `collate_fn` method which provides a way to define how a batch is created from the data.  inside the `collate_fn`, we pad the data using the same `pad_sequence`. This is more efficient as we are not pre-padding the data. so that means each batch can be dynamically padded to its needed size and reduce computational load.

finally, and just so you don't end up staring at exploding gradients, pay attention to what token you're using as padding. it should not be a token that represents some real word in your text; otherwise, your model will become confused. and, as the final advice, never assume that your padding is correct, print it, look at it, visualize it. in summary, always double check.

for further reading, i recommend looking into the "attention is all you need" paper. it covers a lot on sequences and padding. another very good book is "natural language processing with pytorch". they explain a lot in details of all these concepts. also, always have on your bookmarks, the pytorch official documentation. it is your best friend for these situations.

i hope this helps. if you have any further questions just ask, i'm usually around when i am not staring at tensorboards. good luck, and remember, data preparation is half the battle.
