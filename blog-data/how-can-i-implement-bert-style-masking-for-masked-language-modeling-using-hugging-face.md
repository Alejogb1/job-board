---
title: "How can I implement BERT-style masking for masked language modeling using Hugging Face?"
date: "2024-12-23"
id: "how-can-i-implement-bert-style-masking-for-masked-language-modeling-using-hugging-face"
---

Okay, let's tackle this. I remember a project back in 2019, something involving multilingual text classification for a client in the translation industry – that's where I really got my hands dirty with implementing custom masking strategies for BERT. The standard masked language modeling (mlm) provided by Hugging Face's transformers library is excellent, but you often find yourself needing to tweak it. What I've found helpful is not just relying on the default settings; rather, understanding *how* the masking is happening at its core and then tailoring it.

Fundamentally, mlm within BERT (or its variants) involves randomly masking some tokens within a sequence and training the model to predict those masked tokens. This process encourages the model to learn bidirectional contextual representations, a key ingredient to BERT's success. Hugging Face’s transformers library handles most of the heavy lifting here, providing tools that make the process quite manageable. However, knowing the inner workings allows us to modify it effectively if needed.

The standard masking process generally involves randomly selecting a proportion of tokens, typically 15%, and then performing one of the following actions for each selected token: either replacing it with a `[MASK]` token (80% probability), replacing it with a random token from the vocabulary (10% probability), or leaving it unchanged (10% probability). This mix avoids the model overfitting to simply predict the masked token, encourages robustness, and is a standard practice.

Now, let’s talk about implementing this with Hugging Face. The core functionality resides within the `DataCollatorForLanguageModeling` class, specifically when preparing a batch for training. While you can't change the core algorithm directly, you can customize the way you feed it input. You would usually leverage a pre-trained tokenizer to perform the tokenization and then use the data collator to prepare the input batches for training.

Let me give you a couple of code examples that highlight how this works and, more importantly, how you can influence the process.

**Example 1: Basic MLM with a Custom Dataset**

First, let's create a basic setup where we have a custom dataset. Assume that we have a file called `my_text_file.txt` containing multiple lines of text, and we want to mask it for MLM training.

```python
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
      self.tokenizer = tokenizer
      self.texts = []
      with open(file_path, "r") as f:
          for line in f:
              self.texts.append(line.strip())
      self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encoding.items()}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = TextDataset("my_text_file.txt", tokenizer, max_length=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Example Batch Generation for demonstration:
batch = [dataset[i] for i in range(4)]
collated_batch = data_collator(batch)

print("Input IDs:\n", collated_batch['input_ids'][0])
print("Masked IDs:\n", collated_batch['labels'][0])

```

In this first example, we're using the standard collator, and the masking is applied using the default settings (80/10/10 probability scheme). The `TextDataset` prepares the input by tokenizing and padding each line. The `DataCollatorForLanguageModeling` then does the heavy lifting in terms of applying the masks. The `mlm_probability` parameter allows us to control the percentage of masked tokens; changing this modifies how many words will be masked per sentence.

**Example 2: Custom Masking Logic (Word-Level Masking)**

The second example gets a bit more complex. Suppose you're interested in masking whole words rather than random tokens. This might be useful for some languages or tasks where you believe semantic meaning is held more strongly at the word level. Here is how you could modify the data preparation stage to achieve this:

```python
import random
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch

class WordMaskingDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, mask_probability):
        self.tokenizer = tokenizer
        self.texts = []
        with open(file_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())
        self.max_length = max_length
        self.mask_probability = mask_probability


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()
        masked_tokens = []
        labels = []

        for token in tokens:
            rand_prob = random.random()
            if rand_prob < self.mask_probability:
                rand_val = random.random()
                if rand_val < 0.8:
                    masked_tokens.append("[MASK]")
                elif rand_val < 0.9:
                   masked_tokens.append(random.choice(self.tokenizer.vocab.keys()))
                else:
                    masked_tokens.append(token)
                labels.append(token) # Store the token we are replacing, for the label.
            else:
              masked_tokens.append(token)
              labels.append("")
        masked_text = " ".join(masked_tokens)
        labels_text = " ".join(labels)
        # Tokenize with return_tensors="pt" to get tensor output
        encoded_inputs = self.tokenizer(masked_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded_labels = self.tokenizer(labels_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        #remove padding in labels to set to -100 when used by the data_collator
        encoded_labels["input_ids"][encoded_labels["input_ids"] == self.tokenizer.pad_token_id] = -100


        return {
            "input_ids": encoded_inputs["input_ids"].squeeze(0),
            "attention_mask": encoded_inputs["attention_mask"].squeeze(0),
             "labels": encoded_labels["input_ids"].squeeze(0)
             }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = WordMaskingDataset("my_text_file.txt", tokenizer, max_length=128, mask_probability = 0.15)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # disable default mlm
# Example batch generation for demonstration:
batch = [dataset[i] for i in range(4)]
collated_batch = data_collator(batch)
print("Input IDs:\n", collated_batch['input_ids'][0])
print("Masked IDs (Labels):\n", collated_batch['labels'][0])

```

Here, the `WordMaskingDataset` class performs the masking logic on the *words* themselves before feeding the resulting strings into the tokenizer. We set `mlm=False` in the `DataCollatorForLanguageModeling` constructor, as the masking is already done.  Crucially, you must also change the labels to mask padding to -100. Otherwise the padding token will be treated as a mask target, and the model will not train correctly. You also must ensure that your input ids are properly masked before they arrive at the data collator.

**Example 3: Using a Custom Masking Function**

Finally, to generalize further, you can define a separate masking function and apply it within a dataset. This is useful when you have very specific masking requirements.

```python
import random
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch

def custom_masking(tokens, tokenizer, mask_probability):
    masked_tokens = []
    labels = []
    for token_id in tokens:
        rand_prob = random.random()
        if rand_prob < mask_probability:
            rand_val = random.random()
            if rand_val < 0.8:
                masked_tokens.append(tokenizer.mask_token_id)
            elif rand_val < 0.9:
                masked_tokens.append(random.choice(list(tokenizer.vocab.values())))
            else:
                masked_tokens.append(token_id)
            labels.append(token_id)
        else:
          masked_tokens.append(token_id)
          labels.append(-100)
    return masked_tokens, labels


class CustomMaskingDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, mask_probability):
      self.tokenizer = tokenizer
      self.texts = []
      with open(file_path, "r") as f:
          for line in f:
              self.texts.append(line.strip())
      self.max_length = max_length
      self.mask_probability = mask_probability

    def __len__(self):
      return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length = self.max_length)
        masked_tokens, labels = custom_masking(tokens, self.tokenizer, self.mask_probability)

        attention_mask = [1] * len(masked_tokens)
        padding_length = max(0,self.max_length - len(masked_tokens))
        masked_tokens += [self.tokenizer.pad_token_id] * padding_length
        labels += [-100] * padding_length
        attention_mask += [0] * padding_length

        return {
            "input_ids": torch.tensor(masked_tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = CustomMaskingDataset("my_text_file.txt", tokenizer, max_length=128, mask_probability=0.15)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) #disable mlm
# Example batch generation for demonstration:
batch = [dataset[i] for i in range(4)]
collated_batch = data_collator(batch)
print("Input IDs:\n", collated_batch['input_ids'][0])
print("Masked IDs (Labels):\n", collated_batch['labels'][0])


```

In this example, the `custom_masking` function operates on token ids. The dataset calls this function and prepares the batch with masked inputs and correct labels. As before, the data collator has its mlm functionality disabled, as the masking is performed by the dataset directly.

**In Summary**

The key to implementing custom masking with Hugging Face is to understand how the tokenizer and the `DataCollatorForLanguageModeling` work and how you can manipulate the input data to achieve the masking effects that you require. You can modify the dataset’s `__getitem__` method to apply custom masking logic or prepare labels as demonstrated above. Remember to always double-check the tokenization and masking results to confirm that your implementations are working as expected.

For more in-depth knowledge, I strongly recommend the original BERT paper, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” by Devlin et al. This will give you the foundational information about masking strategies. Additionally, thoroughly reviewing the Hugging Face transformers library’s documentation, particularly the `DataCollatorForLanguageModeling` class, is crucial. This should provide you with the required base information. Good luck and have fun experimenting!
