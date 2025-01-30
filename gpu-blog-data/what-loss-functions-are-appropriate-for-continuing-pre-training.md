---
title: "What loss functions are appropriate for continuing pre-training large language models with Hugging Face?"
date: "2025-01-30"
id: "what-loss-functions-are-appropriate-for-continuing-pre-training"
---
Pre-training large language models (LLMs), even those that have undergone extensive initial training, often benefits from further adaptation to specific domains or tasks. The choice of loss function during this *continued pre-training* phase is paramount, directly influencing the model's performance on downstream applications. I've observed through numerous fine-tuning projects that simply reusing the original pre-training objective isn't always optimal; context and objectives may have shifted. The goal here is not to reinvent pre-training, but to gently nudge the model in the desired direction using task-specific or domain-relevant data.

Fundamentally, the loss function guides the model's learning process by quantifying the discrepancy between predicted outputs and ground truth data. In the context of continued pre-training, the absence of specific task labels necessitates reliance on unsupervised or self-supervised objectives. This is distinct from fine-tuning where labeled data is available. The core principle here is to encourage the model to better understand the statistical patterns and relationships within the new corpus, essentially allowing it to adapt its representation space without straying too far from its learned knowledge.

Three commonly effective loss functions for continued pre-training include Masked Language Modeling (MLM), Next Sentence Prediction (NSP), and a custom-designed Sentence Reconstruction loss tailored to the task specifics. It is crucial to recognize that the suitability of each depends heavily on the characteristic of the continued pre-training data.

**1. Masked Language Modeling (MLM)**

MLM, initially employed in models like BERT, is a strong candidate for continued pre-training. Here, a percentage of tokens within a sequence are randomly masked, and the model's task is to predict the masked tokens based on the context of the remaining unmasked tokens. This approach forces the model to rely on contextual understanding of the input sequence.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

class CustomTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.encodings['input_ids'][idx]), 
               'attention_mask': torch.tensor(self.encodings['attention_mask'][idx])}
        
    def __len__(self):
         return len(self.encodings['input_ids'])


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.rand(inputs.shape)
    mask = probability_matrix < mlm_probability
    labels[~mask] = -100  # Ignore index for loss calc

    inputs[mask] = tokenizer.mask_token_id

    return inputs, labels


def custom_collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  attention_masks = [item['attention_mask'] for item in batch]

  input_ids = torch.stack(input_ids)
  attention_masks = torch.stack(attention_masks)

  masked_inputs, masked_labels = mask_tokens(input_ids, tokenizer)

  return {"input_ids": masked_inputs, "attention_mask": attention_masks, "labels": masked_labels}

#Assume text_list is a list of text strings for pretraining
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

encodings = tokenizer(text_list, truncation=True, padding=True, return_tensors='np')
dataset = CustomTextDataset(encodings)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn = custom_collate_fn )


training_args = TrainingArguments(
    output_dir='./output_dir',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator = custom_collate_fn
)


trainer.train()

```

*Code Commentary:*
The example utilizes the `transformers` library, specifically the BERT architecture. I implemented a `CustomTextDataset` to facilitate the formatting of text data into tensors and a `custom_collate_fn` to apply masking. A `Trainer` object then manages the training process based on specified arguments, where `prediction_loss_only` set to True will only return the masked language modeling loss. This is the default when training a masked language model. The custom collate function manages masking via the `mask_tokens` method, setting masked token ids and labels. Masked tokens are replaced by the `mask_token_id` and the corresponding label for the non-masked tokens are set to -100 which is ignored by the loss function, thus, only the masked tokens contribute to the loss value. This approach encourages the model to maintain its language understanding capabilities. In my experience, using the original pre-training loss can reduce overfitting issues in domain-specific continuous pre-training.

**2. Next Sentence Prediction (NSP)**

NSP is another pre-training objective that is often used in conjunction with MLM. It requires the model to predict whether two given sentences are consecutive in the original text or not. While I've found it's effectiveness is variable, depending on the dataset, it can encourage better understanding of inter-sentence relationships and discourse structure.

```python
from transformers import AutoModelForNextSentencePrediction, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import random

class NSPDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.examples = []
        for i in range(0, len(sentences)-1):
            sentence_a = sentences[i]
            
            if random.random() > 0.5:
                sentence_b = sentences[i+1]
                self.examples.append((sentence_a, sentence_b, 0))
            else:
                 sentence_b = random.choice(sentences)
                 if sentence_b == sentence_a:
                     continue
                 self.examples.append((sentence_a, sentence_b, 1))
                 
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        sentence_a, sentence_b, label = self.examples[idx]
        encodings = self.tokenizer(sentence_a, sentence_b, truncation=True, padding=True, return_tensors='pt')
        return {"input_ids": encodings["input_ids"].squeeze(), "attention_mask": encodings["attention_mask"].squeeze(), "labels": torch.tensor(label)}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForNextSentencePrediction.from_pretrained("bert-base-uncased")

#Assume sentences is a list of sentences from the pretraining data
dataset = NSPDataset(sentences, tokenizer)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

training_args = TrainingArguments(
    output_dir='./output_dir',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

*Code Commentary:*
I’ve constructed a `NSPDataset` which creates pairs of sentences, where the pair is either subsequent sentences from the text or randomly selected unrelated sentences and are labeled accordingly.  The `Trainer` facilitates the next sentence prediction training by calculating cross-entropy loss between the model's predicted label and the ground truth labels. This explicit training signal encourages the model to encode not only single sentence meaning but also to understand the logical flow from one sentence to the next. I’ve observed improvements in tasks involving logical reasoning and paragraph level understanding through NSP, though the degree of such improvements remains heavily dataset-dependent.

**3. Custom Sentence Reconstruction Loss**

Sometimes standard objectives are not sufficient for very specific application scenarios.  In such cases, crafting a bespoke loss function may prove valuable. A sentence reconstruction loss, for instance, can emphasize sentence-level semantics. I've employed variants where I've perturbed the sentence by randomly deleting or re-ordering words, then training the model to reconstruct the original sequence from the perturbed version. This custom objective is advantageous in domains where data may be inherently noisy or imperfect.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import random

class SentenceReconstructionDataset(Dataset):
    def __init__(self, sentences, tokenizer, perturbation_probability = 0.2):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.perturbation_probability = perturbation_probability

    def perturb_sentence(self, sentence):
       tokens = sentence.split()
       perturbed_tokens = []
       for token in tokens:
           if random.random() > self.perturbation_probability:
                perturbed_tokens.append(token)
       random.shuffle(perturbed_tokens)
       return " ".join(perturbed_tokens)
           
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
      
        original_sentence = self.sentences[idx]
        perturbed_sentence = self.perturb_sentence(original_sentence)
        
        encodings_perturbed = self.tokenizer(perturbed_sentence, truncation=True, padding=True, return_tensors="pt")
        encodings_original = self.tokenizer(original_sentence, truncation=True, padding=True, return_tensors="pt")

        return {"input_ids": encodings_perturbed["input_ids"].squeeze(), "attention_mask": encodings_perturbed["attention_mask"].squeeze(), "labels": encodings_original["input_ids"].squeeze() }

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

#Assume sentences is a list of sentences from the pretraining data
dataset = SentenceReconstructionDataset(sentences, tokenizer)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

training_args = TrainingArguments(
    output_dir='./output_dir',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

```

*Code Commentary:*
This example makes use of the T5 architecture, a seq2seq model appropriate for sequence generation tasks. The `SentenceReconstructionDataset` randomly reorders the input sentences as the ‘perturbed_sentence’ and treats the original input sentence as the target. This setup encourages the model to learn how to reconstruct and thus to better represent the true meaning of the sequence. This is implemented by the Trainer object calculating the cross entropy loss of the reconstructed and target sequences. In some domain specific NLP tasks, this loss can greatly assist downstream performance. It must be noted that T5 is being used here for illustration as an example of the Seq2Seq paradigm; this is not necessarily the optimal model for such a task. The effectiveness of this approach depends heavily on the nature of the perturbations applied.

**Resource Recommendations:**

For a deeper understanding of these concepts, I recommend exploring publications covering:
1.  **Transformer architectures:** Concentrate on models such as BERT, RoBERTa, and T5 and their associated pre-training objectives.
2.  **Unsupervised and self-supervised learning:** Gain an understanding of the broader techniques involved in learning without labeled data, especially for language representation.
3.  **Pre-training strategies:** Seek in-depth studies focusing on continued pre-training, domain adaptation, and transfer learning techniques for language models.

A solid grasp of these areas, combined with hands-on experimentation, should allow for a more nuanced understanding of choosing appropriate loss functions for continued pre-training. Careful selection of objectives based on the training data and the target task will significantly impact the success of your LLM adaptation.
