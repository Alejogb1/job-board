---
title: "How can an NER model be used to extract symptom names?"
date: "2024-12-23"
id: "how-can-an-ner-model-be-used-to-extract-symptom-names"
---

Alright, let's talk symptom extraction using NER models. It's something I've spent a fair amount of time tackling, especially during a project involving medical records several years ago. The challenges were very real, and the need for a robust and accurate solution was critical. So, rather than starting with the usual textbook definition, let's approach it from the practical angle of someone who's actually been in the trenches.

First off, named entity recognition (ner) isn’t just about finding words that happen to be symptom names; it's about contextual understanding. It's the difference between identifying "headache" as a symptom and understanding when it's being used in a sentence like, "The patient complained of a severe headache". This requires the model to not just recognize the token "headache" but also to classify it within the appropriate category – a symptom in this case.

The fundamental approach involves training a model on annotated data. This means having a dataset where symptom terms are explicitly tagged. For instance, the phrase "patient experienced chest pain and nausea" would have "chest pain" and "nausea" tagged as 'symptom' entities. The more diverse and representative this dataset is, the better our model's generalization will be. We don’t want a model that only knows about headaches from textbook examples; we need one that knows about "stabbing pains in the left side" or the less formal "feeling a bit queasy."

Now, there are a few common model architectures that are suitable for this task. Traditional approaches involved conditional random fields (crfs), which are great at sequence labeling by modeling dependencies between labels. However, current best practices often leverage transformer-based models, such as bert or its variations, fine-tuned on the specific task of symptom extraction. These models are powerful because they capture contextual information effectively.

Let's illustrate this with some code snippets. I'll use python and a simplified version using the `transformers` library which is generally what you'd be using for bert-based models.

```python
# Example 1: Using a pre-trained model and tokenizer.
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

def extract_symptoms(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    ner_tags = [model.config.id2label[token] for token in predictions[0].tolist()]
    
    symptoms = []
    current_symptom = ""
    for token, tag in zip(tokens, ner_tags):
      if tag == 'B-LOC': # Replace 'B-LOC' with your symptom tag, i.e. 'B-SYMPT' or similiar 
        if current_symptom:
          symptoms.append(current_symptom.strip())
        current_symptom = token
      elif tag == 'I-LOC':
        current_symptom += f" {token}"
      elif current_symptom:
        symptoms.append(current_symptom.strip())
        current_symptom = ""
    if current_symptom:
      symptoms.append(current_symptom.strip())
    return symptoms

test_text = "Patient experienced severe headache and dizziness, along with some mild fatigue."
symptom_list = extract_symptoms(test_text)
print(f"Extracted symptoms: {symptom_list}")

```
Note: The example model here is a pre-trained general ner model, so you'll need to fine-tune it to get proper symptom extraction. In a real-world project, you would use something like 'B-SYMPT' instead of B-LOC and 'I-SYMPT' instead of I-LOC when labelling your training dataset, and would fine-tune that pre-trained model on the labeled data to recognize these tags. This code snippet uses a simple loop for identifying the beginnings (B-) and continuations (I-) of an entity, typical for BIO tagging schemes.

Now, let's delve into fine-tuning for improved performance, because the code example above won't be very accurate unless you tweak it. Fine-tuning a model with our own dataset enables it to learn the nuances of medical language and symptom descriptions specific to our target domain. Here's how that process might look:

```python
# Example 2: Fine-tuning a BERT model.
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

#Load the custom dataset. Ensure it is formatted correctly to follow the expected input format.
#Example dataset here with tokens and BIO tags
dataset = load_dataset("json", data_files={"train":"path_to_your_training_data.json", "validation":"path_to_your_validation_data.json"})

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3) # number of labels should include background tag 'O'.
label_names = ["O", "B-SYMPT", "I-SYMPT"]

def align_labels(labels, word_ids):
  new_labels = []
  last_word = None
  for word_id in word_ids:
    if word_id is None:
      new_labels.append(-100) #Special token padding value
    elif word_id != last_word:
      new_labels.append(labels[word_id])
    else:
      new_labels.append(labels[word_id])
    last_word = word_id
  return new_labels


def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=True)

  labels_list = []
  for i, labels in enumerate(examples['ner_tags']):
      word_ids = tokenized_inputs.word_ids(batch_index=i)
      new_labels = align_labels(labels, word_ids)
      labels_list.append(new_labels)

  tokenized_inputs['labels'] = labels_list
  return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)


training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer = tokenizer
)
trainer.train()
#Evaluate with the validation dataset.

```
In this example, the dataset is loaded as a "json" dataset with 'tokens' and 'ner_tags' for each text sequence. Labels are handled through token alignment which is important when using wordpiece tokenization that bert employs. The model is then fine-tuned with the trainer and saved to the output folder. Remember that you would still need to use similar processing steps as the first example to make predictions with this fine-tuned model.

Finally, let's touch on a critical aspect: post-processing. Often, ner models generate results that require additional steps to clean them or handle edge cases. For instance, you might need to combine tokens that form a multi-word symptom into a single entity like "chest pain". This involves a rule-based or statistical approach following the output of the model.

```python
# Example 3: Post-processing of extracted symptoms.

def post_process_symptoms(symptom_list):
    """
    Performs basic post processing of the symptoms. In a real
    project, this function might include normalization techniques to
    account for synonyms and other types of variations.
    """
    cleaned_symptoms = [symptom.lower().strip() for symptom in symptom_list]
    return cleaned_symptoms

test_symptoms = ['  Headache ', 'Dizziness ', 'Mild  fatigue']
processed_symptoms = post_process_symptoms(test_symptoms)
print(f"Processed symptoms {processed_symptoms}")

```
Post-processing includes basic cleaning here, but this could also encompass standardization using synonym lists and fuzzy matching, and might involve other domain-specific transformations.

To go deeper on the technical front, I recommend exploring the seminal papers on sequence labeling with crfs and the original bert paper to understand the foundations. For practical applications, "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper is a great foundational text, and for more advanced deep learning based models, research articles on using transformers and fine-tuning techniques for ner tasks would be beneficial. Additionally, focusing on papers that cover techniques for handling variations in medical terminology would also be very helpful.

In conclusion, while ner models are powerful tools for symptom extraction, their effectiveness hinges on the quality of the training data, careful model selection, precise fine-tuning, and robust post-processing. It's not a simple plug-and-play scenario; it's an iterative process of experimentation and refinement that requires a detailed understanding of both the model and the domain.
