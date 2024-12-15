---
title: "Can anyone share an end-to-end text classification project using BERT?"
date: "2024-12-15"
id: "can-anyone-share-an-end-to-end-text-classification-project-using-bert"
---

well, i've been messing around with bert for quite a while now, and i get the itch to share some of my experiences, especially when it comes to the good ol' text classification challenge. it's a task i've faced more times than i care to count, from sentiment analysis on product reviews (a real headache, let me tell you) to topic categorization of technical documents. so, let me walk you through a typical end-to-end project, mostly from memory of projects i've done.

first things first, you're going to need data. and lots of it, if you want bert to actually learn something useful. let’s say you’ve got a bunch of text snippets and a corresponding category label for each one. this is how i typically format data for this kind of thing: a list of tuples, where each tuple contains the text itself and the category label it belongs to. something like this:

```python
data = [
    ("this movie was absolutely fantastic!", "positive"),
    ("i hated the acting, the worst ever.", "negative"),
    ("the service was acceptable, nothing special.", "neutral"),
    # ... and so on
]
```
notice the simple format. this is just the first step of pre-processing. it's crucial you have clean data here, otherwise, it's garbage in garbage out, as we all painfully know.

now, bert isn't going to magically understand raw text. we need to tokenize it, meaning converting the sentences into a sequence of numbers that bert can understand. for this, we use a tokenizer, specifically one associated with our choice of bert model. hugging face's `transformers` library is, in my opinion, the way to go here. it provides all the tools you need and much more.

here's how you can use it:

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(data):
    tokenized_inputs = []
    labels = []
    for text, label in data:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        tokenized_inputs.append(encoded_text)
        labels.append(label)
    return tokenized_inputs, labels

tokenized_inputs, labels = tokenize_data(data)
```
this code snippet uses bert-base-uncased. that’s a good starting point. it adds the `[cls]` token at the beginning and the `[sep]` token at the end (special tokens); it makes sure that the input sequence is either 128 tokens long via truncation or padding; and, returns tensors, not just numpy arrays. you will note that it also returns the attention masks which bert uses internally during training. these attention masks have to be included or bert will have a tough time figuring out which tokens are pads and which are meaningful. these masks tell the model which tokens are actual parts of the input sequence.

before we jump into the model, let's talk about those labels. bert takes numeric labels as input, not strings like "positive" or "negative". so we need to encode them.  a simple dictionary will do:
```python
label_mapping = {
    "positive": 0,
    "negative": 1,
    "neutral": 2
}

numeric_labels = [label_mapping[label] for label in labels]
```
this assigns each category a specific number. we will use these during training, and the model's output will also be these numbers. after this, it's time for the neural network setup.

now for the model itself, we'll use a bert model with a classification head. again, the `transformers` library is your friend.

```python
import torch
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_mapping)
)
```
this loads a pre-trained bert model and adds a classification layer on top. i've also specified the number of labels, which we need to match the number of categories we have.

training now comes. this is where the magic happens, but it's also where things get tricky, it will require a decent gpu. this piece is usually heavily coupled to the hardware it will run on. it may also have to be modified for specific tasks.

```python
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss

# first, pack up the data into a dataset and a dataloader
input_ids = torch.cat([d['input_ids'] for d in tokenized_inputs])
attention_masks = torch.cat([d['attention_mask'] for d in tokenized_inputs])
numeric_labels_tensor = torch.tensor(numeric_labels)

dataset = TensorDataset(input_ids, attention_masks, numeric_labels_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_function = CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3 # you might need more for a good training
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        batch_inputs = batch[0].to(device)
        batch_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} loss {loss.item()}")


```
this is more involved. we're defining our optimizer, adamw here; the loss function, which is cross-entropy loss for classification tasks; and finally a dataloader that will create batches of the data. this is usually a good practice to make sure your model does not run out of memory. you'll notice that we're moving our data to the gpu, which if available, significantly speeds up the training process. i usually also add a validation process after each epoch, but this would complicate the example a bit. also there are many other hyper parameters you can tweak for this training process, and this is a whole field to itself.

once the training is done, now is testing. the model should be able to predict the correct class of new pieces of text. here's how you test a text piece using a trained model:

```python
def predict(text, model, tokenizer, label_mapping, device):
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    model.eval()
    with torch.no_grad():
        output = model(**encoded_text)
        prediction = torch.argmax(output.logits, dim=1)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = reverse_label_mapping[prediction.item()]
    return predicted_label

new_text = "this product was amazing, i loved it."
predicted_label = predict(new_text, model, tokenizer, label_mapping, device)
print(f"prediction for '{new_text}': {predicted_label}")
```
i added a bit of reverse mapping so that you get the label name back instead of the index.  again, this process does not use any back propagation as you can see by using `torch.no_grad()` which effectively turns of the gradient calculations making the process faster. i think i did ok with this test.

some practical things that i have found when doing these kind of things, first, don't underestimate data pre-processing. clean text input is essential. i've spent weeks debugging models only to find out that the input data was bad. second, hyper parameter tuning is very important. there's a lot of papers on hyper-parameter tuning methods and they can save you a lot of time. also, always use a gpu if you have access to it. training a bert model on a cpu can take a very long time.

a couple of recommendations for the reading list:

*   the original bert paper, "bert: pre-training of deep bidirectional transformers for language understanding" it's quite detailed and the basis of this all.
*   "attention is all you need". this paper, among many other things, explains the attention mechanism that bert uses under the hood. its a bit theoretical, but important.
*   and, the hugging face `transformers` documentation. they provide excellent examples on how to use their library, which makes life a lot easier. and a good understanding of the pytorch documentation. sometimes the error messages you get, are best understood by just going directly to the core documentation.

lastly, one of my past training sessions once took so long that by the time it finished, i’d forgotten what i'd even trained it to do. i then had to retrace my steps, and read my own code, as if i was doing a code review. but, in the end it was all worth it, i guess.

so, yeah. that’s a text classification project with bert in a nutshell. i hope this helps.
