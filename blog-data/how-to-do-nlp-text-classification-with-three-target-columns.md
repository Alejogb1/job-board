---
title: "How to do NLP text classification with three target columns?"
date: "2024-12-14"
id: "how-to-do-nlp-text-classification-with-three-target-columns"
---

alright, so you're looking at tackling a multi-label text classification problem with natural language processing, and specifically, you've got three target columns you're wanting to predict. i've been there, and it’s a common situation that pops up when you start doing text analysis with anything beyond simple sentiment. let me share my experience and how i've approached it in the past.

first off, it's crucial to understand this isn't your typical binary or multi-class classification problem. we're dealing with multiple labels per instance, which changes a few things in how we structure and evaluate our model. i recall my first time doing this, i was working on categorizing customer feedback for a startup. we had labels for ‘bug report,’ ‘feature request,’ and ‘general feedback,’ and a single comment could easily fall into multiple categories. i initially tried a one-vs-rest approach with a different classifier for each column but that led to inconsistencies and was frankly, a pain to manage and the computational overhead. it was not a very enjoyable time in the office i assure you.

the key here is thinking about each target column as a separate classification task that happens in parallel, though not necessarily independently. let’s break it down into stages.

**data preparation:**

this is always where the magic starts. your text data, i'm assuming, is in some tabular form. what we need to do is prep that text for our models. that involves:

1.  **text cleaning:** getting rid of things that could confuse the model - html tags, excessive whitespace, and those pesky special characters. usually, some regex magic will do. here’s a simple example in python using the `re` library:

    ```python
    import re

    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)  # remove html tags
        text = re.sub(r'\s+', ' ', text).strip() # remove excessive spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # keep alphanumerical chars
        return text.lower() # lowercase for uniformity

    example_text = "<p>This is <b> some text </b> with  extra   spaces!!</p>"
    cleaned_text = clean_text(example_text)
    print(cleaned_text)
    # output: this is some text with extra spaces
    ```

2.  **tokenization:** you'll need to break the text down into tokens, words or subwords. the standard in the nlp field is to use something like `nltk` or `spacy` for these tasks. but for a simpler task that is already pretty good, you could use the `transformers` library as well. here’s an example using the `transformers` library’s tokenizer:

    ```python
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text = "tokenization is a necessary step in nlp"
    tokens = tokenizer.tokenize(text)
    print(tokens)
    # output: ['token', '##ization', 'is', 'a', 'necessary', 'step', 'in', 'nl', '##p']
    ```
    notice that 'tokenization' is split into 'token' and '##ization' which is a frequent operation by bert tokenizers.

3.  **numericalization:** now we need to convert text tokens into numerical vectors that the models can ingest. that's where methods like tf-idf or word embeddings come into play. personally, i am a big fan of pre-trained transformer models for this part (the one above is already trained), like bert, roberta or similar. we can directly encode our text using these models.

    ```python
    from transformers import BertTokenizer, BertModel
    import torch

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    text = "this is an example sentence"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
      outputs = model(**inputs)
      embeddings = outputs.last_hidden_state.mean(dim=1)

    print(embeddings.shape)
    # output: torch.Size([1, 768])
    ```
    here, `embeddings` are a tensor containing the 768 dimensional vector representing the text. i'm averaging the output tokens to have one output per sentence.

4.  **label handling**: your target columns may be categorical (e.g., 'yes'/'no' or 'bug'/'feature'/'feedback'). convert these into numerical labels. if they're multilabel, where each instance can belong to multiple categories for each column, you'll probably use one-hot encoding or something similar. also, ensure that you handle missing labels in a consistent way, be it removing the examples or assigning them a specific label.

**model selection and training:**

now that you have your data prepped, its time to select a suitable model. since you have multiple target variables to predict, you should go with a model that can handle a multi-label classification task.

1.  **classifier choice:** you can use a transformer architecture as the base. for example, you can use `bert` and then add a separate linear layer (fully connected layer) for each target column. this makes the training faster since all the layers are shared. each linear layer would then have the same number of outputs as the number of distinct labels in that specific target column.

2.  **loss function:** for multilabel classification, binary cross-entropy is a good loss function. you want to evaluate each label separately so you should use binary cross entropy. for instance, let's imagine that the category `bug report` could be a 0 (not a bug) or a 1 (bug). same with 'feature request' or 'general feedback'. this allows a single text to have multiple active labels. if you're dealing with multi-class labels, like mutually exclusive categories, then categorical cross-entropy is the one to go.

3.  **training loop:** i like using the pytorch lightning framework, but any modern framework works. it simplifies training and allows you to specify different metrics for each column. i am assuming you know the standard training process. it will essentially involve passing your data to the model, calculating the loss using the labels and the outputs and updating the weights using some optimizer, like adam or similar.

4.  **evaluation metrics:** you'll need metrics that handle multilabel classification. precision, recall, f1-score, and hamming loss (or accuracy if multi-class) are very helpful here. calculate each metric for each target variable, which you could then average. it also good to know the values separately for debugging issues on a single column.

**testing and iteration:**

this is not a linear process, it is an iterative one. the accuracy you get in the first try is not usually great. i remember when i started with bert it took me a while to fine-tune the parameters. i tried everything, learning rates, optimizers, etc. you need to keep an eye on your metrics and identify which columns are underperforming. it is typical for one of the target variables to perform way worse than the other ones. it took me a while to understand it but my model was too biased towards some target variables. and i just needed more samples to make it work. that brings me to my last point:

**tips and tricks:**

*   **data is king:** i cannot stress this enough. high-quality, well-labeled data makes the difference. spending time cleaning and organizing your dataset will pay off exponentially in terms of model performance. if possible, try to get more labeled data for the categories that perform poorly, i swear it helps.
*   **hyperparameter tuning:** this is a tedious task. play with learning rates, batch sizes, and model parameters. it's time-consuming, but it can make a noticeable difference. i personally use optuna for this. its a python library that helps to automate the tuning process.
*   **cross-validation:** do proper cross-validation to validate the robustness of your models. this way you can ensure that your results are consistent and not random luck. if it works well on k-folds it is probably going to work well in production.
*   **monitor performance over time:** build in a system to continuously monitor performance. data drifts, and things change over time. models can degrade, so keep re-training on newer data. it took me a good amount of time to learn that one, that cost me some sleepless nights.

regarding resources, i’d recommend the following:

*   the ‘natural language processing with transformers’ book by lewis tunstall, leandro von werra, and thomas wolf. this should be in your shelf. its very useful to learn the basics and advance concepts on the subject. they also have a very good course that goes alongside the book.
*   the ‘speech and language processing’ book by daniel jurafsky and james h. martin is a very complete book that covers not only the deep learning aspect, but also traditional approaches. it is a classic book in the field.
*   the hugging face website (huggingface.co) and its docs for the `transformers` library. they keep a pretty good updated documentation.
*   and as a bonus, if you find that you have a lot of data, you might want to check the distributed training documentation in pytorch lightning. if that does not work try ray. it has always saved me from large-scale training times.

in my first project i remember training models that took more than a week and that cost me lots of money, then i learned to do all these things. and now it takes less than a day. a week? that's like a month in dog years in the model world.

that should give you a pretty good starting point. let me know if you have more specific questions.
