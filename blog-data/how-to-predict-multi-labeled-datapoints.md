---
title: "How to predict (multi) labeled datapoints?"
date: "2024-12-14"
id: "how-to-predict-multi-labeled-datapoints"
---

alright, so you're tackling multi-label classification, huh? been there, done that, got the t-shirt (and probably a few more from late nights debugging). it's not as straightforward as your standard single-label stuff, but it's definitely a solvable problem. i remember back in the day, trying to build a system to automatically tag articles for an online news platform. ended up going through a whole bunch of methods before landing on something that worked halfway decently, haha.

first off, the big difference with multi-label is that each datapoint can belong to multiple categories simultaneously, not just one. like a movie could be tagged 'action', 'sci-fi', and 'thriller' all at once. this changes how you think about prediction and evaluation compared to regular classification.

let's talk methods. you've got a few different routes you can take, depending on the complexity you need and the amount of data you're dealing with. the most straightforward, and a pretty solid starting point, is what i like to call "binary relevance." it's simple: for each label you want to predict, you train a separate binary classifier. you treat each label as an independent classification problem, basically just asking “is this datapoint part of this label or not?" the result is a set of yes/no predictions for each label.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# let's simulate some data
np.random.seed(42)
num_samples = 1000
num_features = 20
num_labels = 5
X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 2, size=(num_samples, num_labels))


# binarize y for multi-label
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifiers = {}
for i in range(num_labels):
    classifiers[i] = LogisticRegression(solver='liblinear', random_state=42)
    classifiers[i].fit(X_train, y_train[:,i])


y_pred = np.zeros_like(y_test)
for i in range(num_labels):
    y_pred[:, i] = classifiers[i].predict(X_test)

# evaluate:
accuracy = accuracy_score(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Hamming loss: {hamming:.4f}")
```

this code gives you the basic idea. notice that i'm using `MultiLabelBinarizer` to transform y into a format the classifiers can handle. i've also included a simple accuracy and hamming loss metric, which are common for this type of setup. don't go overboard with them initially, but understanding them is important. it’s generally accepted that hamming loss is a more suitable metric.

binary relevance is straightforward to implement and it can work reasonably well when the labels are relatively independent. but, here's the catch, it completely ignores label dependencies. like, if you predict 'action', it might be more likely you should also predict 'adventure', but the binary relevance model doesn't know about that relationship.

so, what next? one way to address this is with "classifier chains." the idea here is to create a chain of classifiers, where each classifier predicts a label, but also takes the predictions of the previous classifiers in the chain as input. the order of classifiers can actually matter a bit here, so think about which labels might depend on others.

```python
from sklearn.ensemble import RandomForestClassifier


classifiers = {}
y_pred_chain = np.zeros_like(y_test)
for i in range(num_labels):
    if i == 0:
        classifiers[i] = RandomForestClassifier(random_state=42)
        classifiers[i].fit(X_train, y_train[:, i])
        y_pred_chain[:, i] = classifiers[i].predict(X_test)

    else:
        input_features = np.concatenate((X_train, y_train[:, :i]), axis=1)
        input_features_test = np.concatenate((X_test, y_pred_chain[:, :i]), axis=1)
        classifiers[i] = RandomForestClassifier(random_state=42)
        classifiers[i].fit(input_features, y_train[:, i])
        y_pred_chain[:, i] = classifiers[i].predict(input_features_test)



accuracy = accuracy_score(y_test, y_pred_chain)
hamming = hamming_loss(y_test, y_pred_chain)
print(f"Accuracy: {accuracy:.4f}")
print(f"Hamming loss: {hamming:.4f}")
```

here, instead of logistic regression, i'm using random forests, just for the sake of demonstration. the key difference is in the loop, where each classifier gets the original features plus the predictions of previous classifiers. it can improve performance over binary relevance, but you may see issues if you have a wrong prediction early in the chain, which propagates to the subsequent ones. also, the order of the chain can impact the outcome.

if you have very high dimensional data and/or a lot of labels, you might consider "label embedding" methods. the core concept here is to project both the datapoints and the labels into a lower-dimensional space, where similar data points and labels are located close to each other. a popular model for this is `fasttext`, which can give you great results if you have a text based problem, though there are models that cover more input types.

```python
from sklearn.model_selection import train_test_split
import numpy as np
import fasttext

# simulate
np.random.seed(42)
num_samples = 1000
num_features = 20
num_labels = 5
X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 2, size=(num_samples, num_labels))

mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y)

# convert data to the correct format
X_texts = [' '.join(map(str, row)) for row in X]
y_texts = [', '.join(mlb.inverse_transform(np.array([row]))[0]) for row in y_binary]

# create the dataset
with open("train.txt", "w") as f:
    for i in range(len(X_texts)):
        f.write(f"__label__{y_texts[i]} {X_texts[i]}\n")

# train the model
model = fasttext.train_supervised(input="train.txt")

# function to predict labels (just example of prediction)
def predict_labels(input_text):
    labels = model.predict(input_text, k = 5) #max 5 prediction
    return labels[0]

# for demonstration i will use the first element of x as an example:
print (predict_labels(X_texts[0]))
```

i've used a dummy dataset and fasttext in this snippet. the steps are: 1) convert your data into text format, that is: one string for the input and one string of labels separated by commas 2) training the fasttext model using that data. 3) make predictions using it.

now, fasttext is pretty fast and memory efficient, but it works better with text based data. you can adapt this if you're dealing with other kind of features, you just need to encode them into a string format or represent them in a way that the fasttext library can handle.

remember, the right method depends on your data and the specifics of your problem. it's common to start with simpler ones like binary relevance, and only if you need to squeeze more performance you move to the more complex methods like classifier chains or embedding ones. and even then, don't discount the good old art of carefully engineering features before going with the big guns, you might be surprised. i've spent some time trying to optimize a model once, to realize that the problem was not the model itself, but some data preprocessing steps and the problem ended up being resolved just by cleaning data properly. as they say, garbage in, garbage out.

when it comes to resources, i'd recommend looking into the "statistical learning" by hastie, tibshirani and friedman. it's a bit dense, but it's a solid theoretical foundation for understanding most of these techniques. also, "pattern recognition and machine learning" by christopher bishop is another good text. there are lots of resources online too, but make sure you stick to reputable authors and papers. i've seen a lot of misinformation in random blog posts. i mean, how did the chicken cross the road? just ask google maps, they have the best directions.

and finally, don't be afraid to experiment and try things out. and as with all data science problems, your understanding and experience will improve over time with a lot of hands on experience.
