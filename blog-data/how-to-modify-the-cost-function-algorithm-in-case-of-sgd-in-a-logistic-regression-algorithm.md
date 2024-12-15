---
title: "How to modify the cost function algorithm in case of SGD in a logistic regression algorithm?"
date: "2024-12-15"
id: "how-to-modify-the-cost-function-algorithm-in-case-of-sgd-in-a-logistic-regression-algorithm"
---

hey there,

so you’re asking about tweaking the cost function for stochastic gradient descent (sgd) in logistic regression. i've been down that road a few times myself, and it's a pretty common spot where things get interesting. it’s not always a smooth sail when optimizing these models, and the cost function is really the heart of it all, the part that gets tweaked and modified so much. let’s jump in and i can share how i've approached this problem in the past.

first, let’s quickly recap the usual logistic regression cost function, the one we all know and love and sometimes struggle to get working correctly in time. typically, we use the cross-entropy loss:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15 #prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

```

this works pretty well. it penalizes the model when it makes incorrect predictions and pushes it in the direction of more accurate predictions. but there are situations where we might want to modify this. maybe our data is imbalanced, or maybe we have a specific way we want to penalize the model's mistakes. i once had to deal with a dataset where i had 99% of the data belonging to one class, and the usual cost function failed me miserably, but i will get into that.

now, let's get into some modification techniques for this cost function for sgd specifically. remember, sgd is the technique we use when dealing with massive datasets or when we want faster iterations, but it also brings in noise. so the cost function modifications we make should keep this in mind.

1.  **weighted cross-entropy**: this one's a classic if you have imbalanced data. if, say, your positive class is rare, the standard cross-entropy can be biased toward the negative class, which dominates the dataset. weighting each sample’s contribution to the cost can address this.

```python
def weighted_cross_entropy(y_true, y_pred, weights):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.mean(weights * (y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)))

```

where the 'weights' is a numpy array with a weight for each sample. i used this on a spam detection project a few years back, when the actual spam examples were a tiny percentage, and it really helped. the model suddenly started flagging spam with much more confidence. the key thing here was determining the correct weight per class, and this is more of an art than a science, but i started by giving the rare class more weight and iterating until i got a model that would work for my use case.

the reason i’m showing you the code snippets is not to make them copy and paste ready, but rather to show how the code modifications are simple and effective. the main idea is that you don’t need complicated calculus equations, but simple programming statements to change things up.

2.  **focal loss**: now, if you want to go a step further from the weighted loss, the focal loss is your guy. this function was made famous by object detection, and it's designed to focus more on hard-to-classify examples by down weighting easy examples. the intuition is that when the model is very confident it will make the right decision, we don’t want the cost to be as big as when the model is not sure of the decision it’s making, therefore, making it focus more on the examples it struggles with.

```python
def focal_loss(y_true, y_pred, gamma=2):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    return -np.mean((1-pt)**gamma * (y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)))

```

the `gamma` parameter here controls how much the easy examples are down-weighted. a higher value means that the model will really focus on the harder examples. in my experience this helped a lot, but the trade-off is that it will also make the model much harder to train with this cost function, as the changes are more subtle. it took me a week or two to converge to a good result with this approach. i have to say it was not easy to get the gradient to converge. and at the time it felt like a rollercoaster ride with its ups and downs while i tweaked the hyperparameters.

3.  **adding regularization terms**: this isn’t modifying the base cross-entropy directly, but it does modify the overall cost function. when using sgd it is common that the model overfits, and regularizing the cost function can help the model generalize well, so i like to add L1 or L2 regularization to the loss function.

```python
def l2_regularized_cross_entropy(y_true, y_pred, weights, lambda_reg):
  epsilon = 1e-15
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  cross_entropy = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
  l2_regularization = lambda_reg * np.sum(weights**2)
  return cross_entropy + l2_regularization

```

here, `lambda_reg` is the regularization strength parameter. i found this helpful particularly when dealing with high dimensional datasets with a lot of features, which can make the model more prone to overfitting, and by adding l2 regularization i usually got a much better result than without it. there is this funny thing that happened once where i forgot to add regularization in my code, and the model was all over the place, and for a couple of hours i was confused, thinking that i made some coding mistakes, and then i realize the mistake and laughed so hard, because in retrospect it was obvious what went wrong.

a few more things to remember while doing these modifications. one, it is usually a good idea to test these one at a time, and measure the effect they have on the cost function, and the model parameters. for example, if you add regularization, first make sure it works, then add the weighting function and so on. and finally remember that when making these changes, it changes the gradient computation (backpropagation) as well. in addition, be careful with the learning rate as the modifications to the cost function may change the optimal step-size of the gradient descent algorithm. this part can be a headache, but if you pay attention to each change you make, you will get it working.

now, for further reading, i always recommend that you read papers instead of blogs as they tend to have a more rigorous explanation. for instance, the original paper on focal loss is worth your time. it is called “focal loss for dense object detection” by tsung-yi lin, priya goyal, ross girshick, kaiming he, and piotr dollár, published in 2017. you can find it on arxiv. and for a deeper understanding of sgd itself, i recommend “optimization methods for large-scale machine learning” by léon bottou, frank e. curtis, and jorge nocedal, this is a great reference to see the theory behind sgd and cost function modification, you can find it in a very popular machine learning journal called siam review in 2018.

i hope this gives you a good starting point. modifying the cost function is both an art and a science. you will need to experiment and find what works best for your particular data and your goals. there is no one-size-fits-all solution here. and it also needs to be emphasized that sgd brings in noise, therefore, be prepared that the results might be more unstable than using batch gradient descent. if you have more questions, feel free to ask, that’s why we are here to help each other.
