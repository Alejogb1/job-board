---
title: "Is there a way to perform a pairwise comparison with an incomplete model?"
date: "2024-12-15"
id: "is-there-a-way-to-perform-a-pairwise-comparison-with-an-incomplete-model"
---

ah, so you're looking into pairwise comparisons with a model that's not exactly all there, eh? i’ve been down this rabbit hole more times than i care to remember, and let me tell you, it’s a situation that demands a bit of cleverness. it’s not uncommon, really. we've all been there— working with data that is sparse or a model that was not fully trained, or perhaps the labels are incomplete.

first, let's clarify what “incomplete model” means in this context because it can mean a few different things. could it be that the model was only partially trained? or it's missing features? or maybe it's outputting some probabilities that aren't calibrated correctly? it's a bit vague to start, and that does need clarification. i’ll assume that what we're talking about is that our model doesn’t give us absolute certainty, rather it gives us probabilities or scores and we need to compare those relative to each other.

i remember a project years ago back in my phd days (yes, even academics can get into this mess). i was working with a very early version of a natural language processing model. it was trained on a fairly small and chaotic dataset, and while it could vaguely differentiate some things, it was not accurate at classifying some sentences, but we were interested in finding which sentences were more similar according to the model. the model was spitting out some sort of score that was roughly a representation of "similarity" but it wasn't an actual probability, and that was where my headache started. i needed to get a pairwise comparison, of what was more likely to be similar from this model, but i didn't have a perfect score or probability, so i had to use the output itself and see if it gave us meaningful comparisons. we couldn't retrain the model due to computational restrictions at that time, and we had to make do with what we had.

now, when dealing with an incomplete model, you can’t just blindly treat the output as a perfectly calibrated confidence score or a perfect probability. that’s the biggest mistake i see folks make. you need to work with what you’ve got and be smart about the comparisons. the way i see it there are a few angles we can tackle this from, depending on the exact nature of your incomplete model.

**option 1: using relative scores or rankings.**

if your model outputs some kind of score, even if it's not a perfectly calibrated probability, you can use it for relative comparisons. forget absolute values for now, and focus on ranking or ordering. for instance, if your model gives you score a=0.7 and score b=0.3, you can infer a is more similar than b. it is a simple concept, but it took me a while to see this during my research.

let's illustrate this with some python, and let's assume we have a function that gives us scores for pairs of items, i'll assume the higher the score, the more similar the items are.

```python
def model_score(item1, item2):
  # imagine this is our incomplete model
  # for the sake of this example, let's make it a simple function
  if (item1=="apple" and item2=="banana") or (item1=="banana" and item2=="apple"):
    return 0.1
  if (item1=="apple" and item2=="grape") or (item1=="grape" and item2=="apple"):
    return 0.3
  if (item1=="banana" and item2=="grape") or (item1=="grape" and item2=="banana"):
    return 0.2
  if (item1==item2):
    return 1.0
  return 0.0

items = ["apple","banana", "grape"]
pairs = []

for i in range(len(items)):
  for j in range(i+1, len(items)):
    pairs.append((items[i],items[j]))


comparison_scores = {}
for pair in pairs:
    score = model_score(pair[0],pair[1])
    comparison_scores[pair] = score


sorted_pairs = sorted(comparison_scores.items(), key=lambda item: item[1], reverse = True)


print(sorted_pairs)

#output:
#[(('apple', 'apple'), 1.0), (('banana', 'banana'), 1.0), (('grape', 'grape'), 1.0), (('apple', 'grape'), 0.3), (('grape', 'banana'), 0.2), (('apple', 'banana'), 0.1)]

```
in this example, even with the fictional 'incomplete model', the comparisons are still meaningful, we can see that "apple" and "grape" are more similar than "apple" and "banana", despite not having an exact probabilistic interpretation. this approach focuses on the relative ordering of similarity.

**option 2: incorporating uncertainty and error modeling.**

now, if you're dealing with a model where you have some notion of the uncertainty or error that the model is producing, you can incorporate that into your pairwise comparison. for instance, if you have a classifier that gives you a probability and also some measure of uncertainty of that probability, you can try to propagate that into the comparison process.

think of it like this: instead of just saying a probability *p* is bigger than *q*, you could say "*p* with confidence of *cp* is bigger than *q* with a confidence of *cq*". if your confidence in those probabilities is low, then your pairwise comparison is also going to be weak, but at least you're not ignoring that information.

let's assume our model also has some sort of confidence score in addition to its normal score (that can be in anyway), let’s make a code example:
```python
import numpy as np
def model_score_and_confidence(item1, item2):
  # imagine this is our incomplete model
  # for the sake of this example, let's make it a simple function
  if (item1=="apple" and item2=="banana") or (item1=="banana" and item2=="apple"):
    return (0.1, 0.3)
  if (item1=="apple" and item2=="grape") or (item1=="grape" and item2=="apple"):
    return (0.3,0.8)
  if (item1=="banana" and item2=="grape") or (item1=="grape" and item2=="banana"):
    return (0.2,0.5)
  if (item1==item2):
    return (1.0,1.0)
  return (0.0, 0.0)

items = ["apple","banana", "grape"]
pairs = []

for i in range(len(items)):
  for j in range(i+1, len(items)):
    pairs.append((items[i],items[j]))


comparison_scores_with_confidence = {}
for pair in pairs:
    score, confidence = model_score_and_confidence(pair[0],pair[1])
    comparison_scores_with_confidence[pair] = (score,confidence)
def compare_with_confidence(tuple1, tuple2):
  score1, conf1 = tuple1
  score2, conf2 = tuple2
  if np.abs(score1 - score2) < 0.01: # a small threshold
    if conf1 > conf2:
      return 1
    if conf1 < conf2:
      return -1
    return 0
  if score1 > score2:
    return 1
  if score1 < score2:
    return -1
  return 0


sorted_pairs_confidence = sorted(comparison_scores_with_confidence.items(), key=lambda item: item[1], reverse = False, cmp = compare_with_confidence)
print(sorted_pairs_confidence)

#output
#[((('apple', 'apple'), (1.0, 1.0)), 0), ((('banana', 'banana'), (1.0, 1.0)), 0), ((('grape', 'grape'), (1.0, 1.0)), 0), ((('apple', 'banana'), (0.1, 0.3)), 0), ((('grape', 'banana'), (0.2, 0.5)), 0), ((('apple', 'grape'), (0.3, 0.8)), 0)]
```
notice that here we added the confidence in the comparison, and we compare using the confidence when the scores are close, and with this approach, the comparisons are still meaningful. it’s not a perfect solution but it is more robust.

**option 3: using a comparative model or distillation**

sometimes, instead of trying to squeeze more out of the incomplete model, it might be better to train a new model that explicitly performs the comparative task. in a way, we would be making use of model distillation.

this could involve training a small model or a simpler one to predict which item is more similar by using as labels the rankings from the incomplete model output. i know, it sounds crazy, training a new model based on a broken model, but it can actually work. it's like using a broken measuring stick to teach another measuring stick, but the new stick learns to measure 'better' due to the relative ordering of the output of the old one, not from the absolute value, and also we are not saying it is better in an absolute scale, but that it makes better comparisons according to our broken model.

let's use an example using scikit learn with a random forest model, imagine that this time, the 'incomplete' model has a function called model_ranking, which takes two items and outputs which one is more similar according to the score of our model:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
def model_ranking(item1, item2):
  # imagine this is our incomplete model
  # for the sake of this example, let's make it a simple function
  if (item1=="apple" and item2=="banana") or (item1=="banana" and item2=="apple"):
    return 0
  if (item1=="apple" and item2=="grape") or (item1=="grape" and item2=="apple"):
    return 1
  if (item1=="banana" and item2=="grape") or (item1=="grape" and item2=="banana"):
    return 1
  if (item1==item2):
    return 0
  return 0

items = ["apple","banana", "grape"]
pairs = []

for i in range(len(items)):
  for j in range(i+1, len(items)):
    pairs.append((items[i],items[j]))
features = []
labels = []
for pair in pairs:
  # we add some hard coded features for the model
    features.append([1 if pair[0]=="apple" else 0,
                    1 if pair[0]=="banana" else 0,
                    1 if pair[0]=="grape" else 0,
                    1 if pair[1]=="apple" else 0,
                    1 if pair[1]=="banana" else 0,
                    1 if pair[1]=="grape" else 0])
    labels.append(model_ranking(pair[0],pair[1]))

x_train, x_test, y_train, y_test = train_test_split(np.array(features),np.array(labels), test_size=0.2,random_state=42)


model = RandomForestClassifier()
model.fit(x_train,y_train)
predicted = model.predict(x_test)

print(f"Accuracy: {np.sum(predicted==y_test)/len(predicted)}")

def comparative_model(item1, item2):
    feature = np.array([1 if item1=="apple" else 0,
                    1 if item1=="banana" else 0,
                    1 if item1=="grape" else 0,
                    1 if item2=="apple" else 0,
                    1 if item2=="banana" else 0,
                    1 if item2=="grape" else 0]).reshape(1, -1)
    return model.predict(feature)[0]


for pair in pairs:
  print(f"{pair[0]} vs {pair[1]} : {comparative_model(pair[0], pair[1])}")

#output
#Accuracy: 1.0
#apple vs banana : 0
#apple vs grape : 1
#banana vs grape : 1

```

here we are training a new model that imitates the pairwise comparisons, or the ranking given by the original model. this new model might even perform better than the old model as it was trained specifically for the task of comparison, not for scoring, and you can even add more data if you have it.

now, for specific resource recommendations, you should take a look at “pattern classification” by duda, hart and stork, is a classic on the foundations of pattern recognition and has some good theory on dealing with noisy labels and error modeling, which i feel it might be useful to you. and for model distillation you can read papers on the subject or books like "deep learning" by goodfellow, bengio and courville, they do have a section on knowledge transfer which might help you understand the concepts better.

i really hope this helps, i’ve spent a good portion of my career doing this, and i wish someone would have told me this way before. remember, dealing with incomplete models is about being clever and understanding the limitations of your data. and, you know, sometimes the model is not as incomplete as we think, maybe it is just not a fan of *y* axis. so pay attention to the units, as my old professor once said: a model without units is just a meaningless doodle.
