---
title: "How to do downsampling on a multi-label dataset?"
date: "2024-12-15"
id: "how-to-do-downsampling-on-a-multi-label-dataset"
---

alright, so, downsampling a multi-label dataset, i've been there, done that, got the t-shirt, and probably debugged more lines of code related to it than i care to remember. it's a fairly common issue when you're dealing with imbalanced datasets, especially when each instance can have multiple labels assigned to it. it’s a bit more nuanced than your typical binary classification downsampling.

first off, let's get the terminology straight. when i talk about downsampling, i'm referring to reducing the number of instances in your dataset, specifically instances associated with the over-represented labels. this is a strategy to alleviate the impact of class imbalance which could bias your model towards the majority classes. in a multi-label context, the challenge is that each instance can belong to multiple classes, so we need to think a bit differently than in the single-label case. we don’t just discard instances randomly, or based on the single label.

the classic approach of just randomly under sampling isn't ideal here because it doesn’t respect the multi-label structure, potentially losing crucial connections between labels. if we just randomly discard some samples with the most frequent labels there is a chance we are removing important instances of less frequent labels that are paired to the ones we are down sampling. we can’t just drop any instance containing label ‘x’ if that instance has also label ‘y’ that we need because it’s less represented.

i remember a project back in the day, it was a text classification problem involving medical reports. each report could be tagged with multiple medical conditions. the ‘common cold’ tag was rampant, while the ‘rare genetic disorder’ tags were sparse. if i did not do downsampling correctly or simply removed some instances with 'common cold' tags i would have lost instances that had also the ‘rare genetic disorder’ tag. the initial model performance was awful, biased towards predicting only the common ailments. this was a perfect example of a scenario that called for careful multi-label downsampling. it was a late night, lots of coffee and debugging with gdb involved. the struggle was real, i was so frustrated i even tried to debug my own brain with introspection but that did not work that well, anyway let's continue.

so, how do we tackle this? well, there are a few common strategies to consider:

**1. per-label under sampling (or label-wise under sampling):**

the simplest strategy. for each label, we reduce the number of instances to match a target number. this way, each label is treated individually, but you have to manage how this affects multi-label connections. the problem here is that you might end up removing instances that actually contribute to the less represented labels or that have useful co-occurrences. in practice i found out that you still need to be mindful of those co-occurrence relationships and you could even calculate which labels are more often present together and keep them. in pseudo-code it would be something like this:

```python
def per_label_downsampling(data, labels, target_size):
    downsampled_data = []
    for label_idx in range(labels.shape[1]):  # iterate labels
        label_indices = np.where(labels[:, label_idx] == 1)[0]
        if len(label_indices) > target_size:
            sampled_indices = np.random.choice(label_indices, target_size, replace=False)
            downsampled_data.extend(list(zip(sampled_indices,[label_idx]*len(sampled_indices) )))
    
    # reconstruct our data and labels, this will remove labels as well, but they should be kept if found useful in other instances, but i'm keeping things simple for now
    
    new_data = []
    new_labels = []
    seen_data_indices = set()
    for data_index, label_index in downsampled_data:
       if data_index not in seen_data_indices:
         new_data.append(data[data_index])
         new_labels.append(labels[data_index])
         seen_data_indices.add(data_index)
    return np.array(new_data), np.array(new_labels)

```

this is very basic but it shows the idea, you go through each label and select some indices, then reconstruct your new data and labels, discarding some of the original ones. you could modify the code to keep the indices that are related to other labels as well, and in fact, the next strategy addresses this issue better.

**2.  instance-based downsampling using label frequency counts:**

this method focuses on instances rather than individual labels. we create a frequency count of all labels present in an instance and then use that to define the sampling probability for each instance. this attempts to balance the labels without fully ignoring their relationships. basically, if an instance has a lot of frequent labels and not many infrequent labels, it becomes a stronger candidate for removal, while instances containing rarer labels have higher chance of being kept. think of this like a survival of the fittest, only the fittest for the model of course, not any form of darwinian social darwinism thing or anything like that, just data. we don't want to lose data with the rare labels that could be helpful for our model.
a simple python implementation would be:

```python
import numpy as np

def instance_based_downsampling(data, labels, target_instances):
  instance_label_counts = np.sum(labels, axis=1)

  if len(instance_label_counts) <= target_instances:
    return data, labels  # nothing to be downsampled

  sampling_probabilities = 1 / (instance_label_counts + 1)  # add 1 to avoid division by zero and less aggressive sampling

  sampled_indices = np.random.choice(len(data), size=target_instances, replace=False, p=sampling_probabilities/np.sum(sampling_probabilities))

  return data[sampled_indices], labels[sampled_indices]

```
here the probability to be selected is inversely proportional to how many labels an instance has. the instances with the most labels get sampled with lower chance and instances with less labels will have a higher chance to be sampled. this method helps to keep the diversity in multi-label sets. this method was a lifesaver in a project i had to classify posts in social media into categories of sentiment and topics, the posts usually had multiple topic labels but the sentiment was usually single. with this method i was able to deal with the imbalance without losing too much useful data.

**3.  advanced methods with informed sampling:**

beyond the basics, things get more interesting. we can implement sampling probabilities that are based on more sophisticated methods. a straightforward example is using tf-idf (term frequency-inverse document frequency) scores to weight the importance of labels within instances. this could help us keep instances that are more "representative" or provide more unique information related to less common labels. we could also use advanced methods of measuring data diversity within the instances, the problem with those advanced methods is that they usually add computational cost so it's a balance between improvement of data quality and processing costs. some libraries help on this as well, but sometimes it's hard to fit it in your pipeline, that was my experience anyway.

example of weighted sampling with label frequencies:
```python
def weighted_downsampling(data, labels, target_instances, weights = None):
    if weights is None:
        instance_label_counts = np.sum(labels, axis=1)
        weights = 1 / (instance_label_counts + 1) # basic inverse label count weighting
    if len(data) <= target_instances:
        return data, labels #nothing to do here, no downsampling required

    #normalize weights so they sum to 1
    normalized_weights = weights/np.sum(weights)

    sampled_indices = np.random.choice(len(data), size=target_instances, replace=False, p=normalized_weights)
    return data[sampled_indices], labels[sampled_indices]
```

these weights could be as complex as you want, i remember once, i had a problem with images of handwritten characters and different styles. my weighted downsampling method had a diversity score so i kept instances with higher diversity, it took a while to write the code, but the model performance increased and the model was able to generalize better.

**things to consider:**

*   **target size:** deciding what the target size should be is a matter of experimentation, this is not a one size fits all thing. it’s a good idea to monitor your model's performance as you adjust this parameter, try small steps and see how the model does. there is not magic number that works for every problem.
*   **validation strategy:** make sure your validation set is also correctly representative of the multi-label distribution of your data, you want to properly evaluate your model, if your test set does not have any ‘rare genetic disorder’ samples and you did your downsampling to balance such instances you will not get the result you expect.
*   **combined with upsampling:** sometimes downsampling may remove too much data. you might combine it with upsampling techniques. personally, i’ve had more success with downsampling majority instances than with upsampling minority ones. it is better to discard low information data than to synthesize something that does not represent reality. in a way i find it elegant and more in tune with occam's razor, it helps to keep things simpler.
*   **evaluation metrics:** make sure that the metrics you’re using are appropriate for multi-label classification, simple accuracy is not a good idea, hamming loss, jaccard index, precision recall and f1 score per label are better choices.
*   **iterative refinement:** you will not likely hit the right settings in the first attempt, experiment with different target sizes and downsampling strategies to see what works for your data and model.

**resources:**

for a deeper dive, i'd suggest looking into some of the following resources:

*   **"imbalanced learning: foundations, algorithms, and applications"** by haibo he and yunqian ma. this book covers the theory behind imbalanced learning and the common techniques used.
*   research papers on multi-label learning with a focus on imbalanced data. i remember reading some good ones on sampling methods for text classification, but unfortunately i do not remember the author right now, i’m getting old. you could try searching in ieee xplore or acm digital library, those resources are usually good places to start.

i hope this helps, good luck! oh by the way, i just bought a new rgb keyboard, it has more colors than my coding style. i guess i'm not getting any better, hahaha. well, back to the code, i have to fix that pesky bug.
