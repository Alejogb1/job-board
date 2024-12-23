---
title: "How can a recommender system be properly evaluated?"
date: "2024-12-23"
id: "how-can-a-recommender-system-be-properly-evaluated"
---

Okay, let’s talk about evaluating recommender systems. It's a crucial stage that, frankly, I’ve seen too many projects either gloss over or completely misunderstand. I remember a project back in my days at a fintech firm, where we spent months building this incredibly sophisticated collaborative filtering algorithm, only to realize our evaluation metrics were completely divorced from the actual user experience. It was a painful lesson, and it’s shaped how I approach evaluation ever since. The key takeaway is that evaluation isn't just about picking a single metric; it’s about crafting a holistic strategy that considers various aspects, each relevant to the specific goals of your system.

To properly evaluate a recommender system, you can't just throw accuracy metrics at the problem and call it a day. We need a multi-faceted approach because recommendations have several characteristics you need to measure. We’re generally dealing with implicit feedback (user clicks, views, purchases) instead of explicit ratings which makes things more complex. And, let’s be honest, accuracy alone is almost never sufficient. Think of it this way: a system that always recommends the most popular items will likely score high on accuracy, but it offers little value to the user. It's boring and certainly doesn’t feel personalized.

So, what are the main dimensions to consider? First, **accuracy**, which measures how well the system predicts items a user will interact with. Then there’s **diversity**, which concerns the variety of items presented to users. **Novelty** measures how different the recommended items are from what the user has interacted with before. **Coverage** is about the percentage of the catalog that the system can recommend. And, not least, **relevance** which gauges how aligned the recommendations are to the user's needs, as determined either by explicit user feedback or engagement patterns. We also need to keep an eye on **serendipity**, the ability to surface unexpected but relevant items, which is crucial for user satisfaction, and for certain applications.

Now, to get into more concrete methods. The foundation, usually, is split testing or A/B testing. You have a control group, with the existing or no recommendations, and one or several treatment groups using your new approach. But this is useless without sound metrics.

Let’s dive into some code examples to illustrate these metrics. I'll be using Python, because it's highly accessible and widely used in the field. Let’s assume we have interaction data where each row represents a user interacting with an item, coded like `(user_id, item_id)`. Let’s also suppose we have functions which generate predictions for a given user.

**Example 1: Precision and Recall (Accuracy Metrics)**

Precision asks: of all items we recommended, what percentage are actually relevant? Recall asks: of all items that should have been recommended, what percentage were we able to recommend? For simplicity, I am using the `sklearn` library for a simple calculation in a hypothetical function `generate_predictions`.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

def generate_predictions(user_id, ground_truth_items, top_n = 5):
  #Simulated function that predicts items. In a real scenario it
  # would take a model as a parameter or be within the model itself.
  # This is a dummy implementation.
  num_items_available = 100
  predicted_items = np.random.choice(num_items_available, size=top_n, replace=False)
  return predicted_items

def evaluate_precision_recall(user_ground_truth_dict, top_n=5):
  all_predicted = []
  all_ground_truth = []

  for user_id, ground_truth_items in user_ground_truth_dict.items():
     predicted_items = generate_predictions(user_id, ground_truth_items, top_n=top_n)
     all_predicted.extend(predicted_items)
     all_ground_truth.extend(ground_truth_items)

  #For precision, we must be sure that the predictions are boolean
  # meaning that we have to compare the predicted against the ground_truth.
  # For this example, we'll consider a match an item in the ground truth to a prediction.
  precision = precision_score(
      [int(item in all_ground_truth) for item in all_predicted], #ground truth of prediction.
      [1] * len(all_predicted), #true positive of all predictions (for precision)
      zero_division=0 #Handles cases with no predictions
  )
  # For recall, we have to compare the ground truth against the predictions
  # for this example, we will treat a single occurrence of each item as a hit
  recall = recall_score(
      [int(item in all_predicted) for item in all_ground_truth], #predicted ground truth
       [1] * len(all_ground_truth),#true positives of ground truth (for recall)
       zero_division=0 #Handles cases with no ground truth
  )

  return precision, recall

#Dummy ground truth data, in reality this would come from user interactions
user_ground_truth = {
    1: [12, 34, 56, 78],
    2: [23, 45, 67, 89],
    3: [1, 25, 50, 75]
}


precision, recall = evaluate_precision_recall(user_ground_truth)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

This code provides a basic calculation for precision and recall. Keep in mind that, in practice, you would use a more nuanced way to derive both measures, particularly as you go down the recommendation list. You might be only interested in ‘hits’ in the top-k recommended items. This is why metrics like Precision@K and Recall@K are commonly used. Also note that the ground truth needs to be extracted from your data in a correct way, which is very important.

**Example 2: Diversity (Intra-list similarity)**

Now let’s see how to measure diversity within the recommendations we are producing, focusing on the *intra-list* similarity. A low intra-list similarity typically indicates higher diversity among recommendations. We measure similarity using a simple cosine distance of item vectors. In a practical scenario, those vectors could be extracted using techniques like item2vec or from user-item interactions or content.

```python
import numpy as np
from scipy.spatial.distance import cosine

def get_item_vectors(num_items=100, vector_dim = 10):
  #Dummy generation of vectors. Usually you will calculate those with an
  #embedding of the item. This is just a simulation.
  return {item: np.random.rand(vector_dim) for item in range(num_items)}


def calculate_intra_list_diversity(predicted_items, item_vectors):
  if not predicted_items:
    return 1.0 #no predictions, max diversity
  similarity_sum = 0
  num_pairs = 0
  for i in range(len(predicted_items)):
    for j in range(i+1, len(predicted_items)):
        item1 = predicted_items[i]
        item2 = predicted_items[j]
        if item1 in item_vectors and item2 in item_vectors:
          similarity = 1 - cosine(item_vectors[item1], item_vectors[item2])
          similarity_sum += similarity
          num_pairs += 1
  if num_pairs == 0: return 1.0 #no pairs to compare, maximum diversity
  average_similarity = similarity_sum / num_pairs
  return 1 - average_similarity


def evaluate_diversity(user_ground_truth_dict, top_n = 5, item_vectors = None):
   if item_vectors is None:
      item_vectors = get_item_vectors()

   diversity_values = {}
   for user_id, ground_truth_items in user_ground_truth_dict.items():
      predicted_items = generate_predictions(user_id, ground_truth_items, top_n=top_n)
      diversity_values[user_id] = calculate_intra_list_diversity(predicted_items, item_vectors)

   return diversity_values

#Dummy ground truth data, in reality this would come from user interactions
user_ground_truth = {
    1: [12, 34, 56, 78],
    2: [23, 45, 67, 89],
    3: [1, 25, 50, 75]
}


diversity = evaluate_diversity(user_ground_truth)
for user, div in diversity.items():
  print(f"Diversity for user {user}: {div:.4f}")
```

This example calculates the average similarity between recommendations for a single user. Remember that higher values mean lower diversity, and zero means all recommended items are completely unrelated. The real value here would come from analyzing diversity across all users and comparing the values across different implementations.

**Example 3: Coverage**

Finally, let’s consider coverage. This measures what percentage of your catalog is actually recommended.

```python
def calculate_coverage(user_ground_truth_dict, total_items, top_n=5):
    recommended_items = set()
    for user_id, ground_truth_items in user_ground_truth_dict.items():
        predicted_items = generate_predictions(user_id, ground_truth_items, top_n=top_n)
        recommended_items.update(predicted_items)

    coverage = len(recommended_items) / total_items if total_items > 0 else 0
    return coverage

#Dummy ground truth data, in reality this would come from user interactions
user_ground_truth = {
    1: [12, 34, 56, 78],
    2: [23, 45, 67, 89],
    3: [1, 25, 50, 75]
}

total_items = 100
coverage = calculate_coverage(user_ground_truth, total_items)
print(f"Coverage: {coverage:.4f}")
```

Here, we calculate the total unique items that the recommendation system has predicted over all users, then determine the coverage by dividing by the total number of available items.

These examples, though simple, demonstrate the core principles. In a real-world application, you’d use more nuanced versions of these calculations and more advanced approaches. For instance, if you want to get deeper into the subject, you should check out "Recommender Systems: An Introduction" by Dietmar Jannach et al., and the more recent "Deep Learning for Recommender Systems" by Deep Learning for Recommender Systems. These are both excellent starting points. You should also look into the work done by researchers like Xavier Amatriain, and others in the field, whose work is often found in papers published at conferences such as RecSys or SIGIR.

Ultimately, the goal is not to maximize one single metric, but rather to balance a variety of measures. A system with high accuracy, but low diversity, for example, will likely become stale. Don’t be afraid to tailor your evaluation to fit your product goals and remember that you have to always iterate your metrics. It's a complex but absolutely essential aspect of building effective recommendation systems.
