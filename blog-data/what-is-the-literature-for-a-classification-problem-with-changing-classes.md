---
title: "What is the literature for a Classification Problem with Changing Classes?"
date: "2024-12-14"
id: "what-is-the-literature-for-a-classification-problem-with-changing-classes"
---

alright, so, you're asking about classification problems where the classes themselves change over time, not just the distribution of examples within a fixed set of classes. i've been there, done that, got the t-shirt with the faded print of a confused neural network. it's a tricky area, and it took me a while to get my head around it properly. let me share my experiences.

this isn't your standard supervised learning gig, where you train a model on a static dataset and then throw it into the wild hoping it behaves. we're talking about a more dynamic, evolving landscape. we are dealing with what's generally referred to as concept drift or concept evolution. sometimes classes merge, sometimes they split, sometimes they just vanish and new ones appear, like a badly coded game of snake.

early in my career, back when dial-up was still a thing, i was working on a spam filter (because that's what every tech person did, right?) . the problem we had was that the characteristics of spam were constantly changing. what was obviously spam last month might look like a perfectly normal email this month, and vice-versa. this wasn’t just about the words, but also the headers, the sending patterns - all sorts of things. we tried all the simple classifiers. naive bayes, logistic regression, svms, but they all had a very limited lifespan before they started misclassifying emails like a toddler sorting socks. that's where i first started encountering this dynamic classification problem head-on.

now, the literature around this isn't as neatly packaged as, say, "introduction to convolutional neural networks." you need to pull from different areas and stitch things together. let's discuss what i learned.

first, we have methods that try to detect when these changes happen. there’s the cumulative sum (cusum) algorithm, and others that look at statistical properties of your data and generate signals of distributional shift. they don't tell you how to solve the problem, just when something went wrong. we called these my 'smoke detectors', early warnings of the impending doom of classifier failure.

```python
import numpy as np

def cusum(data, threshold, drift_values = []):
    g_plus = 0
    g_minus = 0
    drift_points = []

    for index, value in enumerate(data):
      g_plus = max(0, g_plus + value)
      g_minus = min(0, g_minus + value)
      drift_values.append(value)
      if g_plus > threshold:
        drift_points.append(index)
        g_plus = 0
      if g_minus < -threshold:
        drift_points.append(index)
        g_minus = 0

    return drift_points, drift_values

# example usage:
data = np.random.normal(0,1,100)
data[50:] = data[50:] + 3  # simulate a shift
data_cusum = cusum(data, 4)

print(data_cusum[0])

```

this very basic example of cusum simply helps visualize how the cummulative sum works. if you run this, you’ll find a drift point close to index 50. you have to remember, the threshold is important, too big a threshold and you won’t detect any changes, too small and everything looks like drift.

once you have detection, the next step is adaptation, which is trickier. one common approach is incremental learning. here we don't throw away the old model. instead, we update it with new examples, assuming the change is slow and gradual. that's not always the case, so we have other more complex techniques. it’s like trying to teach an old dog new tricks, but the tricks keep changing too.

another route is to use ensemble methods, where you maintain a collection of models, each trained on different time windows or data batches. they’re combined, weighted or used in a selection scheme according to their past performance. it helps maintain an overall performance even with drifts that might cause one model to be inaccurate for a while. if one model starts falling behind, it’s weighted less, so if it’s a false alarm, it doesn't affect the performance. if a model starts doing well it’s weighted more, and if a new class pops up, or it shifts in the data, then new models are created.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

class EnsembleClassifier:
    def __init__(self, base_classifier = SGDClassifier(loss='log_loss',random_state=42), ensemble_size=3):
        self.classifiers = [base_classifier for _ in range(ensemble_size)]
        self.weights = [1 / ensemble_size for _ in range(ensemble_size)]
        self.ensemble_size = ensemble_size

    def fit(self, X, y):
        for index, classifier in enumerate(self.classifiers):
          if hasattr(classifier, 'partial_fit'):
            classifier.partial_fit(X,y,classes=np.unique(y))
          else:
            classifier.fit(X,y)

    def predict_proba(self, X):
        predictions = []
        for i in range(self.ensemble_size):
            try:
                proba = self.classifiers[i].predict_proba(X)
            except:
                proba = np.zeros((len(X), len(np.unique(y))))
            predictions.append(proba)
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
        return weighted_predictions

    def predict(self, X):
       proba = self.predict_proba(X)
       return np.argmax(proba, axis=1)

    def evaluate(self, X, y):
      y_pred = self.predict(X)
      return accuracy_score(y, y_pred)

    def update_weights(self, X, y, learning_rate = 0.01):
      y_proba = self.predict_proba(X)

      for i in range(self.ensemble_size):
        current_error = self._calculate_weighted_error(y, y_proba, i)
        self.weights[i] = self.weights[i] * np.exp(-learning_rate * current_error)
      self.weights = self.weights/np.sum(self.weights) #normalize them

    def _calculate_weighted_error(self, y_true, y_proba, classifier_index):
      y_pred = np.argmax(y_proba, axis=1)
      correct = y_true == y_pred
      classifier_pred = np.argmax(self.classifiers[classifier_index].predict_proba(X), axis=1)
      classifier_correct = y_true == classifier_pred
      return np.mean(correct != classifier_correct) # error is 1 when wrong and 0 when correct

# Example Usage:
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ensemble = EnsembleClassifier()
ensemble.fit(X_train,y_train)
print(f'accuracy {ensemble.evaluate(X_test, y_test)}')
ensemble.update_weights(X_train, y_train)
print(f'accuracy after weight update {ensemble.evaluate(X_test, y_test)}')


X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
y[50:] = np.random.randint(1,3,50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ensemble = EnsembleClassifier()
ensemble.fit(X_train,y_train)
print(f'accuracy with drift {ensemble.evaluate(X_test, y_test)}')
ensemble.update_weights(X_train, y_train)
print(f'accuracy with drift after weight update {ensemble.evaluate(X_test, y_test)}')

```

this is a basic ensemble that learns weights for its different classifiers, but it's only a tiny toy example. to be fair i’ve seen worse code pass review. ensemble methods tend to be more robust to concept drift, but they do come with added complexity of training and maintenance.

another area that is worth exploring is reinforcement learning (rl). the idea is to see the classification problem as an agent constantly interacting with an environment that provides data and labels. the classifier receives a reward (accuracy) and adjusts its actions (classification decisions). it's more computationally intensive but it handles situations where the classes and labels can change dynamically. we moved on from these early rule based spam filters to this new approach.

i remember implementing a rl approach and seeing it gradually adapt to new spam tactics. it was a long training process but the ability to evolve really made it worthwhile in the end. it's fascinating and it's why i love this field. here’s how that would look like, if we had to use something like a q-learning approach:

```python
import numpy as np
from collections import defaultdict

class QLearningClassifier:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.q_table = defaultdict(lambda: np.zeros(n_actions)) # state => actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.n_states = n_states
        self.n_actions = n_actions

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state][action] = new_q

    def predict(self, state):
       return np.argmax(self.q_table[state])

# Example Usage
n_states = 10 #simplified version, not full data
n_actions = 3 #classification output
classifier = QLearningClassifier(n_states, n_actions)

for i in range(10000):
  state = np.random.randint(0, n_states)
  action = classifier.select_action(state)
  next_state = np.random.randint(0, n_states)
  if np.random.rand() > 0.8:
    reward = 1 #correct prediction
  else:
    reward = -1 # wrong prediction
  classifier.learn(state, action, reward, next_state)

# Predict
print(f'example prediction {classifier.predict(np.random.randint(0, n_states))}')


```

this is a very, very simplified example. usually, the states are the actual features coming from the data, but that will complicate the example. the main takeaway is that the classifier learns how to classify by interacting with the data and receiving a reward for each prediction. with a proper setup it can learn to adapt to changes in the classes over time. the reinforcement learning paradigm is far more advanced than this simple example.

for further reading, i'd suggest you explore some key texts. "data stream mining: algorithms and techniques" by joao gama is great for the basics of handling data with evolving properties. the “concept drift adaptation” section in "mining massive datasets" by jure leskovec et al is a good starting point as well. it's important not to limit yourself to books either, research papers are essential in this field. look for conference publications in areas like icml, nips and aistats, they usually contain the cutting edge methods. don’t be afraid to get stuck in the math, it will only improve your understanding.

so, in short, dealing with changing classes isn't a single technique. it's more of a mindset of constantly monitoring, detecting, and adapting. it's a continuous process.

and that's what i've learned so far. always keeping track of new developments and trying new ideas and algorithms is part of the process.
