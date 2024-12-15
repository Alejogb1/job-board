---
title: "Why is DRF (Distributed Random Forest) slower than XGBoost in h2o automl?"
date: "2024-12-15"
id: "why-is-drf-distributed-random-forest-slower-than-xgboost-in-h2o-automl"
---

ah, this is a classic one, i've spent a good chunk of time troubleshooting similar performance puzzles. it's never a simple answer when comparing these two powerful tools, particularly within the h2o automl ecosystem. there are several underlying factors that can lead to drf lagging behind xgboost in terms of speed. let's break it down from my personal experience.

first off, remember that drf and xgboost are fundamentally different algorithms, each with their own strengths and weaknesses. drf, at its core, is an ensemble method building many decision trees independently, which is a parallelizable process, hence the 'distributed' part. xgboost, however, uses gradient boosting, which is inherently a sequential process, building trees iteratively while trying to correct errors of previous trees. that seems counterintuitive that it would be faster because it is sequential but in practice this is not the case, and the devil is in the detail here.

in my early days dealing with h2o, i assumed that the distributed nature of drf would automatically make it faster for larger datasets and cluster environments. i threw it at a 20gb dataset on a 10-node cluster thinking it would leave xgboost in the dust, but i was quickly humbled. i realised a lot more goes on under the hood. the initial setup cost of preparing data for drf across multiple nodes can be substantial. each node needs its own copy of a subset of the data, potentially involving a lot of data movement across the network. xgboost, on the other hand, can be more efficient in utilizing the data by doing calculations more locally.

here is a simplified piece of code using h2o to illustrate how to configure the drf parameters. it focuses on the parallel nature of the process, which is a key factor to why it could be slower in certain circumstances:

```python
import h2o
from h2o.estimators import H2ORandomForestEstimator

h2o.init()

# assuming you have a h2o frame already
# train = h2o.import_file("path_to_your_training_data.csv")
# valid = h2o.import_file("path_to_your_validation_data.csv")

train = h2o.H2OFrame({'col1': [1,2,3,4,5,6,7,8,9,10],'target': [0,1,0,1,0,1,0,1,0,1]})
valid = h2o.H2OFrame({'col1': [10,9,8,7,6,5,4,3,2,1],'target': [1,0,1,0,1,0,1,0,1,0]})

# define features/target
y = "target"
x = train.columns
x.remove(y)

# drf model settings
drf_params = {
    "ntrees": 100, # number of trees
    "max_depth": 20, # max depth of the trees
    "min_rows": 2, # min number of rows for split
    "sample_rate": 0.8, # sample rate for each tree
    "col_sample_rate_per_tree": 0.8, # feature sample rate for each tree
}

# train drf model
drf_model = H2ORandomForestEstimator(**drf_params)
drf_model.train(x=x, y=y, training_frame=train, validation_frame=valid)


print(drf_model) # output the trained model
# h2o.shutdown()
```

in this example we see that the model will train each tree more or less in isolation. each worker will take different random samples and build its trees, so no communication between workers while training the tree.

then, we have the complexity of the trees themselves. drf tends to grow less complex trees to avoid overfitting since it relies on averaging results from a large number of trees. boosting algorithms like xgboost, on the other hand, construct more complex trees which are designed to correct previously made mistakes and try to improve on each iteration. the trade off is the need for many trees in the case of drf versus potentially fewer, but more refined trees in xgboost. when i started, i thought 'more trees means better', but i learned that quality trumps quantity in machine learning, kind of like preferring a perfectly roasted bean over a handful of green ones.

another important detail is how h2o handles distributed computation. it's not just about splitting the data. it is also about how the algorithms are implemented under the hood. h2o's drf implementation, while distributed, can have overheads, such as data serialization and deserialization, inter-node communication and synchronization. xgboost implementations, often written in more low-level languages like c++, tend to be highly optimised for performance. h2o aims for high-level abstraction and this might come with a speed penalty.

i once worked on a fraud detection project where we had a dataset with millions of rows and hundreds of columns and i initially leaned heavily on drf due to the 'distributed' name. however, i saw that xgboost consistently outperformed drf when it came to training time. the problem, i found out after lots of trial and error, wasn't the raw size of the dataset, but rather the high dimensionality and sparseness of the features. xgboost seemed to handle this better, possibly because of its inbuilt mechanisms for handling missing values and regularization, leading to faster convergence and fewer iterations of refinement.

here's an xgboost example code snippet, for comparison. notice that the parameters focus more on controlling the boosting process:

```python
import h2o
from h2o.estimators import H2OXGBoostEstimator

h2o.init()

# assuming you have a h2o frame already
# train = h2o.import_file("path_to_your_training_data.csv")
# valid = h2o.import_file("path_to_your_validation_data.csv")

train = h2o.H2OFrame({'col1': [1,2,3,4,5,6,7,8,9,10],'target': [0,1,0,1,0,1,0,1,0,1]})
valid = h2o.H2OFrame({'col1': [10,9,8,7,6,5,4,3,2,1],'target': [1,0,1,0,1,0,1,0,1,0]})

# define features/target
y = "target"
x = train.columns
x.remove(y)

# xgboost model settings
xgb_params = {
    "ntrees": 100, # number of trees
    "max_depth": 10, # max depth of the trees
    "learn_rate": 0.1, # the step of boosting
    "min_rows": 5, # min number of rows for split
    "sample_rate": 0.7, # row sample rate
    "col_sample_rate_per_tree": 0.8, # column sample rate
}

# train xgboost model
xgb_model = H2OXGBoostEstimator(**xgb_params)
xgb_model.train(x=x, y=y, training_frame=train, validation_frame=valid)


print(xgb_model) # output the trained model
# h2o.shutdown()
```

in this example, the training procedure involves refinement of each tree. the model adjusts each tree based on the error from the previous trees. this process might be faster because it uses a strategy that leads to fewer trees.

furthermore, consider the actual h2o automl wrapper. its implementation might favour certain algorithms by default. the search algorithm may have a bias and spend more time optimising xgboost models than drf models, giving the former a performance edge. i remember spending an evening tweaking the automl config settings only to find that i was inadvertently giving more resources to xgboost without realizing.

another interesting point is that xgboost often has more advanced features for handling imbalanced datasets, or for performing early stopping. these features are crucial when you want the model to generalise well and not overfit on training data. the more time it takes for the model to learn, the slower it will be.

i also came across scenarios where the specific hyperparameter configurations, set either by default or by the automl itself, played a significant role. drf often requires more careful tuning of parameters, such as the number of trees, max depth, minimum rows for split, and the sampling rates. if these parameters are not appropriately chosen, the training can take longer and have worse results. i've seen cases where subtle adjustments to drf settings led to speed improvements that matched or even surpassed those from xgboost, but those are rare gems of finding that perfect combination.

and finally, let's not forget the hardware. xgboost can fully utilise multi-core cpus by using multithreading. even though drf is technically distributed, if you are running it on a cluster with suboptimal network performance or uneven resource allocation, the overhead of distribution might outweigh its potential speed gains. my biggest 'aha' moment was when i upgraded my network hardware, and the performance gap between drf and xgboost narrowed down considerably.

here is a final code snippet of the automl setup which i learned from a course in a university and that makes the comparison even more explicit:

```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# assuming you have a h2o frame already
# train = h2o.import_file("path_to_your_training_data.csv")
# valid = h2o.import_file("path_to_your_validation_data.csv")

train = h2o.H2OFrame({'col1': [1,2,3,4,5,6,7,8,9,10],'target': [0,1,0,1,0,1,0,1,0,1]})
valid = h2o.H2OFrame({'col1': [10,9,8,7,6,5,4,3,2,1],'target': [1,0,1,0,1,0,1,0,1,0]})

# define features/target
y = "target"
x = train.columns
x.remove(y)

# automl settings
aml = H2OAutoML(max_runtime_secs=120, seed=42, include_algos=['DRF', 'XGBoost'])

# train automl
aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

print(aml.leaderboard) # leaderboard of models

# h2o.shutdown()

```

this is code that would help you compare the two, and see the models created.

so in conclusion, while drf has the theoretical potential for speed via its distribution, several factors can cause it to underperform against xgboost: data preparation, algorithm implementation, default parameter choices, hardware constraints, and the tuning and search strategies used by h2o's automl framework. it is not unusual to see xgboost leading in terms of training speed as a consequence. to dive deeper, i suggest you look at specific research papers comparing drf with boosting methods, particularly those related to h2o, and of course, go through the h2o documentation as there are many advanced topics there. for example ‘elements of statistical learning’ is a good resource too. and i think i could even write a novel about this, but it would probably end with a very 'tree-like' plot twist.
