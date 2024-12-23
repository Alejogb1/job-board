---
title: "How do I select the initial model in H2OAutoML?"
date: "2024-12-23"
id: "how-do-i-select-the-initial-model-in-h2oautoml"
---

Alright, let's tackle the question of initializing a model in H2O’s AutoML. It’s a deceptively nuanced part of the process, and while the framework does a lot of the heavy lifting for us, understanding the underpinnings can significantly improve your results and reduce frustration. In my time architecting machine learning pipelines, I've certainly seen teams fall into the trap of just accepting the default settings, which can lead to suboptimal outcomes. Let's break down the how and why.

The question actually implies a slight misunderstanding. AutoML in H2O doesn't typically *start* with a user-specified model, the initial model selection process operates under the hood. What we influence, instead, is the *strategy* it uses for that initial search. However, understanding this initial exploratory phase is key, as the early models often guide the hyperparameter space that is further explored in later iterations. Effectively, you're influencing the initial *direction* of the AutoML search.

There's no direct "initial model" parameter you'd configure. Instead, you’re shaping the search space and the algorithms it initially prioritizes via parameters like `max_runtime_secs`, `max_models`, and the family of `include`/`exclude` model types. Let's unpack how these mechanisms work and then see them in action with some practical Python snippets using the `h2o` library.

First, consider the overall goal: we want a good performance quickly. This usually means initially exploring a diverse set of models to get a feel for which model types are promising. H2O AutoML’s default behavior reflects this: it starts with a range of relatively fast models, such as basic GLMs, tree-based algorithms (like random forests and gradient boosting machines), and, optionally, Deep Learning. It's effectively trying to paint a rough sketch of the best-performing model space in the initial phases, setting the stage for further refinement.

The choice of algorithms in this initial stage is influenced by the nature of your data. For instance, if you have primarily numerical features, tree-based models might be preferred early on. If you have primarily categorical features, models such as GLMs, especially those with regularisation, or distributed Random Forests may initially show potential. AutoML uses heuristics and data inspection under the hood to subtly tailor this initial search.

Let's delve into the parameters that give you control. The `max_runtime_secs` and `max_models` parameters govern the stopping criteria for the entire AutoML process, not just the initial phase, but they indirectly affect the initial search exploration. Setting `max_runtime_secs` to a short period means AutoML will allocate less time to those early models, thus not fully utilizing the initial phase exploration. Conversely, a very high `max_runtime_secs` will allow it to investigate more early model candidates before spending time on iterative model improvements. Similarly, `max_models` limits the total models evaluated, and if set too low, you might not get the range of initial models needed.

More direct control comes through the `include_algos` and `exclude_algos` parameters, allowing you to specify which families of algorithms should or should not be considered. This is where your expertise can be injected. If your experience indicates a particular model type is promising for your data set – say, you've found that distributed gradient boosting with certain hyperparameter tuning works well on similar problems – you can force AutoML to prioritize this initially. Conversely, if you've consistently had issues with a specific algorithm, like deep neural nets on a dataset that's too small, you can exclude them from the search entirely.

Now, for some working code examples. Note that in these examples I'm assuming `h2o` is already imported and the H2O cluster is initiated, as well as the appropriate data has already been loaded, which is standard.

**Example 1: Restricting the algorithm search to only gradient boosting machines.**

```python
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster (if not already done)
h2o.init()

# Assuming 'train' dataframe is already an H2OFrame
#  and 'target' is the name of the target column

aml = H2OAutoML(max_runtime_secs=300,
               include_algos=["GBM"],
               seed=42)

aml.train(y='target', training_frame=train)

# get best model
best_model = aml.leader
print(best_model)

```

Here, I've explicitly constrained the search to only Gradient Boosting Machines (GBM). AutoML will then explore various GBM configurations first, making it useful for when you have strong reasons to believe GBM is the appropriate approach. The random seed ensures a consistent outcome.

**Example 2: Excluding Deep Learning from the Search.**

```python
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster (if not already done)
h2o.init()

# Assuming 'train' dataframe is already an H2OFrame
#  and 'target' is the name of the target column

aml = H2OAutoML(max_runtime_secs=300,
               exclude_algos=["DeepLearning"],
               seed=42)

aml.train(y='target', training_frame=train)

# get best model
best_model = aml.leader
print(best_model)
```

In this example, I've prevented Deep Learning models from being used. This might be suitable if your data is not sufficiently large or you suspect deep learning may be overkill for the task.

**Example 3: Using `max_models` to limit the exploratory search**

```python
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster (if not already done)
h2o.init()

# Assuming 'train' dataframe is already an H2OFrame
#  and 'target' is the name of the target column

aml = H2OAutoML(max_runtime_secs=120,
                max_models = 10,
                seed=42)

aml.train(y='target', training_frame=train)

# get best model
best_model = aml.leader
print(best_model)

```

This example limits total models evaluated. This strategy can be useful for faster experimentation to get an initial feel for model performance, at the potential cost of missing the best model. The `max_runtime_secs` is also reduced to allow for faster runs.

In summary, while you can't directly select an "initial model," these parameters offer the levers to control the exploratory search in H2O AutoML. For a deeper understanding of the inner workings, I'd suggest exploring the H2O documentation extensively, as well as the research papers that originally presented the AutoML algorithms. Books such as "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron, while not H2O-specific, cover the fundamentals of model selection strategies that will help you in reasoning through algorithm choice. Also, explore research papers specifically on AutoML algorithm selection strategies which usually get published at conferences such as NeurIPS, ICML and ICLR.

Remember, judicious use of these options, based on your knowledge of the dataset and the problem, will enable you to get to more robust and performant models, while speeding up the overall AutoML process. It's all about using your experience to guide the search in the right direction.
