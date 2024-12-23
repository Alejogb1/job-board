---
title: "how to deal with imbalanced data classification?"
date: "2024-12-13"
id: "how-to-deal-with-imbalanced-data-classification"
---

so you've got imbalanced data classification right? Been there done that got the t-shirt and probably a few grey hairs. Let me tell you its a classic pain point we all hit sooner or later when you're knee-deep in machine learning projects. You think you've got everything set up your model's humming along then bam accuracy is great but specificity or recall is in the toilet. We've all stared at those numbers feeling like we're talking to a wall I've had that feeling myself way back when I was working on a fraud detection system for a small fintech startup.

I remember my first time vividly. We had transaction data millions of them and maybe a couple thousand labeled as fraudulent. The model was absolutely thrilled predicting not-fraud every single time because it was so easy I mean 99.99% not-fraud what’s the big deal right? High accuracy but absolutely useless. That’s when I really understood the sting of imbalanced datasets. So let’s break this down we'll go through some common solutions and hopefully give you some real world advice and not just theoretical BS.

**First things first Data Resampling**

This is your bread and butter really when you're facing imbalance you either gotta reduce the majority class samples (undersampling) or create new samples for the minority class (oversampling). We're trying to get the model to see a more balanced perspective right? Like giving the minority class a voice in the room instead of being drowned out by the majority.

Here's a quick example with Python using scikit-learn:

```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

# Assume you have your features in 'X' and labels in 'y' as pandas Series
# I will generate some dummy data for demo
X= pd.DataFrame({'feature1':[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10],
 'feature2':[11,12,13,14,15,16,17,18,19,20,11,12,13,14,15,16,17,18,19,20]
 })
y = pd.Series([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# SMOTE for oversampling. Good start for generating data if you do not know other better algorithms
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


print("Original training set shape:", X_train.shape)
print("Resampled training set shape:", X_train_resampled.shape)
```

SMOTE Synthetic Minority Over-sampling Technique creates new synthetic samples by interpolating between the existing minority class data points. It’s not perfect especially if you are dealing with very high dimensional data but it's a solid starting point. You should try different ratios like 1:2 or 1:1 to see what works better for your specific data. Undersampling has some potential to not work at all by removing relevant data points so proceed with caution when using that technique

Now I've seen some folks just blindly oversample and end up overfitting the model to noise of the synthetic samples. Be smart about it look at the performance metrics if you see that's an issue use less oversampling or try a different algorithm.

**Next Up Cost-Sensitive Learning**

Instead of tweaking the data directly we can change the model's perspective by penalizing the misclassification of the minority class more than the majority class. Think of it like this if you’re playing a game and the reward for getting something right is much higher if you’ve got a hard task compared to an easy one. Your model is the same so if it misclassifies a minority class instance it gets a much higher penalty. Many model libraries have built in ways to do that.

Here’s how you could do it with a simple logistic regression in Python:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Assume you have the training and testing data as in the previous example
model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42) #solver needed
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

The key here is `class_weight='balanced'`. This tells the model to automatically adjust the weights so that the minority class gets more attention during training or by defining the class weight yourself with your custom weight. I've personally seen significant improvements just by using this little argument in my models. We used that in the fintech setup along with boosting techniques. We were able to increase the recall to 70% from 25% just by setting that one line it was amazing.

**Let's Talk About Algorithms**

Some algorithms are naturally more resilient to imbalanced datasets than others. For instance tree-based methods like Random Forests or Gradient Boosting Machines are often good starting points. You could also look into algorithms designed for imbalance specifically like One-Class SVM for example.

Here's an example with XGBoost which is my go to algorithm:

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Assume you have your features in 'X' and labels in 'y'
X= pd.DataFrame({'feature1':[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10],
 'feature2':[11,12,13,14,15,16,17,18,19,20,11,12,13,14,15,16,17,18,19,20]
 })
y = pd.Series([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'scale_pos_weight': (len(y_train) - y_train.sum())/y_train.sum(), #this is the weight that is applied to the loss function
    'seed': 42
}


model = xgb.train(params, dtrain)
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)


print(classification_report(y_test, y_pred))

```

The crucial bit here is `'scale_pos_weight'`. Again like in the cost sensitive model, we are assigning weights to the classes during training by giving higher weight to the minority class, this tells XGBoost to pay more attention to the minority class when it calculates its loss. This isn’t magic it’s just about making the model realize that the minority class is important to make correct predictions. It's like being at a party and the only person asking good questions are a couple of people who are drowned out by louder talking. You have to focus on the voices that matter.

Remember though its not just the algorithm itself you need to be sure to use a more appropriate metric for imbalanced data such as precision recall F1 score AUC ROC etc

**Some Additional Advice From the Trenches**

*   **Don't blindly apply techniques.** Explore your data understand why it’s imbalanced is it a sampling issue or truly a skewed distribution?
*   **Use cross-validation properly.**  Avoid data leakage at all costs. Make sure you sample the train set and then split into cross-validation sets so there is no data leakage.
*   **Tune your hyperparameters.** Its rare to have the default parameter values be the optimum ones. Try to run a gridsearch or bayesian optimization.
*   **Be sure to monitor your metrics correctly.** Make sure to use precision recall or F1 scores not just accuracy.
*   **Sometimes better data can solve everything.** See if you can get more data for the minority class if possible or if you can generate better features from the ones you have

**Resources**

Instead of sending you random links I'd suggest looking into books like:

*   "Applied Predictive Modeling" by Max Kuhn and Kjell Johnson. Good overall resource that includes techniques for imbalanced data
*   "Data Mining: Practical Machine Learning Tools and Techniques" by Ian H. Witten, Eibe Frank, Mark A. Hall. Another solid foundation book

I would also advise you to read articles from the original papers if you wish to better understand the technique you are using rather than just using the default library methods. You can find those by doing a quick search in Google Scholar or similar search engine.

And that's the gist of it. Imbalanced data is a tricky issue and there is no universal magic bullet. But with a good understanding of the different approaches and careful experimentation you should be able to get better results. I have seen that sometimes it is all that is required and sometimes all the data is just not enough to have decent results but at least you will know you've done the best you could with the problem at hand. Good luck. Oh and I hope your code compiles the first time always. Haha just joking of course.
