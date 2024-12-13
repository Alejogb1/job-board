---
title: "fitted probabilities numerically 0 or 1 occurred warning?"
date: "2024-12-13"
id: "fitted-probabilities-numerically-0-or-1-occurred-warning"
---

Okay so you're getting those annoying "fitted probabilities numerically 0 or 1 occurred" warnings right Been there done that a thousand times it's like a rite of passage for anyone messing around with machine learning models particularly when you're dealing with classification problems and probabilities My guess is you're working with some sort of logistic regression or something similar that outputs probabilities between 0 and 1 and when the model is super confident and its predictions for classes it assigns values very close to 0 or 1 not *exactly* 0 or 1 but so close that some libraries throw a fit

First off let me tell you my story When I first stumbled upon this I was working on a project where we had to predict customer churn a classic I know I trained a logistic regression model on this dataset which was actually pretty clean I thought I'd nailed it everything was running great during training validation scores were looking beautiful Then during actual deployment when we started feeding the model real unseen data BAM This warning pops up everywhere fitted probabilities numerically 0 or 1 occurred and it's throwing a wrench in the works not exactly what I needed that day It felt like my well-crafted model was trying to break out of the cage of acceptable mathematical ranges

Basically this warning signals potential issues with your model's confidence and the mathematical computations underlying the process It means that at some points the probability estimates are converging so close to the boundaries 0 or 1 that it's like the model is shouting "I'M ABSOLUTELY CERTAIN" even when its rarely that clear cut in real life This is not necessarily *wrong* but can cause computational instability and issues further down the line for some methods You will see this problem in gradient descent during training it may mess up the loss computations or give you problems later on during prediction with the same issue

The problem is usually with what we call numerical precision Computers don't represent numbers with infinite precision they have a limited number of bits to represent floating point numbers So when you get probabilities that are say 0000000000000000001 or 09999999999999999999 it's practically 0 or 1 from a mathematical and practical point of view This means your model is probably overconfident or your dataset might have a specific problem that is causing this The common culprit tends to be class imbalance or features that are too predictive which can make the model over-fit and go "all-in" with its predictions We don't want the AI version of a bad poker player.

So what can you do about it Here are a few things that have worked for me over the years

**1 Check Your Data**

This is like rule number one in all machine learning problems You gotta make sure your data is balanced If one class is vastly overrepresented the model might learn to just predict the majority class and it will be very very confident of its predictions leading to this 0 or 1 situation For this try undersampling the majority class or oversampling the minority class or even better using techniques like SMOTE if you are working with tabular data that works better most of the time

```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assume 'X' is your features and 'y' is your target
# Split into train/test first before oversampling to not cause information leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = LogisticRegression(solver='liblinear')  # or your chosen model
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is {accuracy}")
```

Also check your features if you have a variable that is just too predictive that can also cause this problem if one specific feature can identify with perfect precision the label your model will naturally assign a very confident value to each prediction try and remove the variable or add some noise to it or maybe do some feature engineering to make it less perfect for prediction

**2 Regularization**

Regularization is your friend here If the model is too confident its because it's over-fitting to the training data introducing some regularization can help to reduce this overconfidence Regularization penalizes the model for having large coefficients preventing it from overfitting and making those extreme probability predictions You should always do cross validation to tune the hyperparameters

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#Assume 'X' is your features and 'y' is your target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
model = LogisticRegression(penalty = 'l2', solver = 'liblinear')
grid = GridSearchCV(model, params)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is {accuracy}")

```

This example uses L2 regularization through the `penalty = 'l2'` this will tend to shrink the coefficients of the model and that will decrease model confidence in prediction which would solve the problem to some extent this method can be used in logistic regression support vector machines and other linear models

**3 Calibration**

Sometimes the model is just not calibrated it will be overconfident but that does not necessarily mean that it is overfitting Calibration is the process of making sure that the predicted probabilities actually reflect the true confidence of the model for example when the model gives 095 probability in most cases you want it to predict the correct class 95% of the time Here's an example using isotonic regression or Platt scaling to calibrate your probabilities

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

#Assume 'X' is your features and 'y' is your target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear')
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)
y_pred = calibrated_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy is {accuracy}")
```

**Other considerations:**

*   **Different Models:** Sometimes the issue is inherent to the type of model you're using maybe the model is just too simple or too complex for the task. Consider trying other types of classification algorithms like decision trees random forests gradient boosting machines or neural networks sometimes this will not fix the problem but it may improve prediction performance a lot
*   **Increase the numerical precision:** While this is more difficult because it may involve changing fundamental settings in libraries or in languages you could see if there is a way to increase the precision of your floating point numbers for some numerical computations though I will be honest it is a last resort method and not usually practical to change the numerical precision that much.
*   **Clipping:** Sometimes a crude but effective method is just to clip probabilities to a reasonable range like between 0.01 and 0.99 basically if your predicted probability goes to 0000000000001 then you make it 001 and the same with 0999999999999 make it 099 it is simple but can work well for deployment

**Important Notes**

Don't ignore these warnings just because you are hitting a good training score This isn't a video game where you can just ignore all the warning messages that pop-up in the corner you have to address them or you might have some serious consequences later on and that can range from your model not working as intended or having hard to debug and find problems

Also try to use cross validation to better assess and evaluate your models using cross-validation you avoid biases and you have a better measure of how well your model is doing

And you should always keep up to date with the current literature on machine learning read research papers and keep up with blogs and resources from experts in the field that way you will better know if the methods you are using are actually working well and you can improve your model

**Recommended resources**

*   **"The Elements of Statistical Learning"** by Hastie Tibshirani and Friedman this is a textbook on machine learning fundamentals you need this if you want a deep understanding of the subject
*   **"Pattern Recognition and Machine Learning"** by Christopher Bishop this is another classic great book very rigorous and will help you a lot
*   **Scikit-learn's documentation** and other related libraries documentation is your best friend for everything machine learning read it all and know it

I hope that helps you out and that you manage to get rid of these annoying warnings trust me I've been where you are and I know it can be frustrating if you have other issues feel free to ask I am always around.

Disclaimer: I cannot provide guarantees but I have been working with this type of thing for many many years so I think it should work out for you.
