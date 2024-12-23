---
title: "functiontransformer scikit-learn pipeline usage?"
date: "2024-12-13"
id: "functiontransformer-scikit-learn-pipeline-usage"
---

 so you're asking about `FunctionTransformer` in scikit-learn pipelines right Been there done that more times than I care to admit I get it Its a powerful tool but can be a bit tricky the first few times so lets break it down like a bad bug fix

Essentially you're dealing with situations where your data isn't quite ready for the model you wanna use You need to preprocess it but not with the usual scikit-learn transformers like `StandardScaler` or `PolynomialFeatures` no no you need something custom Something that's specific to your data and its quirks That's where `FunctionTransformer` comes in It lets you apply an arbitrary function to your data within a pipeline basically giving you the ability to wrap that weird pre-processing you need into something you can use in a pipeline

I remember this one project I had involving time series data Oh boy It was this mess of sensor readings from some old machinery You know the type where the raw data is just a bunch of numbers that mean nothing until you get them transformed into something meaningful In my case it was converting timestamps into cyclical features like hour of the day and day of the week This isn't a built in scikit-learn feature it is more like domain knowledge that has to be done to use the data at all I first tried doing this outside of the pipeline you know just applying my function then feeding it into the pipeline Well you guessed it massive pain in the butt and the problem was I had to redo the transformation when I retrained the model and it was so cumbersome because it had to be done at a very particular part of the training process

So yeah `FunctionTransformer` saved my neck with that project because the alternative was a mess I mean it allowed me to incorporate this custom transformation directly within the pipeline so that when the pipeline is used to make predictions the transformation was automatically applied No more manual transformation of the data before I feed it to the model It was an absolute game changer let me tell you

Now lets get into the nitty-gritty of it Lets use an example lets say you're working with log scale data that you need to convert back to the linear scale for some reason Now if you are just going to go and do some np.exp() that is not going to be the way this is handled in a pipeline That means that when you make a prediction or want to train the pipeline it will not have this transformation done on the data and it's not going to work

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Sample data in log scale
log_data = np.array([[0.693, 1.609],
                    [1.098, 2.302],
                    [0.223, 0.916]])

# Convert the log data to the linear scale using np.exp()
def exponential_transform(x):
    return np.exp(x)

# Function transformer wrapper for exponential function
transformer = FunctionTransformer(exponential_transform)

# Apply the transformation
linear_data = transformer.transform(log_data)

print("Original Data (Log Scale):\n", log_data)
print("\nTransformed Data (Linear Scale):\n", linear_data)

# Build the pipeline
pipeline_exp = Pipeline([
    ('log_to_linear', FunctionTransformer(exponential_transform)),
    ('regression', LinearRegression())
])
pipeline_exp.fit(log_data,np.array([1,2,3])) #Fit on some fake labels

#Now we can use this to predict and it will use the data that has been transformed automatically
print('\nPrediction result:', pipeline_exp.predict(log_data))
```

See that it's really just a way to package functions so you can use them in your pipeline its really that simple but that simplicity is where the power comes in Now in this snippet we transformed the data before fitting the pipeline and this is not how it is supposed to work you have to fit and transform the same data This is one of the common issues when using the transformer in the pipeline you might transform the data outside the pipeline and then feed this transformed data into the model and it might work fine on your training set but when you want to predict it will not have the same transform applied

Another example lets imagine you have some feature that is a ratio of some other feature and you want to create this feature on the fly as part of the pipeline This is also a prime candidate for the function transformer and the code is something like this

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Sample data with two features
data = np.array([[10, 5],
                 [20, 8],
                 [15, 6]])

# Feature creation function that creates a ratio column
def create_ratio_feature(X):
    return np.c_[X, X[:, 0] / X[:, 1]]

# Function transformer wrapper for ratio feature
transformer = FunctionTransformer(create_ratio_feature)

# Apply transformation
transformed_data = transformer.transform(data)
print("Original Data:\n", data)
print("\nTransformed Data:\n", transformed_data)


# Build the pipeline
pipeline_ratio = Pipeline([
    ('create_ratio', FunctionTransformer(create_ratio_feature)),
    ('regression', LinearRegression())
])

pipeline_ratio.fit(data,np.array([1,2,3]))#Fit on some fake labels

#Now we can use this to predict and it will use the data that has been transformed automatically
print('\nPrediction result:', pipeline_ratio.predict(data))

```

Here's a common pitfall you might encounter especially when dealing with more complex transformations say you're doing a time-series rolling average or something similar using a function transformer If your custom function doesn't handle NumPy arrays well or maybe it expects something else it'll throw errors Like that one time I was trying to use a function that used Python lists when scikit-learn was giving it a NumPy array Oh man that was fun to debug I ended up having to add some `.tolist()` in my function and it magically worked then I realized I was not actually doing anything useful and re wrote it completely I know right classic rookie mistake

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# Sample data with one feature
data = np.array([[1], [2], [3], [4], [5]])

# Function to calculate rolling mean
def rolling_mean(X):
    window = 3
    return np.array([np.mean(X[max(0, i - window + 1) : i + 1]) for i in range(len(X))]).reshape(-1, 1)


# Function transformer wrapper for the rolling mean
transformer = FunctionTransformer(rolling_mean)

# Apply the transformation
transformed_data = transformer.transform(data)

print("Original Data:\n", data)
print("\nTransformed Data:\n", transformed_data)


# Build the pipeline
pipeline_mean = Pipeline([
    ('rolling_mean', FunctionTransformer(rolling_mean)),
    ('regression', LinearRegression())
])

pipeline_mean.fit(data,np.array([1,2,3,4,5]))#Fit on some fake labels

#Now we can use this to predict and it will use the data that has been transformed automatically
print('\nPrediction result:', pipeline_mean.predict(data))
```

Now regarding resources you should probably be looking at the official scikit-learn documentation for `FunctionTransformer` it is usually a good starting point but it won't really help you solve complex scenarios Also I would suggest reading some of the more in depth machine learning books that focus on pipeline design and engineering for example "Hands-On Machine Learning with Scikit-Learn Keras & TensorFlow" by Aurélien Géron has some good examples of how to use the transformer in practice and the book "Python for Data Analysis" by Wes McKinney might be good for learning how to do data transformation using numpy and pandas the book also covers how to optimize your code for speed and that is very important when working with large datasets

The key takeaway here is that `FunctionTransformer` lets you integrate your custom preprocessing logic seamlessly into the pipeline That is how you are supposed to use it that was the original reason this transformer was made and that's all there is to it so if you are doing something complicated with it make sure that your transformation is done inside the pipeline itself and not before or after it
