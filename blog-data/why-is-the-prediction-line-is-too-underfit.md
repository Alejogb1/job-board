---
title: "Why is the prediction line is too underfit?"
date: "2024-12-15"
id: "why-is-the-prediction-line-is-too-underfit"
---

so, you've got a prediction line that's just not doing its job, huh? i've been there, staring at plots that look like a toddler drew them, while the actual data is a complex symphony. underfitting, it's a pain. let's break down why this happens and what we can do about it.

basically, underfitting means your model is too simple. it's not capturing the underlying patterns in your data. imagine trying to fit a straight line to a dataset that curves and weaves. that straight line will miss most of the action, and that's what your prediction line is doing. your model's too dumb, in short.

i've seen this countless times, even after years in this field. i remember this one project back in my early days, we were building a demand prediction model for a grocery chain. we were using simple linear regression – the default, if you will – and i remember looking at the prediction line thinking ‘this is not good’. our model just kept predicting the same average demand all the time, no matter the input features. turns out, the real data had strong seasonal patterns and daily fluctuations. our model was completely blind to that complexity. i spend days scratching my head until i realised, i needed something with more flex.

there are several reasons why your model might be too simple. let's go through some of the common culprits:

* **too few features**: you might not be feeding your model enough information. if you're trying to predict house prices using only square footage, for example, you're missing out on crucial factors like location, number of bedrooms, condition, school district, and so on. your model has to work with what it gets. if it doesn't have much to work with, well... it's like asking a carpenter to build a skyscraper with just a hammer.
* **using a simple model**: linear models are a classic example. they're great for simple relationships, but they're terrible at capturing non-linear patterns. if your data has curves, waves, or anything more complex than a straight line, linear regression is going to underfit badly. this is precisely what happened with my grocery chain project. we were using a linear model for a non-linear problem.
* **too much regularization**: regularization is a technique that helps prevent overfitting, which is the opposite of underfitting. but too much regularization can push your model towards being too simple. it basically shrinks the weights in your model, making it less sensitive to input data. it's like trying to mute a loud speaker, but you turn it too low. now you can't hear anything.
* **not enough training**: sometimes the problem is simply that your model hasn't had enough time to learn. it needs more examples, more passes through the data, and more opportunities to adjust its internal parameters. this is especially true for models with many parameters, such as neural networks.

now, let's talk solutions. here's what i usually do when i find myself battling underfitting:

1.  **add more features**: this is usually the first thing i try. think critically about what information might be relevant to your problem. feature engineering can be an art itself. sometimes you need to combine existing features, or even create completely new ones based on domain knowledge. make sure the features are actually relevant though, i've had situations where i added features that made zero sense at all and it made the problem worst. this is a very common pitfall to fall in.

    here's a little python snippet showing how to add polynomial features, a common way to add complexity:

    ```python
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np

    X = np.array([[1], [2], [3], [4]]) # Example feature
    poly = PolynomialFeatures(degree=2) # add a second degree (e.g. x^2)
    X_poly = poly.fit_transform(X)

    print(X_poly) # prints the original features and the new squared features
    ```

2.  **use a more complex model**: if linear models aren't cutting it, switch to non-linear models. decision trees, support vector machines (svms), and neural networks are good starting points. the choice of model depends on your specific problem, but don't be afraid to experiment. we switched from linear regression to support vector regression in our grocery demand model, and that made a world of difference in results, that day i learn something new and that is that all data is different and requires tailored solution.
    
    here's a simple example of switching to a polynomial regression model which can model non-linear relationships:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    # Sample non-linear data
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # Create polynomial regression
    degree = 3
    poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('linear', LinearRegression())
    ])
    poly_pipeline.fit(X, y)
    
    # Predict values using the trained model
    X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = poly_pipeline.predict(X_test)

    # Plot results
    plt.scatter(X, y, label='Actual Data')
    plt.plot(X_test, y_pred, color='red', label=f'Polynomial Regression (degree={degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    ```

3.  **reduce regularization**: if you’re already using regularization, try decreasing the strength. be cautious though, reducing regularization too much can cause overfitting (your model becomes too complex). it's like trying to get the temperature of water just right, but always overshooting. you have to test how much is good.

4.  **gather more data**: if you have the opportunity to collect more data, do it. more data often helps the model learn more robust patterns. the more, the merrier. and when you don't have data, there is this technique that they call 'data augmentation' which, when done carefully, can help to generate new data from existing data with small changes to add a bit more of data and see if that improves the model's performance. it's like giving the model another bite of the apple to really understand the flavour.

5.  **increase training time**: allow the model to train for longer. try more epochs for neural networks or increase the number of iterations for other models. it takes time for these models to learn, don't rush them. i even saw some people set up automatic training loops that would train for days, the result was worth it, they had a very well performing model.

remember that this is a process of iteration, try different approaches, see what works and what doesn't. each dataset is a unique snowflake, so sometimes you have to try many methods until you find one that works well. i spend countless days testing different models and features to find the sweet spot. be patient.

here's an example showing how to reduce regularization in a linear regression model by lowering the regularization parameter:
    
    ```python
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    # generate some dummy data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Ridge regression with a specific alpha value (reg parameter)
    alpha = 1.0  # start with 1.0 as a default
    model = Ridge(alpha=alpha)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Print score
    score = model.score(X_test,y_test)
    print(f"score with alpha {alpha}: {score}")
    
    alpha = 0.01  # Try smaller value
    model = Ridge(alpha=alpha)
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)

    print(f"score with alpha {alpha}: {score}")

    ```
    
    notice that the second model has a smaller alpha value of 0.01, meaning less regularization. depending on the data this can give a better or worse result, but it’s how you control the regularization.

for delving deeper into the underlying mathematics and theories, i strongly recommend reading “the elements of statistical learning” by hastie, tibshirani, and friedman. that book goes into the nitty-gritty of these algorithms and is a great resource for understanding the mechanics of fitting models. another great book, albeit a bit more hands-on, is “hands-on machine learning with scikit-learn, keras & tensorflow” by aurélien géron. you will get a good sense of how to build these models and all the different aspects of the model development pipeline with code examples. these resources are, in my view, the backbone of any good data science practioner, and knowing them would help you to diagnose your issue faster and provide better solutions for the kind of problems you are facing. also, don’t discount the documentation of packages you are using, the documentation of scikit-learn is very well made and has a lot of example use cases for several algorithms, i learn lots from those examples.

so, the key takeaway here is that underfitting usually means your model is too simple for the task at hand. you need to add complexity either by adding features, choosing a more appropriate model or reducing regularization. it can be a bit of a puzzle sometimes, but if you follow a systematic approach and experiment, you'll eventually get there. and remember that all of us face the same problems at some point, don't be too hard on yourself if something doesn't work immediately, we've all been there. happy modeling, and may your prediction lines fit just perfectly.
