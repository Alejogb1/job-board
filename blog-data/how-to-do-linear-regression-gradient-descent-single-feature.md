---
title: "How to do Linear regression (Gradient descent) single feature?"
date: "2024-12-15"
id: "how-to-do-linear-regression-gradient-descent-single-feature"
---

so, you're asking about linear regression with gradient descent, focusing on just a single feature. i've definitely been down that road a few times, so let me share what i've learned. it's a pretty fundamental topic, but sometimes the little details can trip you up.

basically, we're trying to find a line that best fits a set of data points, where each point is defined by a single input feature (x) and an output target (y). the line is described by two parameters: the slope (often called 'w' or weight) and the y-intercept (often called 'b' or bias). we're hunting for the optimal values of 'w' and 'b' that minimize the error between our line's predictions and the actual data. gradient descent is the algorithm we use to get there.

think of it like this: imagine you’re standing on a hill, and you want to get to the lowest point in the valley, but you can't see the whole valley, only what’s around your feet. gradient descent works similarly. it looks at the slope of the error surface (a curve formed by error values depending on 'w' and 'b' values) at our current parameter values and takes a step in the direction that makes the error less steep. it does this iteratively, gradually moving us towards the lowest point - the minimum error.

the error we try to minimize is often the mean squared error (mse). it basically calculates the average of the squared differences between the actual y-values and our predicted y-values. the predicted y is calculated using our current estimate for 'w' and 'b' with the equation y_predicted = w*x + b.

the gradient of this error function with respect to 'w' and 'b' points towards the direction of the steepest increase in error. so, to minimize error we take the negative direction of this gradient (hence, descending).

now, to get practical, let's look at how we calculate the updates for 'w' and 'b' in each step. for single feature linear regression, these updates are computed as follows:

`dw = (1/m) * sum((y_predicted - y_actual) * x)`

`db = (1/m) * sum((y_predicted - y_actual))`

where:
-   m is the total number of data points.
-   x is the input feature.
-   y_actual is the actual target value.
-   y_predicted is the value predicted by our linear function based on the current 'w' and 'b' and the feature x (y_predicted = w\*x + b)
-   dw and db are the gradients for weights and bias respectively.

after that you update the weights like this:

`w = w - learning_rate * dw`

`b = b - learning_rate * db`

where `learning_rate` is a hyperparameter that controls the size of the steps we take. a very large learning rate can make the algorithm jump around the minimal, or diverge, while a very small learning rate can make it very slow to converge. getting a good learning rate is key.

i remember when i was first learning this, i was using an online dataset with housing prices and size of the houses and i did not realize i had not scaled my input data. the gradient descent was taking a long, long time to find the weights and bias because the size of the houses was 100-1000 times bigger than the prices. i did not understand what was going on until i read a section in “pattern recognition and machine learning” by christopher bishop which pointed me to the importance of feature scaling for gradient descent. i was pulling my hair out for a while on that, you do not know what a difference that made. i mean it should have been obvious, but you know, the devil is in the details.

here's some example python code that you can use to get started:

```python
import numpy as np

def linear_regression_single_feature(x, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    w = 0
    b = 0
    for _ in range(iterations):
        y_predicted = w * x + b
        dw = (1/m) * np.sum((y_predicted - y) * x)
        db = (1/m) * np.sum(y_predicted - y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b


# generating some sample data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# training
w, b = linear_regression_single_feature(x, y, learning_rate=0.01, iterations=1000)

print(f'optimized w: {w[0]}')
print(f'optimized b: {b[0]}')

```

this code initializes the weight (`w`) and bias (`b`) to zero, then it loops through gradient descent iterations, calculating the predictions, the partial derivatives of the error relative to 'w' and 'b', updating them by applying gradient descent rule. finally returns the trained 'w' and 'b'.

also, this example uses a fixed number of iterations, but in a real application you'd use a convergence criterion, stop when `dw` and `db` are small enough.

i’ve been messing around with these models for a while now and i once spent an entire weekend debugging one, because i was calculating the sum of the error gradient before dividing by 'm' (the number of samples). i should have know, a misplaced parenthesis can lead a long day (and night). i swear, these bugs always look obvious in hindsight, it is almost a law of nature.

now, let's get more flexible and use a class to wrap up the linear regression:

```python
import numpy as np

class LinearRegressionSingleFeature:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0
        self.b = 0

    def fit(self, x, y):
        m = len(y)
        for _ in range(self.iterations):
            y_predicted = self.w * x + self.b
            dw = (1 / m) * np.sum((y_predicted - y) * x)
            db = (1 / m) * np.sum(y_predicted - y)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        return self

    def predict(self, x):
         return self.w * x + self.b

# generating some sample data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# training
model = LinearRegressionSingleFeature(learning_rate=0.01, iterations=1000)
model.fit(x, y)

print(f'optimized w: {model.w[0]}')
print(f'optimized b: {model.b[0]}')

# generating some sample data to predict
x_predict= 2 * np.random.rand(10, 1)
y_predict = model.predict(x_predict)

print(f'predicted y based on random x input: {y_predict}')
```

this code puts the linear regression model in a class, providing an `__init__` constructor, the `fit` function for training, and `predict` function for producing predictions based on fitted parameters. It is way better for larger projects, you can also add functionalities like error tracking, and more.

here's a version using a convergence criteria instead of a fixed number of iterations:

```python
import numpy as np

class LinearRegressionSingleFeature:
    def __init__(self, learning_rate=0.01, tolerance=1e-5):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.w = 0
        self.b = 0

    def fit(self, x, y):
        m = len(y)
        prev_w = np.inf
        prev_b = np.inf
        while True:
            y_predicted = self.w * x + self.b
            dw = (1 / m) * np.sum((y_predicted - y) * x)
            db = (1 / m) * np.sum(y_predicted - y)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            if np.abs(self.w - prev_w) < self.tolerance and np.abs(self.b - prev_b) < self.tolerance:
                break

            prev_w = self.w
            prev_b = self.b
        return self

    def predict(self, x):
         return self.w * x + self.b

# generating some sample data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# training
model = LinearRegressionSingleFeature(learning_rate=0.01, tolerance=1e-5)
model.fit(x, y)

print(f'optimized w: {model.w[0]}')
print(f'optimized b: {model.b[0]}')

# generating some sample data to predict
x_predict= 2 * np.random.rand(10, 1)
y_predict = model.predict(x_predict)

print(f'predicted y based on random x input: {y_predict}')
```
this class incorporates a `tolerance` parameter which helps control when to finish the iterative process. it finishes iterating when the absolute difference between the weights of the last and before the last steps are less than the tolerance.

for further reading and to get a deeper theoretical background, check out “the elements of statistical learning” by hastie, tibshirani, and friedman. it's a solid textbook that will explain the maths in more detail, and has a great explanation of gradient descent, it’s good and detailed without being totally impenetrable.

so, that's the core of it. it's a simple algorithm but crucial to know well, and also is a fundamental building block for much more complex machine learning models. hope this clarifies it for you. if you have more specific questions later, just give me a shout.
