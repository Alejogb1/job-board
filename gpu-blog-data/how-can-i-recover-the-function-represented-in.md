---
title: "How can I recover the function represented in the graph?"
date: "2025-01-30"
id: "how-can-i-recover-the-function-represented-in"
---
The fundamental challenge in recovering a function from its graph lies in the inherent loss of information during the visualization process. A graph, whether plotted on paper or a screen, provides a discrete sampling of the function's continuous domain. Thus, perfect recovery is generally impossible unless certain prior knowledge about the underlying function is available. I’ve encountered this issue frequently in my work with sensor data, where raw readings are plotted and then require mathematical representation for analysis. I'll outline how to approach function recovery based on common scenarios I’ve faced.

**Understanding the Problem**

The process of "recovering" a function from a graph fundamentally involves approximation or interpolation. The given graph provides a set of (x, y) data points. Our goal is to find a mathematical expression, f(x), that best describes the relationship between these points. The key is to choose an appropriate model that best fits the distribution of these points. The complexity ranges from simple linear models to non-linear regressions or even employing machine learning techniques if no parametric model is suitable. The method of selection is highly dependent on the nature of the graphed data points. For example, is it a straight line, a curve, a periodic wave or a complex pattern? Knowing, or hypothesizing based on observation, the type of function the data represents is the essential first step.

**Methods for Function Approximation**

1.  **Linear Regression:** If the data points appear to follow a linear trend, linear regression is the appropriate technique. This involves finding the equation of a line, typically expressed as *y = mx + c*, that minimizes the sum of squared differences between the predicted y-values and the observed y-values. I've frequently used this for approximating sensor drift and for calibrating devices.

    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Example data from the graph (x, y)
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Reshape for sklearn
    y = np.array([2.1, 3.9, 6.1, 8.3, 9.8])

    # Create a linear regression model
    model = LinearRegression()

    # Train the model using the data
    model.fit(x, y)

    # Get the coefficients (slope and intercept)
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Linear Equation: y = {slope:.2f}x + {intercept:.2f}")

    # To predict the function at other values,
    new_x = np.array([6]).reshape(-1, 1)
    predicted_y = model.predict(new_x)[0]
    print(f"Predicted y for x = 6: {predicted_y:.2f}")
    ```

    In the above code, the *LinearRegression* from *sklearn* provides a robust way to fit the linear equation. The data needs to be reshaped as the *sklearn* library expects features to be 2D arrays. After training, we get the slope and intercept, which defines the linear function that best describes the sampled data.

2.  **Polynomial Regression:** If the graph demonstrates curvature, a polynomial regression model should be considered. This can be a quadratic equation (*y = ax² + bx + c*), a cubic equation (*y = ax³ + bx² + cx + d*), or higher orders depending on the complexity of the curve. The degree of the polynomial determines the flexibility of the curve fitting, with higher degrees generally allowing for closer fit, but potentially leading to overfitting. In my time working with non-linear circuits, I’ve often found a second or third order polynomial adequate.

    ```python
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    # Example data from the graph (x, y)
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2.2, 5.2, 10.3, 17.1, 26.0])

    # Transform features to polynomial features
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)

    # Train the model
    model = LinearRegression()
    model.fit(x_poly, y)

    # Get coefficients and intercept
    coeffs = model.coef_
    intercept = model.intercept_

    print(f"Quadratic Equation: y = {coeffs[2]:.2f}x^2 + {coeffs[1]:.2f}x + {intercept:.2f}")

    # To predict with the polynomial model
    new_x = np.array([6]).reshape(-1, 1)
    new_x_poly = poly.transform(new_x)
    predicted_y = model.predict(new_x_poly)[0]
    print(f"Predicted y for x = 6: {predicted_y:.2f}")

    ```

    In the preceding code, I used the *PolynomialFeatures* class to transform the feature set, making them usable for linear regression. The output polynomial equation can then be used for predicting at other points in the domain. The order of polynomial needs to be selected based on visual inspection of the data distribution and cross validation to avoid over fitting.

3.  **Curve Fitting with Non-Linear Models:** When the relationship is not linear nor easily approximated by a polynomial, or when domain knowledge suggests a specific functional form, curve fitting techniques using libraries such as *scipy.optimize* are crucial. This approach involves defining the equation of the anticipated curve (e.g., exponential, logarithmic, trigonometric), then finding the parameters that best match the data. This method is often used in scientific fields such as physics and electrical engineering.

    ```python
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt


    # Example data from the graph (x, y)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.5, 2.7, 7.1, 18.5, 47.4])


    # Define an exponential function
    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    # Fit the exponential curve
    popt, pcov = curve_fit(exponential_func, x, y)

    a = popt[0]
    b = popt[1]
    print(f"Exponential Equation: y = {a:.2f} * exp({b:.2f} * x)")


    # To predict with the exponential function
    new_x = np.array(6)
    predicted_y = exponential_func(new_x,a,b)
    print(f"Predicted y for x = 6: {predicted_y:.2f}")

    # Plotting the fitted curve and original data
    x_curve = np.linspace(min(x),max(x),100)
    y_curve = exponential_func(x_curve,a,b)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')
    plt.plot(x_curve, y_curve, label='Fitted Exponential Curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Exponential Curve Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()
    ```

    In this example, we use *scipy.optimize.curve_fit* to fit an exponential model. The function *curve_fit* returns optimal parameters, along with a covariance matrix which is helpful for statistical analysis. We then have an equation which can be used for prediction. We also use matplotlib to visualize the fitted curve. This process is highly useful when there is some domain knowledge regarding the nature of the data.

**Additional Considerations and Recommended Resources**

*   **Data Cleaning and Preprocessing:** Data taken from a graph will likely involve some degree of noise or inaccuracies. Cleaning the data, checking for and excluding outliers and interpolating missing data points may be required before applying a model.
*   **Model Evaluation:** Assess the performance of your chosen model via metrics such as R-squared, Mean Squared Error (MSE), or by visual inspection to determine the quality of the fit.
*   **Domain Knowledge:** Whenever possible, leverage existing understanding about the expected behavior of the data to guide the selection of a model.
*   **Resources:** Explore books on numerical analysis, statistical modeling, and signal processing. Furthermore, texts on data analysis and machine learning contain extensive information and techniques relating to function fitting and modeling. Official documentation of the libraries used (*NumPy*, *scikit-learn*, *SciPy*, *Matplotlib*) is another fundamental resource.

Function recovery from a graph is a complex and iterative process. Selecting the correct model based on visual examination of the data and using appropriate techniques such as those outlined are the keys to successful recovery and approximation. The quality of the recovered function will depend on the fidelity of data available, and the underlying structure of the data itself.
