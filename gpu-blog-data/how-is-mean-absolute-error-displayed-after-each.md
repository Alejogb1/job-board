---
title: "How is mean absolute error displayed after each round in LightGBM?"
date: "2025-01-30"
id: "how-is-mean-absolute-error-displayed-after-each"
---
The LightGBM library, while robust in its default behavior, doesn't intrinsically display the mean absolute error (MAE) after each boosting round directly within its core training loop.  This contrasts with metrics like training loss, which are often monitored internally.  My experience working on high-frequency trading models heavily reliant on LightGBM necessitated the explicit calculation and logging of MAE during training.  This is because real-time performance monitoring was crucial for identifying model drift and potential issues early in the training process.  Understanding how to achieve this requires a combination of LightGBM's callback functionality and custom metric implementation.


1. **Clear Explanation:**

LightGBM's primary focus is on efficient model training, prioritizing speed and predictive accuracy. While it provides access to various evaluation metrics during the training process through its `eval_set` parameter,  MAE is not directly outputted iteratively.  To obtain MAE after each round, a custom callback function needs to be implemented. This function intercepts the evaluation process, calculates MAE on the validation set(s), and logs the result. This logging can occur using standard Python logging mechanisms, custom file writing, or even real-time visualizations using libraries like `matplotlib`.  The custom function leverages LightGBM's internal evaluation procedures, but extends it to specifically compute and record MAE at each iteration. This contrasts with simply relying on the default output, which primarily focuses on other metrics like the objective function itself.  The key is to understand that LightGBM provides the building blocks, but the specific metric visualization needs explicit coding.

2. **Code Examples with Commentary:**

**Example 1: Basic MAE Callback using Python Logging:**

```python
import lightgbm as lgb
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mae_callback(env):
    if env.iteration % 10 == 0:  # Log MAE every 10 iterations for efficiency
        y_pred = env.model.predict(env.valid_sets[0].data)
        mae = mean_absolute_error(env.valid_sets[0].label, y_pred)
        logging.info(f"Iteration {env.iteration}: MAE = {mae}")


# ... (Data loading and preprocessing) ...

params = {
    'objective': 'regression',
    'metric': 'l1', #l1 is MAE, included for LightGBM's internal evaluation
    'boosting_type': 'gbdt'
}

train_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_val, y_val)

gbm = lgb.train(params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[mae_callback])

```

**Commentary:** This example uses the `logging` module for straightforward output. The `mae_callback` function is triggered after each boosting round. It retrieves predictions from the validation set using `env.model.predict` and then computes MAE using `sklearn.metrics.mean_absolute_error`.  The logging level and frequency can be adjusted. Note that including `'metric': 'l1'` in the parameters enhances performance by utilizing LightGBM's internal MAE calculation for the primary evaluation metric, though the callback still performs a separate calculation for logging purposes, to ensure accuracy and independance from LightGBM's internal implementation details.

**Example 2:  MAE Callback with Custom File Writing:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error

def mae_callback(env):
    y_pred = env.model.predict(env.valid_sets[0].data)
    mae = mean_absolute_error(env.valid_sets[0].label, y_pred)
    with open('mae_log.txt', 'a') as f:
        f.write(f"Iteration {env.iteration}: MAE = {mae}\n")

# ... (Data loading and preprocessing, similar to Example 1) ...

# ... (Training, similar to Example 1, but using mae_callback) ...
```

**Commentary:** This example writes the MAE values to a file named `mae_log.txt`.  Appending (`'a'`) allows for continuous logging across multiple runs.  This approach avoids potential performance overhead associated with frequent logging calls and offers a more persistent record of the training process which can later be analyzed.  This method is preferred when dealing with large datasets or long training times where logging to the console may become unwieldy.

**Example 3:  MAE Callback with Early Stopping:**

```python
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error

def mae_callback(env):
    y_pred = env.model.predict(env.valid_sets[0].data)
    mae = mean_absolute_error(env.valid_sets[0].label, y_pred)
    return {'mae': mae}  # Return MAE as a dictionary for early stopping

# ... (Data loading and preprocessing) ...

params = {
    'objective': 'regression',
    'metric': 'l1'
}

train_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_val, y_val)

gbm = lgb.train(params,
                train_data,
                num_boost_round=1000, #Increased rounds to showcase early stopping
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10, first_metric_only=True, verbose=True), mae_callback],
                feval=mae_callback)
```

**Commentary:** This example integrates early stopping based on the MAE.  The `mae_callback` now returns a dictionary containing the MAE.  LightGBM's built-in `early_stopping` callback monitors this returned MAE and stops training if the MAE doesn't improve for a specified number of rounds.  This dramatically improves training efficiency by avoiding unnecessary iterations, particularly crucial when dealing with extensive feature sets or complex models. Note the use of `first_metric_only=True` to ensure only the MAE is considered for early stopping.


3. **Resource Recommendations:**

The LightGBM documentation, particularly sections detailing callbacks and custom metrics.  The `sklearn.metrics` module for various evaluation metrics.  Standard Python logging documentation for efficient log management.  For more advanced visualization and analysis, consider exploring data analysis libraries like Pandas and plotting libraries like Matplotlib or Seaborn.  Familiarizing oneself with the underlying principles of gradient boosting machines and regression analysis is highly recommended for a comprehensive understanding.
