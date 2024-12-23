---
title: "How can ML predictions be created and recorded for new, unseen data?"
date: "2024-12-23"
id: "how-can-ml-predictions-be-created-and-recorded-for-new-unseen-data"
---

Alright,  I’ve certainly spent enough late nights facing the challenge of extending machine learning model predictions to completely new data. It’s one thing to train a model on a well-defined dataset; it's another to handle the real-world scenario where the model needs to make predictions on data it has never encountered during training. The process isn’t just a simple case of `model.predict()`; there are considerations of data handling, transformation consistency, and reliable recording practices.

The core principle here is to ensure consistency between the data preparation pipeline used during model training and the data preparation applied to new, unseen data. If your model was trained on normalized features, any new data must be normalized using the *same* parameters. Likewise for categorical encodings, imputation, and feature engineering; every step of the process must be mirrored precisely. Neglecting this aspect will undoubtedly lead to degraded prediction quality, regardless of how well the model performed during training.

Let's consider a practical example, and how I approached it in the past. We were building a customer churn prediction system, and initial training data included a ‘customer_age’ feature. We used min-max scaling for this variable. When a new customer came along, our initial approach didn’t account for the scaling, resulting in predictions that were wildly off. The fix, of course, was to store the min and max values from the training set, and use those during prediction for new data points.

This brings me to the first crucial aspect: **pipeline management**. It's insufficient to just apply the model. You need to encapsulate the entire data preprocessing and prediction pipeline into a single, repeatable function or class. This ensures that the same transformations used during training are applied consistently to new data. A common approach is to use a class with methods for preprocessing, making predictions, and recording the result.

Here’s a python snippet illustrating this, using `scikit-learn` as an example:

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd


class PredictionPipeline:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.feature_names = ['age', 'income', 'usage']

    def preprocess(self, data):
        df = pd.DataFrame(data, columns=self.feature_names)
        scaled_data = self.scaler.transform(df)
        return scaled_data

    def predict(self, data):
        preprocessed_data = self.preprocess(data)
        prediction = self.model.predict(preprocessed_data)
        return prediction

    def record_prediction(self, input_data, prediction, output_filepath="predictions.csv"):
        df_input = pd.DataFrame(input_data, columns=self.feature_names)
        df_output = pd.DataFrame({'prediction': prediction})
        df_combined = pd.concat([df_input, df_output], axis=1)
        df_combined.to_csv(output_filepath, mode='a', header=not (os.path.exists(output_filepath)), index=False)


#Example Usage
if __name__ == "__main__":
    # Simulated Training Data
    train_data = np.array([[25, 50000, 10],
                           [35, 60000, 20],
                           [45, 70000, 30],
                           [55, 80000, 40]])
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    model = LogisticRegression()
    model.fit(scaled_train_data, [0, 1, 1, 0])  # Simulated labels for training

    pipeline = PredictionPipeline(model, scaler)
    new_data = np.array([[30, 55000, 25], [60, 90000, 50]])
    predictions = pipeline.predict(new_data)
    pipeline.record_prediction(new_data, predictions)
```
In this snippet, the `PredictionPipeline` class handles the transformation and prediction steps. The constructor saves a trained model, and the fitted scaler and keeps the feature names. The `preprocess` method uses the fitted scaler from training. The `predict` method makes the predictions. Lastly `record_prediction` adds the new data with its prediction and writes that to a csv file.

Secondly, the *recording* aspect is critically important. It's not enough to just make predictions; you need to log and record them systematically. This provides valuable traceability and allows you to monitor model performance over time. Storing the original input data alongside the model predictions is absolutely essential for debugging and retrospective analysis. I’ve learned the hard way that not saving input features makes understanding prediction failures nearly impossible. I’d recommend saving this data into a database table, or a time-series data store if that fits your project. Don’t rely on console logging alone.

Building on that idea, here’s a slightly more refined version that integrates a simple data recording example that uses a timestamp:
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from datetime import datetime
import os

class AdvancedPredictionPipeline:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.feature_names = ['feature1', 'feature2', 'feature3']

    def preprocess(self, data):
        df = pd.DataFrame(data, columns=self.feature_names)
        scaled_data = self.scaler.transform(df)
        return scaled_data

    def predict(self, data):
        preprocessed_data = self.preprocess(data)
        prediction = self.model.predict(preprocessed_data)
        return prediction

    def record_prediction(self, input_data, prediction, output_filepath="advanced_predictions.csv"):
        df_input = pd.DataFrame(input_data, columns=self.feature_names)
        df_output = pd.DataFrame({'prediction': prediction})
        timestamp = datetime.now().isoformat()
        df_timestamp = pd.DataFrame({'timestamp': [timestamp] * len(df_input)})
        df_combined = pd.concat([df_timestamp, df_input, df_output], axis=1)
        df_combined.to_csv(output_filepath, mode='a', header=not (os.path.exists(output_filepath)), index=False)


if __name__ == "__main__":
    # Simulated Training Data
    train_data = np.array([[1.2, 3.4, 5.6],
                           [2.1, 4.3, 6.5],
                           [3.5, 5.1, 7.2],
                           [4.3, 6.2, 8.1]])
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    model = RandomForestClassifier(random_state=42)
    model.fit(scaled_train_data, [0, 1, 0, 1])  # Simulated labels for training

    pipeline = AdvancedPredictionPipeline(model, scaler)
    new_data = np.array([[2.5, 4.8, 6.9], [3.9, 5.8, 7.6]])
    predictions = pipeline.predict(new_data)
    pipeline.record_prediction(new_data, predictions)
```

This implementation includes a timestamp for each prediction recorded into a CSV file and has added some better feature names. This makes the log data more useful for future analysis.

Third, consider handling model drift. In many real-world cases, the data distribution can change over time, causing a decrease in model performance. This phenomenon is called model drift. Monitoring prediction performance on new data over time and potentially retraining the model regularly will mitigate the impacts of drift. To identify drift, one needs a way to compare the distributions of features, a Kolmogorov-Smirnov test may help here, or the concept drift detectors described in "Adaptive Data Mining: Techniques and Applications" by Richard J. Butz. It’s essential to think about a system for model re-training and redeployment when new data are seen to no longer fit the current distribution. This also requires a robust strategy for versioning models so a past model can be used if the new one shows a degradation of performance.

Finally, here's a snippet showcasing prediction recording for a batch of new data, incorporating error handling and model versioning using a simple version string:
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime
import os
import traceback

class BatchPredictionPipeline:
    def __init__(self, model, preprocessor, version="v1.0"):
        self.model = model
        self.preprocessor = preprocessor
        self.version = version
        self.feature_names = ["x"]

    def preprocess(self, data):
        df = pd.DataFrame(data, columns=self.feature_names)
        preprocessed_data = self.preprocessor.transform(df)
        return preprocessed_data

    def predict(self, data):
        try:
            preprocessed_data = self.preprocess(data)
            prediction = self.model.predict(preprocessed_data)
            return prediction
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return None

    def record_predictions(self, input_data, predictions, output_filepath="batch_predictions.csv"):
        timestamp = datetime.now().isoformat()
        df_input = pd.DataFrame(input_data, columns=self.feature_names)
        if predictions is not None:
            df_output = pd.DataFrame({'prediction': predictions})
            df_version = pd.DataFrame({'model_version': [self.version] * len(df_input)})
            df_timestamp = pd.DataFrame({'timestamp': [timestamp] * len(df_input)})
            df_combined = pd.concat([df_timestamp, df_version, df_input, df_output], axis=1)

        else:
            df_output = pd.DataFrame({'prediction': [None]*len(df_input)})
            df_version = pd.DataFrame({'model_version': [self.version] * len(df_input)})
            df_timestamp = pd.DataFrame({'timestamp': [timestamp] * len(df_input)})
            df_combined = pd.concat([df_timestamp, df_version, df_input, df_output], axis=1)

        df_combined.to_csv(output_filepath, mode='a', header=not (os.path.exists(output_filepath)), index=False)


if __name__ == "__main__":
    # Simulate Training Data
    train_data = np.array([[1], [2], [3], [4]])
    preprocessor = PolynomialFeatures(degree=2)
    transformed_train_data = preprocessor.fit_transform(train_data)
    model = LinearRegression()
    model.fit(transformed_train_data, [2, 5, 10, 17]) #Example training values

    pipeline = BatchPredictionPipeline(model, preprocessor, version="v2.0")

    # New data as a batch
    new_data = np.array([[5], [6], [7], [8]])
    predictions = pipeline.predict(new_data)
    pipeline.record_predictions(new_data, predictions)

    # Simulate an error with new data
    new_data_error = np.array([[None], [10], [11], [12]])
    predictions_error = pipeline.predict(new_data_error)
    pipeline.record_predictions(new_data_error, predictions_error)
```
Here, we introduced a version tag to the prediction pipeline, along with the recording of a null or ‘None’ prediction when an exception is found during prediction and an error stack trace is printed to the console.

In summary, correctly predicting and recording new data involves a holistic approach that includes a well-defined pipeline, robust recording mechanisms, and continuous monitoring for potential model drift. This is less about using a single function, and much more about building a system that can reliably handle new information.
For additional resources, I’d recommend checking “Machine Learning Engineering” by Andriy Burkov, or perhaps "Designing Machine Learning Systems" by Chip Huyen for practical insights in putting such systems together. They provide good guidance beyond single model development.
