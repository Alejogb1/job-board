---
title: "How can I efficiently compare deep learning model performance changes after adding or removing multiple data cleaning steps?"
date: "2024-12-23"
id: "how-can-i-efficiently-compare-deep-learning-model-performance-changes-after-adding-or-removing-multiple-data-cleaning-steps"
---

, let's unpack this. I've certainly been in that boat, many times, trying to decipher the impact of various data cleaning permutations on model performance. It’s rarely a straightforward path, and simply looking at final evaluation metrics can be misleading. Instead, we need a systematic approach. Over the years, I've refined a method that leans heavily on statistical rigor while staying practical in a development environment. It's not about finding the absolute "best" set of cleaning steps, but rather understanding *how* these changes influence the model's behavior and performance.

The first critical point is to understand that changes can manifest in subtle ways. A marginal increase in overall accuracy, for example, might mask significant variations in performance across different subgroups within your data. Therefore, it is essential to conduct thorough testing that is not merely limited to aggregated performance statistics.

Here's the breakdown of what I've found effective:

**1. Establish a Robust Baseline and Control:**

Before changing anything, we must define a rock-solid baseline. This involves selecting a representative subset of your data that isn't overly biased (a common pitfall). A good approach is to employ stratified sampling if you know your dataset has class imbalances, or if you’re using a regression task, ensure representation across your target variable’s range. Train your model on this initial data with minimal pre-processing, and record its performance metrics, focusing on more than just the overall accuracy. Include metrics such as precision, recall, f1-score, and area under the receiver operating characteristic (auroc) if you have a classification task. For regression consider root mean squared error (rmse), mean absolute error (mae), and perhaps r-squared. Be sure to log these metrics precisely. This serves as the control against which all subsequent changes will be measured.

**2. Incremental Changes and Controlled Experiments:**

Avoid making wholesale changes to your cleaning pipeline all at once. Instead, implement alterations one at a time (or in small, carefully considered groups). For each change, train your model, and critically assess the performance gains (or losses) against your control baseline. This controlled experimentation is crucial for isolating the impact of each data cleaning step. Keep track of not only the changed steps, but also the performance of the model by comparing different metrics from the control metrics. Be meticulous in logging changes and experiments.

**3. Statistical Significance Testing:**

It’s important to determine whether the performance differences are statistically significant. A 0.5% increase in accuracy might be noise, and there is no reason to overfit towards a possibly meaningless improvement. Here, we move beyond simply *observing* differences. Common approaches involve hypothesis testing, such as a t-test if you have two conditions, or ANOVA when comparing multiple cleaning pipelines if your assumptions are met. For scenarios where the data might not meet the normality assumption of these tests, non-parametric options like the Mann-Whitney U test or Kruskal-Wallis test can be useful. Note, statistical tests alone should not be your only criteria. Substantive understanding of the data should also be critical, but testing provides evidence. This step gives you concrete evidence to support your claims that these changes are meaningful and not just random fluctuations.

**4. Subgroup Analysis and Error Investigation:**

Pay close attention to the metrics beyond aggregated summaries. Does the performance gain come mostly in one particular group while simultaneously decreasing in another? If so, this needs careful consideration and suggests that a given cleaning method is not uniformly advantageous. Dig deeper into error analysis. What kinds of examples are these changes improving? What are they hurting? Tools that visualize model performance can be incredibly insightful, such as confusion matrices for classification tasks, and error distribution plots for regression. These help you spot patterns and provide additional insight. You might be surprised to find that certain data points are now misclassified due to your cleaning process. This insight could suggest potential bugs in your cleaning method.

**5. Code Implementation for Tracking:**

Here is an example in Python, assuming we are working with a classification problem. The first snippet focuses on the baseline training and evaluation.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, X_test, y_test, prefix):
    y_pred = model.predict(X_test)
    logging.info(f"{prefix} Accuracy: {accuracy_score(y_test, y_pred)}")
    logging.info(f"{prefix} Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0)}")
    logging.info(f"{prefix} Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0)}")
    logging.info(f"{prefix} F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0)}")
    return y_pred

def baseline_training(df, target_column, feature_columns):

    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    y_pred_baseline = evaluate_model(model, X_test, y_test, "Baseline")

    return model, X_test, y_test, y_pred_baseline

if __name__ == '__main__':
    # Placeholder - replace with your actual data loading and column assignments
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)
    target_col = 'target'
    feature_cols = ['feature1', 'feature2']
    baseline_model, baseline_x_test, baseline_y_test, baseline_pred = baseline_training(df, target_col, feature_cols)

```

This code creates our baseline metrics. The next code snippet showcases introducing a cleaning step, in this case, feature scaling using standardization.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def scaled_training(df, target_column, feature_columns):
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(solver='liblinear', random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred_scaled = evaluate_model(pipeline, X_test, y_test, "Scaled")

    return pipeline, X_test, y_test, y_pred_scaled

if __name__ == '__main__':
    # Using the dataframe created before
    scaled_model, scaled_x_test, scaled_y_test, scaled_pred = scaled_training(df, target_col, feature_cols)
```

The final snippet will show the statistical test to determine if the results are significant.

```python
from scipy.stats import ttest_rel

def compare_models(y_pred_baseline, y_pred_scaled, y_test):
    # Convert predictions to numpy arrays and create a binary mask to only consider correctly classified samples
    y_test = np.array(y_test)
    y_pred_baseline = np.array(y_pred_baseline)
    y_pred_scaled = np.array(y_pred_scaled)
    correct_mask_baseline = (y_pred_baseline == y_test)
    correct_mask_scaled = (y_pred_scaled == y_test)
    # Correctly classified predictions get a score of 1, incorrect get a score of 0
    correct_baseline = np.where(correct_mask_baseline, 1, 0)
    correct_scaled = np.where(correct_mask_scaled, 1, 0)

    # Paired t-test
    t_statistic, p_value = ttest_rel(correct_baseline, correct_scaled)
    logging.info(f"Paired t-test statistic: {t_statistic}, p-value: {p_value}")
    if p_value < 0.05:
        logging.info("There is a statistically significant difference in the prediction performance between the baseline model and the model after scaling.")
    else:
        logging.info("There is not a statistically significant difference in the prediction performance between the baseline model and the model after scaling.")
if __name__ == '__main__':
    compare_models(baseline_pred, scaled_pred, baseline_y_test)
```

In this last example, I decided to compare accuracy scores by creating a binary mask of only correctly classified samples, then running a t-test to determine if the results are significant. While this comparison isn't of all metrics, it showcases one method for statistically comparing performance. In practice, this process is often done in conjunction with other forms of statistical tests, such as those mentioned earlier.

**Further Considerations:**

For a deep dive into statistical testing, I recommend reading “All of Statistics” by Larry Wasserman. If you’re interested in more practical model evaluation, “Applied Predictive Modeling” by Max Kuhn and Kjell Johnson is an excellent choice. Additionally, the scikit-learn documentation for model selection and evaluation is a constant resource for code examples and best practices.

Finally, remember that there is no single, perfect way to clean your data. The "best" cleaning pipeline is intimately tied to your specific problem and dataset. Continuous experimentation, coupled with rigorous statistical analysis, is vital for gaining a granular understanding of how each step impacts your model’s behaviour. This allows for making informed decisions rather than relying on trial and error alone.
