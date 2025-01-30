---
title: "Can unlabeled data be used for testing?"
date: "2025-01-30"
id: "can-unlabeled-data-be-used-for-testing"
---
No, unlabeled data cannot directly serve as a substitute for labeled data in the *conventional* evaluation of machine learning models, specifically within the context of supervised learning. However, the utility of unlabeled data in model *assessment* is nuanced and far broader than simple testing, often playing a critical role in the development process and model robustness.

My experience, spanning several years architecting machine learning solutions in diverse industries ranging from medical imaging to financial fraud detection, consistently highlights the limitations of solely relying on accuracy metrics derived from labeled data. While metrics like precision, recall, and F1-score are essential for quantifying a model's performance on data resembling what it was trained on, they fail to capture its generalization ability and its robustness to real-world variability. This is precisely where unlabeled data becomes invaluable, not as a *testing* mechanism in the classical sense, but as a *validation* tool and a vehicle for identifying potential failure points.

The primary purpose of a test set, within a supervised paradigm, is to assess how well a model will generalize to unseen data that still adheres to the underlying distribution of the labeled data. It assumes that the labels within that test set are *correct*, *representative*, and indicative of the true operational environment. Unlabeled data, by its very nature, lacks this crucial element. We cannot directly calculate accuracy or any other performance metric because we lack the ground truth against which to compare the model's predictions.

However, this absence of labels does not render unlabeled data useless for model assessment. Instead, its value lies in its ability to expose situations where the model might behave unexpectedly, identify potential biases, and measure the model’s sensitivity to out-of-distribution data. The techniques employed to analyze unlabeled data usually focus on analyzing the model's output, or the intermediate representations learned by the model, instead of calculating accuracy scores directly.

Here are three examples of how I have practically used unlabeled data for insightful model assessment, along with code commentary to clarify implementation details.

**Example 1: Anomaly Detection using Reconstruction Error**

Consider a convolutional autoencoder, trained for image denoising using a set of labeled images. The primary objective was to remove noise artifacts from medical X-ray images. The training data contained clear X-ray images with synthetic noise added, with the clean image serving as the label. After achieving high accuracy on the labeled test set, we fed the autoencoder completely novel, unlabeled X-ray images from patients with a new, previously unseen, medical condition. We didn't have labels for these new conditions, but we were keenly interested in the quality of the reconstructions.

```python
import tensorflow as tf
import numpy as np

def reconstruction_error(model, image):
    """Calculates the pixel-wise mean squared error between the original
    image and its reconstruction by the autoencoder.

    Args:
    model: A trained tensorflow autoencoder model.
    image: A single input image as a numpy array.

    Returns:
      A scalar, representing the reconstruction error.
    """
    image_input = np.expand_dims(image, axis=0)  # Add batch dimension
    reconstructed = model.predict(image_input)[0] # Remove batch dimension again
    return tf.reduce_mean(tf.square(image - reconstructed)).numpy()


def analyze_unlabeled_images(model, unlabeled_images, threshold=0.01):
    """Detects potential anomalies in unlabeled images based on reconstruction error.

      Args:
        model: A trained autoencoder model.
        unlabeled_images: A list of unlabeled image numpy arrays.
        threshold: The threshold above which an image is considered an anomaly.

      Returns:
        A list of indices of images which are flagged as anomalous.
    """
    anomalous_indices = []
    for index, image in enumerate(unlabeled_images):
        error = reconstruction_error(model, image)
        if error > threshold:
            anomalous_indices.append(index)
        print(f'Error for image {index}: {error}')
    return anomalous_indices


# Example usage assuming 'model' is the trained autoencoder, and unlabeled_data is a list
anomalous_images_indices = analyze_unlabeled_images(model, unlabeled_data)
print(f'Anomalous images: {anomalous_images_indices}')
```

This Python code snippet defines two functions. The first, `reconstruction_error`, computes the mean squared error (MSE) between an original image and its reconstruction by the autoencoder. This MSE measures the model's ability to reproduce the input data. The `analyze_unlabeled_images` function then leverages this error to flag images exceeding a predefined `threshold`, effectively identifying those that are "unexpected" by the model. In practice, this flagged images were often those containing the new medical condition, demonstrating that the model was underperforming on data outside of its training distribution. While this isn't a formal test, it's an essential step in understanding model limitations and potentially discovering biases in data.

**Example 2: Output Distribution Analysis**

In another project, I developed a natural language processing model for sentiment analysis, trained on labeled movie reviews. After achieving satisfactory results on a held-out labeled test set, we applied the model to a large corpus of unlabeled social media posts. While we didn’t know the true sentiment for each post, analyzing the distribution of the model’s predicted sentiment scores across this unlabeled corpus revealed a significant discrepancy. The scores tended to cluster heavily at the extremes, indicating high confidence positive and negative predictions. However, we knew that social media sentiment is often nuanced and highly variable. This observation indicated a potential overconfidence bias learned from the training data.

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_predictions_distribution(model, unlabeled_texts):
    """Analyzes the distribution of model predicted sentiment scores across unlabeled texts.
        Args:
            model: A trained NLP sentiment analysis model.
            unlabeled_texts: A list of unlabeled text strings.
    """
    predictions = model.predict(unlabeled_texts)
    sentiment_scores = predictions[:,1] # Assuming positive class is index 1.
    plt.hist(sentiment_scores, bins = 20)
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Sentiment Scores on Unlabeled Data")
    plt.show()
    print(f'Mean sentiment score: {np.mean(sentiment_scores)}')
    print(f'Standard Deviation: {np.std(sentiment_scores)}')

# Assuming 'model' is the sentiment analysis model, and unlabeled_texts is a list of strings.
analyze_predictions_distribution(model, unlabeled_texts)
```

This code example takes a trained sentiment analysis model and applies it to a corpus of unlabeled text data, then visualizes the resulting distribution of sentiment scores using a histogram. Additionally, we calculated the mean and standard deviation. If the trained model was behaving correctly, this histogram would likely show a less polarised distribution, with scores distributed across the full spectrum of sentiments.  A highly polarized distribution, as was observed, suggested that the model was extrapolating beyond its training data, highlighting a potentially critical issue.  This was not something the test set, with its balanced sentiment distribution, was able to detect.

**Example 3: Feature Space Visualization**

In a fraud detection system, trained on labeled transaction data, I utilized a dimensionality reduction technique (like t-SNE) to visualize the feature representations learned by a deep neural network. After a successful test on labeled data, I applied the model to a set of unlabeled transactions. These points were then embedded in the same reduced space alongside both the training data and the test data.  Clusters of unlabeled data points that appeared distant from both the labeled training data and test data clusters indicated potentially fraudulent activities of a kind not present in the original labeled datasets. This analysis does not provide a concrete score, but it gives valuable qualitative feedback on how well the model understands new data patterns.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def visualize_feature_space(model, training_data, test_data, unlabeled_data):
  """Visualizes the feature space of a model using t-SNE.

  Args:
        model: A trained machine learning model.
        training_data: The labeled training data.
        test_data: The labeled test data.
        unlabeled_data: The unlabeled data to analyze.
  """
  training_features = model.predict(training_data)
  test_features = model.predict(test_data)
  unlabeled_features = model.predict(unlabeled_data)

  all_features = np.concatenate((training_features, test_features, unlabeled_features), axis=0)
  tsne = TSNE(n_components=2, random_state=0)
  tsne_embedding = tsne.fit_transform(all_features)

  plt.figure(figsize=(10, 8))
  plt.scatter(tsne_embedding[:len(training_data), 0], tsne_embedding[:len(training_data), 1],
               label='Training Data', marker='o', s = 10)
  plt.scatter(tsne_embedding[len(training_data):len(training_data) + len(test_data), 0],
               tsne_embedding[len(training_data):len(training_data) + len(test_data), 1],
               label='Test Data', marker='x', s = 10)
  plt.scatter(tsne_embedding[len(training_data) + len(test_data):, 0],
               tsne_embedding[len(training_data) + len(test_data):, 1],
               label='Unlabeled Data', marker='+', s = 10)
  plt.title("Feature Space Visualization with t-SNE")
  plt.xlabel('t-SNE Dimension 1')
  plt.ylabel('t-SNE Dimension 2')
  plt.legend()
  plt.show()


# Assuming 'model' is a trained model, and training_data, test_data, and unlabeled_data are available.
visualize_feature_space(model, training_data, test_data, unlabeled_data)
```
This snippet illustrates how t-SNE dimensionality reduction can be employed to project the model’s internal feature representations onto a 2D plane, allowing for visual assessment of data distributions and identifying anomalies within unlabeled data.  This reveals if the model generalizes well to data dissimilar from the training set and allows human experts to identify potentially problematic data.

In conclusion, while unlabeled data cannot provide the quantitative performance metrics of a typical labeled test set, it is a potent tool for assessing model behavior, identifying potential vulnerabilities, and fostering the development of more robust and reliable models. Relying solely on traditional test sets can create a false sense of security, particularly in the face of constantly changing, real-world data. Incorporating strategies to analyze unlabeled data is not just advantageous, it is often essential for truly effective machine learning deployments.  For further study, I recommend exploring literature on unsupervised learning techniques, anomaly detection methods, and the concepts of model drift and out-of-distribution generalization. These offer a deeper understanding of how unlabeled data can be leveraged for model improvement.
