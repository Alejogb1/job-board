---
title: "How can multiple, differently-trained ML models be combined into a single predictive model in SageMaker?"
date: "2024-12-23"
id: "how-can-multiple-differently-trained-ml-models-be-combined-into-a-single-predictive-model-in-sagemaker"
---

Alright,  The challenge of combining multiple machine learning models, each trained perhaps on slightly different datasets or using differing algorithms, into a unified predictor within SageMaker is definitely a nuanced one, and something I've personally grappled with in several projects. It's not just about stacking things together; it's about achieving a robust and often more accurate prediction by leveraging the strengths of each individual model.

The core principle we're focusing on here is *ensemble learning*. This approach isn't a single technique but rather a family of methods designed to improve predictive performance by combining the predictions of multiple models. I've seen firsthand how moving past a single model strategy can significantly boost overall system performance, especially in scenarios where you're dealing with complex datasets or have a variety of features that different models can excel at capturing. SageMaker offers several ways to implement ensemble methods. We will focus on three common strategies: averaging, voting, and stacking, each with its own set of benefits and implementation details.

Let’s begin with *averaging*, arguably the simplest approach. Here, we obtain predictions from each of our individual models and calculate the average of these predictions to arrive at the final output. This is incredibly useful when the individual models are relatively equal in performance and when we expect random noise in their predictions to cancel out through averaging.

```python
import numpy as np
import sagemaker

def averaging_ensemble(model_endpoints, input_data):
    """
    Combines predictions from multiple sagemaker models via averaging.

    Args:
      model_endpoints (list): A list of sagemaker model endpoint names.
      input_data (np.ndarray): Input data as a numpy array.

    Returns:
      np.ndarray: The average prediction across all models.
    """

    sagemaker_session = sagemaker.Session()
    predictor = sagemaker.predictor.Predictor
    predictions = []
    for endpoint in model_endpoints:
        pred = predictor(endpoint_name=endpoint, sagemaker_session=sagemaker_session,
                         serializer=sagemaker.serializers.NumpySerializer(),
                         deserializer=sagemaker.deserializers.NumpyDeserializer())
        prediction = pred.predict(input_data)
        predictions.append(prediction)

    return np.mean(predictions, axis=0)


# Example Usage:
model_endpoints = ['model1-endpoint','model2-endpoint', 'model3-endpoint'] # Replace with your endpoint names
sample_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)  # Example data
final_prediction = averaging_ensemble(model_endpoints, sample_data)
print(final_prediction)
```

In this code, each of the SageMaker endpoints specified in `model_endpoints` is queried using the SageMaker `Predictor`. The raw predictions are appended to a list, and finally, their mean is computed along axis 0 (across the models, not down the dataset rows). Note the explicit specification of serializers and deserializers, ensuring that data is correctly formatted for each model endpoint. This example assumes all models output numerical predictions.

The second strategy is *voting*. Voting is most effective when dealing with classification problems. For hard voting, each model predicts a class label, and the final prediction is the class label that receives the majority of votes. For soft voting, each model predicts probabilities for each class, and the final prediction is the class with the highest average probability.

```python
import numpy as np
import sagemaker
from scipy.stats import mode

def voting_ensemble(model_endpoints, input_data, voting_type='hard'):
    """
    Combines predictions from multiple sagemaker models via voting.

    Args:
      model_endpoints (list): A list of sagemaker model endpoint names.
      input_data (np.ndarray): Input data as a numpy array.
      voting_type (str): 'hard' or 'soft', indicating type of voting.

    Returns:
      np.ndarray: The final class label or probability prediction.
    """

    sagemaker_session = sagemaker.Session()
    predictor = sagemaker.predictor.Predictor
    predictions = []
    for endpoint in model_endpoints:
        pred = predictor(endpoint_name=endpoint, sagemaker_session=sagemaker_session,
                         serializer=sagemaker.serializers.NumpySerializer(),
                         deserializer=sagemaker.deserializers.NumpyDeserializer())
        prediction = pred.predict(input_data)
        predictions.append(prediction)

    if voting_type == 'hard':
       # In hard voting, pick most frequent class label
        return mode(np.array(predictions), axis=0)[0][0]
    elif voting_type == 'soft':
        # In soft voting, average the class probabilities
       return np.mean(predictions, axis=0)
    else:
        raise ValueError("voting_type must be 'hard' or 'soft'")


# Example Usage:
model_endpoints = ['classifier1-endpoint','classifier2-endpoint', 'classifier3-endpoint']
sample_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)  # Example data
hard_vote_prediction = voting_ensemble(model_endpoints, sample_data, voting_type='hard')
print("Hard Vote:", hard_vote_prediction)
soft_vote_prediction = voting_ensemble(model_endpoints, sample_data, voting_type='soft')
print("Soft Vote:", soft_vote_prediction)

```

This snippet demonstrates both hard and soft voting, choosing the appropriate strategy based on the `voting_type` parameter. For 'hard' voting, we use `scipy.stats.mode` to determine the most frequent prediction for each input instance. For 'soft' voting, we return the average class probability, analogous to averaging with a different interpretation.

Finally, the more sophisticated approach of *stacking* involves training a meta-model or blender on the predictions of the base models. The base models generate predictions, which then become the input features for the meta-model, which makes the final prediction. This method can potentially capture complex relationships between the base model predictions that would be lost using simple averaging or voting.

```python
import numpy as np
import sagemaker
from sklearn.linear_model import LogisticRegression

def stacking_ensemble(model_endpoints, input_data, meta_model=None, fit_meta=True):
    """
    Combines predictions from multiple sagemaker models via stacking.

    Args:
      model_endpoints (list): A list of sagemaker model endpoint names.
      input_data (np.ndarray): Input data as a numpy array.
      meta_model: a meta-model instance, optional, if provided will use given metamodel, else will use a new logreg model.
      fit_meta: indicates whether the metamodel needs training.

    Returns:
      np.ndarray: The stacked final prediction.
    """
    sagemaker_session = sagemaker.Session()
    predictor = sagemaker.predictor.Predictor
    base_predictions = []
    for endpoint in model_endpoints:
        pred = predictor(endpoint_name=endpoint, sagemaker_session=sagemaker_session,
                         serializer=sagemaker.serializers.NumpySerializer(),
                         deserializer=sagemaker.deserializers.NumpyDeserializer())
        prediction = pred.predict(input_data)
        base_predictions.append(prediction)
    base_predictions = np.array(base_predictions).transpose((1, 0, 2)).squeeze()

    if meta_model is None:
       meta_model = LogisticRegression() #default meta model if none is provided

    if fit_meta:
      # In real scenarios one would split their data into train and test
      # and train the base models on a train set and the meta-model on a 
      # validation set using out of fold predictions
      # but here just use simple fit
       dummy_labels = np.random.randint(0,2,size=base_predictions.shape[0])
       meta_model.fit(base_predictions, dummy_labels)

    return meta_model.predict(base_predictions)

# Example Usage:
model_endpoints = ['model_a-endpoint', 'model_b-endpoint']
sample_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)  # Example data
stacked_prediction = stacking_ensemble(model_endpoints, sample_data, fit_meta=True)
print(stacked_prediction)
```

Here, the outputs of each model are treated as features for a meta-model. In this code, we use `sklearn.linear_model.LogisticRegression` as a demonstrative meta-model. The core of any proper stacking implementation is the concept of out-of-fold predictions: during the training process, meta-model training should utilize base model predictions generated from data not used to train the base model themselves to prevent leakage and overfitting. I’ve omitted the proper splitting and out-of-fold generation for brevity, but that would be essential in any real-world application.

For deeper insight into the theoretical foundations of ensemble learning, I would highly recommend reviewing "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman. Also, “Pattern Recognition and Machine Learning” by Christopher Bishop provides a detailed discussion of various ensemble methods. For practical hands-on experience, scikit-learn's documentation on ensemble methods is an excellent resource, providing implementations of many of these algorithms. Additionally, the documentation of SageMaker itself offers invaluable context on how to orchestrate these models within the AWS ecosystem.

These examples highlight how to leverage several ensemble techniques within SageMaker. Each method has advantages and disadvantages, which should be considered when selecting a solution for your specific application. As with all things in applied machine learning, experimentation is critical, and the selection of the best ensemble method and any hyperparameter tuning will usually involve careful experimentation and performance measurement on a validation set.
