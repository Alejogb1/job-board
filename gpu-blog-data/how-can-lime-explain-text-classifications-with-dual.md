---
title: "How can LIME explain text classifications with dual inputs?"
date: "2025-01-30"
id: "how-can-lime-explain-text-classifications-with-dual"
---
The core challenge in explaining text classifications with dual inputs using LIME (Local Interpretable Model-agnostic Explanations) lies in effectively representing the combined influence of both input modalities within the local vicinity of a specific prediction.  My experience working on sentiment analysis for multilingual e-commerce reviews highlighted this precisely.  Simply concatenating the inputs isn't sufficient; it obscures the individual contribution of each modality (e.g., text and associated metadata like product category).  A nuanced approach requires careful feature engineering and perturbation strategies tailored to the dual input structure.

**1. Explanation of LIME for Dual Inputs:**

LIME functions by approximating the model's behavior locally around a prediction.  For single input scenarios, this typically involves perturbing the input features (e.g., replacing words in a sentence) and observing the model's response.  With dual inputs, say, text (T) and metadata (M), the perturbation strategy needs to address both independently and in combination.  We cannot simply treat the concatenated vector (T, M) as a single feature space. This is because the model's internal representation likely processes T and M through distinct pathways before integrating them.  Therefore, perturbations should be applied separately to T and M, generating a weighted neighborhood around the instance of interest.

To illustrate, consider a review classified as negative.  Perturbations might involve: (a) replacing individual words in the text (T), (b) altering the metadata (M) – for example, changing the product category – and (c) combinations of both (a) and (b).  The weighted neighborhood is then constructed based on the distance metric chosen, which needs to account for the heterogeneity of the two input types.  A simple approach might use a weighted Euclidean distance, where the weights reflect the relative importance assigned to T and M based on domain knowledge or preliminary analysis.

The weighted neighborhood is then used to train a simpler, interpretable model (e.g., linear regression) to approximate the complex model's behavior locally.  The coefficients of this simpler model then provide feature importances, explaining how individual words in T and specific metadata fields in M contribute to the final classification.  Crucially, this process allows for isolating the impact of each input modality and understanding their interplay in shaping the prediction.


**2. Code Examples:**

The following examples demonstrate different approaches to implementing LIME with dual inputs.  These are simplified illustrations and would require adaptation based on the specific model and data.

**Example 1:  Concatenated Inputs with Weighted LIME:**

```python
import lime
import lime.lime_tabular
import numpy as np

# Assume 'X_text' is preprocessed text data, 'X_meta' is metadata (numerical)
X = np.concatenate((X_text, X_meta), axis=1) # Concatenate for LIME

# Weights for the distance metric (higher weight implies higher importance)
weights = np.array([0.7, 0.3])  # Example: 70% text, 30% metadata

explainer = lime.lime_tabular.LimeTabularExplainer(
    X,
    feature_names=text_feature_names + meta_feature_names,
    class_names=['Positive', 'Negative'],
    discretize_continuous=True,
    feature_selection='auto',
    kernel_width=3,
    verbose=False
)

instance_index = 0 # Index of the instance to explain
explanation = explainer.explain_instance(
    X[instance_index],
    model.predict_proba,
    num_features=10, #Top 10 features
    top_labels=1,
    distance_metric='weighted_euclidean',
    weights=weights
)

explanation.show_in_notebook()
```

This example directly concatenates text and metadata features. The crucial aspect is the use of a weighted Euclidean distance within LIME to account for the disparate importance of each input type.  The weights are hyperparameters to be tuned.

**Example 2:  Separate LIME Explanations and Aggregation:**

```python
import lime
import lime.lime_text
import lime.lime_tabular

# Text LIME
text_explainer = lime.lime_text.LimeTextExplainer(class_names=['Positive', 'Negative'])
text_explanation = text_explainer.explain_instance(
    text_data[instance_index],
    lambda x: model.predict_proba(np.concatenate([x, metadata[instance_index]])),
    num_features=5
)
text_explanation.show_in_notebook()

# Metadata LIME
meta_explainer = lime.lime_tabular.LimeTabularExplainer(metadata,feature_names=meta_feature_names)
meta_explanation = meta_explainer.explain_instance(
    metadata[instance_index],
    lambda x: model.predict_proba(np.concatenate([text_data[instance_index],x])),
    num_features=5
)
meta_explanation.show_in_notebook()

#Manual Aggregation (e.g., averaging weights)
# ...Further processing to combine the explanations.
```

Here, we apply LIME separately to text and metadata, using a lambda function to include the other input modality in the prediction function.  This allows for independent analysis, but requires a post-processing step to combine and integrate these separate explanations which often proves challenging and inherently subjective.

**Example 3:  Custom Kernel for Dual Inputs:**

```python
import lime
import lime.lime_tabular
from scipy.spatial.distance import cdist

class DualInputExplainer(lime.lime_tabular.LimeTabularExplainer):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def _kernel(self, x1, x2):
        # Custom kernel considering text and metadata separately
        text_distance = cdist(x1[:, :text_feature_dim], x2[:, :text_feature_dim], 'cosine') #Example:cosine for text
        meta_distance = cdist(x1[:, text_feature_dim:], x2[:, text_feature_dim:], 'euclidean') #Example:euclidean for meta
        combined_distance = 0.7 * text_distance + 0.3 * meta_distance #Weighted combination
        return np.exp(-combined_distance / self.kernel_width)

# ... Usage remains largely similar to LimeTabularExplainer
```

This example showcases a customized LIME explainer. It defines a kernel function that explicitly handles the dual inputs, weighing text and metadata distances differently to reflect their importance. This kernel is then used within the LIME framework to construct a weighted neighborhood.


**3. Resource Recommendations:**

For deeper understanding, consult the original LIME paper and related publications on model explainability.  Explore documentation and examples related to the `lime` Python package.  Examine advanced topics in feature engineering and dimensionality reduction for high-dimensional text data.  Consider research papers on explainable AI (XAI) in the context of multimodal learning.  Investigate techniques for handling categorical and numerical data in interpretability methods.


The choice of approach depends heavily on the characteristics of the model and data.  A thorough understanding of the underlying model architecture and the nature of the dual inputs is crucial for selecting the most effective LIME strategy.  Remember that LIME provides local explanations, and the global picture may require aggregating many local explanations, potentially using techniques that account for uncertainty and potential biases.
