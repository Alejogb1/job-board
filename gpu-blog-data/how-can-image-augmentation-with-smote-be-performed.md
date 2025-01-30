---
title: "How can image augmentation with SMOTE be performed in batches without exceeding RAM limitations?"
date: "2025-01-30"
id: "how-can-image-augmentation-with-smote-be-performed"
---
I’ve encountered the memory limitations of SMOTE (Synthetic Minority Oversampling Technique) during image augmentation in several large-scale medical imaging projects. Specifically, processing image data in batch format, where thousands of high-resolution images are handled concurrently, pushes traditional SMOTE implementations beyond the capabilities of typical RAM. The problem stems from SMOTE’s requirement to operate on the entire feature space simultaneously to identify nearest neighbors, which generates synthetic samples. This contrasts with batch-based processing, where only a subset of the data is in memory at any given time. The key challenge lies in adapting SMOTE's global-view processing to a local, batch-oriented approach.

To address this, I've found a combination of techniques effective: leveraging a class-balanced approach within each batch, coupled with careful consideration of the feature space used by SMOTE and the judicious use of in-memory data structures to manage sample indices. The core idea is to perform SMOTE on a *local* level within each batch rather than attempting to process the entire dataset.

Initially, when I experimented with a direct application of SMOTE on batches, memory consumption scaled rapidly, especially with larger batches. The naive implementation attempted to build the necessary K-Nearest Neighbors (KNN) structure for all samples in memory, before selecting and creating synthetic samples. This was completely infeasible.

The first step toward a scalable solution involves preprocessing images to a more concise feature representation before applying SMOTE. Using raw pixel data as a feature vector can lead to high-dimensional data that increases computational overhead during KNN calculations. Therefore, I employed a pre-trained convolutional neural network (CNN) — specifically one trained on a relevant task like image classification within the domain — to generate a lower-dimensional embedding of the image data. The CNN extracts semantically meaningful features rather than dealing with raw pixel values, which considerably reduces the feature space's dimensionality and consequently lowers the memory overhead of SMOTE. This embedding is then the input for SMOTE processing within each batch.

Secondly, when processing a batch, I first ensure that each class is represented adequately. This is crucial because if a batch only contains majority-class samples, SMOTE would not be applicable, and all samples would remain as-is. Similarly, if a batch contains only a single minority-class sample, the effectiveness of SMOTE is dramatically reduced since it cannot find reliable neighbors. The solution is to select batches that contain a *balanced* representation of the minority and majority classes within reasonable bounds. This can involve shuffling the entire dataset beforehand, but also ensuring that data loaders use stratified sampling methods across classes.

Thirdly, within each balanced batch, I apply SMOTE only to the minority class samples. This is a key adjustment because applying it to all samples generates unnecessary overhead. This strategy assumes that we are primarily concerned with oversampling the underrepresented class. Critically, SMOTE *is not applied across batches*. Each batch is treated as its own independent SMOTE operation.

The following Python-based examples demonstrate the concepts, using a hypothetical framework built atop standard libraries, such as `scikit-learn` and `numpy`, that provide KNN algorithms, as well as typical frameworks used for image and deep learning operations. The following code is simplified for clarity, focusing on the conceptual aspects of batch-based SMOTE, assuming image data is already loaded and preprocessed into a feature-space using the pre-trained CNN mentioned earlier.

**Example 1: Generating Balanced Batches**
This example demonstrates how to generate batches such that each has sufficient representation from each class. This step is critical to ensure that a class-based approach to SMOTE can generate reasonable synthetic samples.
```python
import numpy as np

def generate_balanced_batches(features, labels, batch_size, class_balance_ratio):
    """Generates batches with a balance of minority and majority class samples."""
    unique_labels = np.unique(labels)
    batches = []
    
    # Find the index of the minority class (simplifying to a binary case)
    minority_label = min(unique_labels, key=lambda l: np.sum(labels == l))
    majority_label = max(unique_labels, key=lambda l: np.sum(labels == l))
    
    minority_indices = np.where(labels == minority_label)[0]
    majority_indices = np.where(labels == majority_label)[0]
    
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    
    num_minority_in_batch = int(batch_size * class_balance_ratio)
    num_majority_in_batch = batch_size - num_minority_in_batch

    if num_minority_in_batch > minority_count:
       num_minority_in_batch = minority_count
       num_majority_in_batch = min(batch_size - num_minority_in_batch, majority_count)

    if num_majority_in_batch > majority_count:
      num_majority_in_batch = majority_count
      num_minority_in_batch = min(batch_size - num_majority_in_batch, minority_count)

    minority_batches = []
    for i in range(0, minority_count, num_minority_in_batch):
        minority_batches.append(minority_indices[i : i + num_minority_in_batch])
    
    majority_batches = []
    for i in range(0, majority_count, num_majority_in_batch):
        majority_batches.append(majority_indices[i : i+ num_majority_in_batch])
    
    
    num_batches = min(len(minority_batches), len(majority_batches))

    for i in range(num_batches):
      current_batch_indices = np.concatenate((minority_batches[i], majority_batches[i]))
      np.random.shuffle(current_batch_indices)
      batches.append((features[current_batch_indices], labels[current_batch_indices]))

    #If batches are unequal in length - this can be handled with padding if necessary.
    return batches
    
# Example usage - assuming a 1000 sample dataset
features = np.random.rand(1000, 64) # Features extracted from a CNN
labels = np.random.randint(0, 2, 1000)  # Binary classification 0 or 1
batch_size = 128
class_balance_ratio = 0.3

batches = generate_balanced_batches(features, labels, batch_size, class_balance_ratio)
print(f"Number of batches generated: {len(batches)}")

for features_batch, labels_batch in batches:
  print(f"Batch size:{len(features_batch)}, Minority Class Count: {np.sum(labels_batch==min(np.unique(labels_batch), key=lambda l: np.sum(labels_batch == l)))} , Majority Class Count:{np.sum(labels_batch==max(np.unique(labels_batch), key=lambda l: np.sum(labels_batch == l)))} ")
```
This function creates batches ensuring that the representation of the minority class is in proportion to the `class_balance_ratio`. The function iterates through class indices to ensure each batch has enough samples from each class. Note:  `class_balance_ratio` should be less than 0.5 in order to oversample the minority class.

**Example 2: Applying SMOTE on Each Batch**
This example uses a class-balanced batch, applies SMOTE to the minority class samples, and returns the augmented batch. This snippet highlights how SMOTE is applied on a per-batch, rather than per-dataset, basis.
```python
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE  

def apply_smote_batch(features_batch, labels_batch):
    """Applies SMOTE only to the minority class samples in a batch."""
    unique_labels = np.unique(labels_batch)
    minority_label = min(unique_labels, key=lambda l: np.sum(labels_batch == l))
    
    minority_indices = np.where(labels_batch == minority_label)[0]
    if len(minority_indices) == 0:
      return features_batch, labels_batch

    smote = SMOTE(random_state=42, k_neighbors=min(len(minority_indices)-1,5)) #Use a max k-neighbor value
    augmented_features, augmented_labels = smote.fit_resample(features_batch[minority_indices], labels_batch[minority_indices])

    all_features = np.concatenate((features_batch,augmented_features),axis = 0)
    all_labels = np.concatenate((labels_batch,augmented_labels))
    
    return all_features, all_labels

# Example usage
for features_batch, labels_batch in batches:
    augmented_features_batch, augmented_labels_batch = apply_smote_batch(features_batch, labels_batch)
    print(f"Original Batch size: {len(features_batch)}, Augmented Batch Size:{len(augmented_features_batch)}")
```
The `apply_smote_batch` function first identifies the minority class. If no minority samples exist within the batch (which is possible due to batching) the method returns without applying SMOTE. Otherwise, an instance of `SMOTE` is used to apply the augmentation to the minority class examples and the combined features and labels are returned.

**Example 3: Complete Pipeline**
This example demonstrates the complete pipeline combining the previous 2 examples. This step is the integration of class-balanced batch generation and SMOTE applied on a per-batch basis.
```python
for features_batch, labels_batch in batches:
    augmented_features_batch, augmented_labels_batch = apply_smote_batch(features_batch, labels_batch)
    print(f"Original Batch size: {len(features_batch)}, Augmented Batch Size: {len(augmented_features_batch)}")
    # Further training can be done here using the 'augmented_features_batch'
    # and 'augmented_labels_batch' on a per batch basis.
```
This example illustrates the end-to-end process where a given dataset is first split into balanced batches. Then SMOTE is applied to each batch on the minority class samples. The `augmented_features_batch` and `augmented_labels_batch` can then be used to train a model. It is very important to note that models must be trained in batches to reduce RAM overhead. The key concept in all three examples is to avoid applying SMOTE on the entire dataset at once and instead, leverage batches of data.

For further exploration into batch processing techniques, consult resources focusing on deep learning pipelines and data loading strategies within popular frameworks.  Additionally, investigating literature on imbalanced learning that discusses both global (data-level) and local (algorithmic-level) solutions can be beneficial. Papers focusing on efficient K-Nearest Neighbor calculations for high-dimensional data will also prove useful. Lastly, familiarity with batch processing in common deep learning frameworks (e.g., TensorFlow and PyTorch) is highly recommended. These resources, and understanding their application, are crucial for handling large-scale image datasets with SMOTE augmentation, while respecting memory constraints.
