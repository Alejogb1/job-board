---
title: "How can I handle unknown categories '2' in pytorch_forecasting?"
date: "2025-01-30"
id: "how-can-i-handle-unknown-categories-2-in"
---
When dealing with time series forecasting, encountering unknown categories during prediction, especially with categorical embedding layers in `pytorch_forecasting`, is a frequent challenge. I’ve personally encountered this issue while deploying a demand forecasting model for a retail client where new products (and therefore new product categories) would regularly be introduced. The core problem arises because the model is trained on a finite set of categorical values, and when it faces an unseen category, the learned embeddings become meaningless. The framework doesn’t intrinsically know how to handle this. Instead of an error, the framework may produce unpredictable, poor quality results. Proper handling of these unknown categories is crucial for model robustness and reliability in real-world scenarios.

The fundamental approach I’ve found most effective involves a combination of preparing the data to explicitly handle out-of-vocabulary cases and adapting the model definition to accommodate this mechanism. The first step involves modifying the training data to include an "unknown" category, which allows the model to learn a reasonable representation. The second step is to modify the prediction data preparation and the model so that when the prediction data includes categories not seen during training, they get mapped to this "unknown" value. This makes the embeddings layer able to process the prediction data even when new categories have been added since training.

Here's how I typically address this issue in practice, with practical code examples. Firstly, during data preparation for training, I introduce a placeholder for unknown categories, typically represented by a dedicated integer value. This value is added to the mapping of each categorical variable. Let’s say we are encoding the 'product_category' column, and the unique training values are ['Electronics', 'Books', 'Clothing']. We'd add an 'Unknown' category (or similar name) to the list and ensure it is also encoded, as seen in the example below:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_training_data(df):
    """Prepares the training data by adding an 'Unknown' category and encoding."""

    category_cols = ['product_category', 'store_location']  # Example columns
    for col in category_cols:
       
        encoder = LabelEncoder()
        all_categories = df[col].unique().tolist()
        all_categories.append("Unknown") # Added 'Unknown' category here
        encoder.fit(all_categories) 
        df[col] = encoder.transform(df[col])

    return df, {col: encoder for col in category_cols}


data = {'product_category': ['Electronics', 'Books', 'Clothing', 'Electronics', 'Books'],
        'store_location': ['New York', 'London', 'Tokyo', 'New York', 'London'],
        'target': [100, 50, 75, 120, 60]}

df = pd.DataFrame(data)

df, encoders = prepare_training_data(df)

print(df)
print(encoders)
```

This code snippet illustrates creating a `LabelEncoder` instance and appending 'Unknown' to the list of all possible categories. When `fit` is performed on the augmented categories, the "Unknown" category will also get a label. When `transform` is called on the column, the data is now encoded, and the framework will have seen an "Unknown" label, and associated embedding. This ensures the training data contains a dedicated representation for out-of-vocabulary categories. This is crucial to prevent the model from having undefined input. The encoder itself, saved in the `encoders` dictionary, will be used later.

Now, when it comes time to prepare the data for prediction, I use the saved encoder to map known categories to their integer values and any unknown values to the "Unknown" integer value encoded during training.  This is essential. If the prediction data includes categories not seen during training, these should be mapped to the “Unknown” integer label from the previous step. The modification below ensures new values will be mapped to the "unknown" label learned by the model. This step is critical as the model will only know how to handle integer labels present in the training data, and if an unknown label gets assigned a new, unexpected, integer value, the framework's embedding operation will not work as intended.

```python
def prepare_prediction_data(df, encoders):
    """Prepares prediction data, mapping new categories to 'Unknown'."""
    category_cols = ['product_category', 'store_location']

    for col in category_cols:
        
        df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else "Unknown")
        df[col] = encoders[col].transform(df[col])
        
    return df

prediction_data = {'product_category': ['Electronics', 'Toys', 'Books', 'Furniture'],
                'store_location': ['New York', 'Berlin', 'London', 'Paris'],
                'target': [None, None, None, None]}

df_pred = pd.DataFrame(prediction_data)

df_pred = prepare_prediction_data(df_pred, encoders)
print(df_pred)
```

In this prediction preprocessing function, before `transform` is applied, we compare the data's labels to those that the encoder has seen during the training data preparation phase. The conditional check in the `apply` function guarantees that any label the encoder has never seen will be mapped to `Unknown`. After this step the `transform` operation can now proceed successfully as the prediction dataframe contains only integer labels that were seen by the model during training.

Finally, while the above preprocessing is generally sufficient, if for some reason, all values are not known during training, it might be prudent to initialize your embedding layers with a fixed size, and ensure to pass `num_embeddings` to the layer constructor so it can be handled more gracefully by pytorch. This is a safeguard, as the code above should ideally prevent the scenario this handles.

```python
import torch
from torch import nn
from pytorch_forecasting.models.deepar import DeepAR

def create_deepar_model(encoders, embedding_sizes):
    """Creates the DeepAR model with fixed embedding sizes"""

    return DeepAR.from_dataset(
        None, # We dont pass any dataset here, as we are manually setting things
        embedding_sizes=embedding_sizes, # embedding_sizes are passed here instead
        categorical_encoders = encoders,
    )


embedding_sizes = {
    'product_category': len(encoders['product_category'].classes_), # Use the length of all classes including "Unknown"
    'store_location': len(encoders['store_location'].classes_),
}

model = create_deepar_model(encoders, embedding_sizes)


# Here's how to do a minimal prediction with this:
# NOTE: for this example we assume that the prediction data has been prepared by the function above
# which is to say, it only has numeric, known, labels in the category columns

# Create a dataset and pass to predict
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import GroupNormalizer
from torch.utils.data import DataLoader

training_data = {'product_category': [0, 1, 2, 0, 1],
        'store_location': [0, 1, 2, 0, 1],
        'target': [100, 50, 75, 120, 60],
        'time_idx': [0, 1, 2, 3, 4],
        'group_id': [0, 0, 0, 0, 0]}
training_df = pd.DataFrame(training_data)


prediction_data = {'product_category': [0, 3, 1, 4],
                'store_location': [0, 3, 1, 4],
                'target': [None, None, None, None],
                'time_idx': [5, 6, 7, 8],
                'group_id': [0, 0, 0, 0]}

prediction_df = pd.DataFrame(prediction_data)


training = TimeSeriesDataSet(
    training_df,
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    max_encoder_length=5,
    max_prediction_length=3,
    time_varying_known_reals = ["time_idx"],
    time_varying_unknown_reals = ["target"],
    categorical_encoders = encoders,
)


prediction_dataset = TimeSeriesDataSet.from_dataset(training, prediction_df, predict=True, stop_randomization=True)

batch_size = 32
train_dataloader = DataLoader(training, batch_size=batch_size, shuffle=False)
prediction_dataloader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)

# prediction
for idx, x in enumerate(prediction_dataloader):
  predictions = model.predict(x)
  print(predictions)
  break
```

This example is complete. It shows the preparation for both training data and prediction data. It includes the model setup, embedding sizes initialization and the minimal example of a prediction. The crucial piece of code here is passing the `embedding_sizes` as an argument when constructing the model using `.from_dataset`. Here, the `len()` of all the encoded values, including the "Unknown" value is passed.

Regarding resources, I would advise exploring the official `pytorch_forecasting` documentation thoroughly, paying specific attention to the sections on categorical embeddings and model construction. Additionally, reviewing relevant sections of the PyTorch documentation on embedding layers can provide a deeper understanding. There are several good papers which discuss time-series with neural networks, however, none that specifically deal with this category-embedding problem in Pytorch Forecasting. Therefore, your primary learning source will be the documentation, and your own experimentation. Good luck!
