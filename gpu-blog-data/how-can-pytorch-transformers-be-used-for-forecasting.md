---
title: "How can PyTorch Transformers be used for forecasting multiple and multivariate time series?"
date: "2025-01-30"
id: "how-can-pytorch-transformers-be-used-for-forecasting"
---
PyTorch Transformers, primarily known for their success in natural language processing, can be adapted for forecasting multiple and multivariate time series, leveraging their attention mechanisms to capture complex temporal dependencies across series. The core challenge in this context lies in representing time series data in a format compatible with transformer architectures and tailoring the output to predict future values rather than sequences of tokens. Based on my experience building a demand forecasting system for a large retail chain, I found that the key modifications center on input encoding, output decoding, and loss function adaptation.

The fundamental premise is to treat each time series as a sequence, analogous to a sentence in NLP. However, unlike words, time series data consists of numerical values. Therefore, the first step involves embedding each time point's value into a high-dimensional vector space. This embedding can be a simple linear projection or, more effectively, learnable embeddings specific to each series and the position within the time window. Furthermore, to handle multivariate time series, we concatenate the embeddings of each individual variable at each time step. Consider, for instance, a dataset with sales, promotions, and weather data. Each feature, at a given time, will be converted to an embedding; these embeddings are then combined into a single input embedding for that time step.

Next, these time-aligned embeddings are fed into the Transformer encoder. The encoder's attention mechanism is particularly powerful in time series forecasting because it can learn correlations and dependencies across different time steps, even distant ones, without the limitations of RNNs' sequential processing. This ability is crucial for capturing both short-term and long-term patterns within each time series and dependencies across multiple time series. Furthermore, if applicable, the model can also consider static features by incorporating them into a single embedding that gets added to all temporal embeddings, enhancing the overall context representation.

The output of the Transformer encoder is a sequence of encoded representations, one for each time step in the input. Since the goal is to forecast future values, these are then passed through a decoder that generates forecasts. In sequence-to-sequence translation tasks, the decoder generates a sequence of tokens, one at a time. For forecasting, we instead require numerical values. This means replacing the token prediction layer with a linear layer that projects the decoder's final output at each desired forecasting horizon to the predicted value. Essentially, the decoder predicts values in sequence for each series in the future. The length of the output sequence corresponds to the desired forecasting horizon.

Adapting the loss function is crucial. Mean Squared Error (MSE) or Mean Absolute Error (MAE) are common choices. During my work, I found that the loss function should be applied only to the predicted time-steps and not to the entire output of the decoder. Thus, we must select the relevant time steps for calculating loss. Moreover, incorporating additional loss terms, such as those penalizing the variance of predictions, can regularize the training and potentially avoid overconfident forecasts.

Here are three code examples illustrating these concepts, along with commentary:

**Example 1: Input Embedding**

```python
import torch
import torch.nn as nn

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, num_series):
        super().__init__()
        self.series_embeddings = nn.Embedding(num_series, embed_dim)
        self.time_projection = nn.Linear(input_dim, embed_dim)

    def forward(self, x, series_ids):
        # x: [batch_size, seq_len, input_dim]
        # series_ids: [batch_size, num_series]
        batch_size, seq_len, _ = x.shape
        series_embeds = self.series_embeddings(series_ids).unsqueeze(1)  # [batch_size, 1, embed_dim]
        projected_x = self.time_projection(x) # [batch_size, seq_len, embed_dim]
        # Adding Series Specific Embedding to Each Time Step
        output = projected_x + series_embeds.repeat(1,seq_len,1)
        return output
```

In this code snippet, `TimeSeriesEmbedding` handles the initial embedding. It takes the input data `x`, a tensor with shape `[batch_size, seq_len, input_dim]`, where `input_dim` is the number of features in each time series and the series IDs, `series_ids`, shape `[batch_size, num_series]`. The `series_embeddings` is an embedding layer that assigns a distinct vector to each series. `time_projection` is a linear layer mapping the input to the embedding space. It is important to note that for a given time series, the embeddings of time steps are simply the linearly projected input of the time steps and the same series embedding added to them. This method allows each time series to contribute to the representation in the embedding space.

**Example 2: Transformer Encoder Setup**

```python
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, nhead, num_layers):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # src: [batch_size, seq_len, embed_dim]
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2) # [batch_size, seq_len, embed_dim]
        return output
```

`TimeSeriesTransformerEncoder` encapsulates the transformer encoder. The key here is the use of PyTorch's `TransformerEncoderLayer` and `TransformerEncoder`. Notice that the input `src` is permuted before passing through the encoder and back. The dimensions must conform to the transformer's expectation which expects the sequence length first rather than the batch size. This example illustrates the common use of the standard PyTorch transformer implementation for time series data.

**Example 3:  Output Decoding and Prediction**

```python
import torch.nn as nn

class ForecastHead(nn.Module):
    def __init__(self, embed_dim, output_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.prediction_layers = nn.ModuleList([nn.Linear(embed_dim, output_dim) for _ in range(horizon)])

    def forward(self, encoded_seq):
         # encoded_seq: [batch_size, seq_len, embed_dim]
        output = []
        for t in range(self.horizon):
             # Forecast for each time-step sequentially
             output.append(self.prediction_layers[t](encoded_seq[:, -1, :]))
        # Concatenate predictions of the horizons
        output = torch.stack(output, dim=1)
        return output
```

`ForecastHead` takes the output from the encoder, `encoded_seq`, and projects it to predicted values. The key difference from standard transformer decoders is the removal of sequence-to-sequence operations. Here, we loop through the required forecasting horizon, applying a different projection layer for each step. This allows greater flexibility in learning distinct patterns for different time points in the future. The final output is of shape `[batch_size, horizon, output_dim]`, where `output_dim` corresponds to the number of variables being forecasted.

In summary, forecasting multiple and multivariate time series with PyTorch Transformers involves encoding time series data as sequence embeddings, leveraging a transformer encoder to capture dependencies, and then utilizing a customized forecasting head to predict future values. These modifications transform a sequence-to-sequence model into a potent tool for time series prediction.

For further exploration, I recommend studying the following resources, which provide theoretical background, practical implementation insights, and advanced techniques regarding transformers and time series analysis:

*   "Attention is All You Need" by Vaswani et al. for understanding the fundamentals of transformer architecture.
*   "Deep Learning for Time Series Forecasting" by Jason Brownlee for comprehensive coverage of time series forecasting techniques, including those that can be combined with transformer architectures.
*   "Time Series Forecasting: Principles and Practice" by Hyndman and Athanasopoulos for a detailed treatment of statistical time series models, providing a solid theoretical understanding of time series concepts.
*   "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron for practical guidance on implementing deep learning models with PyTorch, especially focusing on best practices and hyperparameter tuning.
*   Various research papers from conferences like NeurIPS, ICML, and ICLR focusing on time series analysis with transformers, which can provide specific methods and novel model variants.
