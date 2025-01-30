---
title: "How can LSTM networks in PyTorch be used for multi-step time series forecasting?"
date: "2025-01-30"
id: "how-can-lstm-networks-in-pytorch-be-used"
---
Long Short-Term Memory (LSTM) networks, a specialized type of recurrent neural network (RNN), have proven effective in capturing temporal dependencies within sequential data, making them suitable for time series forecasting. My own experience implementing these models in a financial forecasting context has highlighted both their power and the nuances involved in multi-step prediction. This response details how to leverage LSTMs in PyTorch for such tasks.

The fundamental challenge in multi-step forecasting is predicting future values beyond a single time point. Instead of predicting only the next value in the sequence, the goal is to generate a sequence of future values, often several time steps into the future. Directly training an LSTM to generate this entire sequence can be problematic due to error accumulation and the inherent difficulty in propagating gradients over long sequences. Several approaches mitigate these issues, each with their tradeoffs. The most commonly used are iterative or recursive prediction, direct prediction, and encoder-decoder architectures. I will focus on demonstrating and explaining the recursive approach using a single LSTM layer for clarity.

The recursive (or iterative) method uses the LSTM to predict one time step into the future. Then, this predicted value is fed back into the LSTM model as if it were a ground truth observation, and the process repeats to forecast the subsequent time steps. This method is straightforward to implement but can lead to accumulating error with each prediction, as errors in early forecasts influence later ones. This compounding effect can make prediction accuracy degrade as the forecasting horizon increases.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len, input_size)
        # hidden: (num_layers, batch_size, hidden_size) if provided, otherwise defaults to zeros.
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        output = self.fc(lstm_out[:, -1, :]) # Take the last output for prediction.
        # output: (batch_size, output_size)
        return output, hidden

def train_lstm(model, train_loader, criterion, optimizer, num_epochs=100):
   for epoch in range(num_epochs):
        for i, (sequences, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output, _ = model(sequences)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
           print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def multi_step_forecast(model, initial_sequence, forecast_steps, hidden=None):
    model.eval() # Set model to evaluation mode.
    forecasts = []
    current_sequence = initial_sequence
    with torch.no_grad(): # Disable gradient tracking for evaluation.
        for _ in range(forecast_steps):
            output, hidden = model(current_sequence, hidden)
            forecasts.append(output)
            current_sequence = output.unsqueeze(1) # use predicted value as input for next step.
    return torch.cat(forecasts, dim=1)
```

This first code segment presents the core LSTM model class and training regime in PyTorch. The `LSTMForecaster` class encapsulates the LSTM layer and a final linear layer (`fc`) to map the LSTM’s hidden state to the desired output. The training function iterates through the data loader, calculates the loss, and updates model parameters. The key aspect for multi-step prediction lies within the `multi_step_forecast` function. The model is used iteratively to generate predictions: a new prediction becomes the input for the next prediction. I've included commentary within the code to clarify specific tensor dimensions, input, and output shapes.

```python
# Example data generation and usage
input_size = 1  # Single feature time series.
hidden_size = 64
output_size = 1
num_layers = 1
seq_len = 20
forecast_steps = 10
batch_size=32
num_epochs = 100
learning_rate = 0.001

# Generate a synthetic sine wave
def generate_sine_wave(seq_len, num_batches):
    time = np.linspace(0, 10*np.pi, seq_len+forecast_steps)
    sine = np.sin(time)
    sequences = []
    labels = []
    for i in range(num_batches):
         start_idx = np.random.randint(0, len(sine) - (seq_len + forecast_steps))
         seq = sine[start_idx : start_idx+seq_len]
         label = sine[start_idx + seq_len : start_idx + seq_len + 1]
         sequences.append(seq)
         labels.append(label)
    sequences = torch.tensor(np.array(sequences), dtype=torch.float).unsqueeze(-1)
    labels = torch.tensor(np.array(labels), dtype=torch.float).squeeze(-1)
    return sequences, labels

sequences, labels = generate_sine_wave(seq_len, batch_size * 10) # Create training data
train_loader = torch.utils.data.TensorDataset(sequences, labels)
train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)

model = LSTMForecaster(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_lstm(model, train_loader, criterion, optimizer, num_epochs)

# Test forecast data generation
test_sequences, _  = generate_sine_wave(seq_len, 1) # Create one example for forecast
with torch.no_grad():
   forecasted_values = multi_step_forecast(model, test_sequences, forecast_steps)

print(f"Shape of Forecasted Values {forecasted_values.shape}")
print(f"Forecasted values: {forecasted_values.squeeze().numpy()}")
```

The second segment provides a concrete example of how to utilize the previously defined model. It generates synthetic sine wave data, creating training samples of specified length. Training data is passed through the train function. Finally, it shows how to use the `multi_step_forecast` function to generate predictions, illustrating the use of trained model. The example includes printing the shape and example values of the forecast output. The single feature and synthetic sine function represents a simple time-series input for testing and model demonstration purposes, mirroring a more simple financial time series I worked with during my initial experiments.

```python
class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, decoder_input, hidden=None):
        # encoder_input: (batch_size, encoder_seq_len, input_size)
        # decoder_input: (batch_size, decoder_seq_len, output_size)

        encoder_out, hidden = self.encoder(encoder_input, hidden)

        # Initial decoder hidden state set as final encoder hidden state.
        decoder_out, _ = self.decoder(decoder_input, hidden)

        output = self.fc(decoder_out)
        return output

def train_seq2seq(model, train_loader, criterion, optimizer, num_epochs=100):
   for epoch in range(num_epochs):
        for i, (encoder_sequences, decoder_sequences, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(encoder_sequences, decoder_sequences)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
           print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example data generation and usage
def generate_seq2seq_data(seq_len, forecast_steps, num_batches):
    time = np.linspace(0, 10*np.pi, seq_len+forecast_steps)
    sine = np.sin(time)
    encoder_seqs = []
    decoder_seqs = []
    labels = []
    for i in range(num_batches):
         start_idx = np.random.randint(0, len(sine) - (seq_len + forecast_steps))
         encoder_seq = sine[start_idx:start_idx+seq_len]
         decoder_seq = sine[start_idx + seq_len -1 : start_idx + seq_len + forecast_steps - 1] # Start last value of encoder
         label = sine[start_idx+seq_len : start_idx + seq_len + forecast_steps]
         encoder_seqs.append(encoder_seq)
         decoder_seqs.append(decoder_seq)
         labels.append(label)
    encoder_seqs = torch.tensor(np.array(encoder_seqs), dtype=torch.float).unsqueeze(-1)
    decoder_seqs = torch.tensor(np.array(decoder_seqs), dtype=torch.float).unsqueeze(-1)
    labels = torch.tensor(np.array(labels), dtype=torch.float).unsqueeze(-1)
    return encoder_seqs, decoder_seqs, labels

input_size = 1  # Single feature time series.
hidden_size = 64
output_size = 1
num_layers = 1
seq_len = 20
forecast_steps = 10
batch_size=32
num_epochs = 100
learning_rate = 0.001

encoder_sequences, decoder_sequences, labels = generate_seq2seq_data(seq_len, forecast_steps, batch_size * 10) # Create training data
train_dataset = torch.utils.data.TensorDataset(encoder_sequences, decoder_sequences, labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = LSTMSeq2Seq(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_seq2seq(model, train_loader, criterion, optimizer, num_epochs)

# Generate a test prediction using seq2seq.
encoder_test_sequences, decoder_test_sequences, _ = generate_seq2seq_data(seq_len, forecast_steps, 1) # Generate one test example.
with torch.no_grad():
    forecast = model(encoder_test_sequences, decoder_test_sequences)

print(f"Forecast seq2seq shape {forecast.shape}")
print(f"Forecast seq2seq values: {forecast.squeeze().numpy()}")
```

This third code segment demonstrates a sequence-to-sequence approach using LSTMs. Instead of iterative generation, the `LSTMSeq2Seq` model has separate encoder and decoder LSTMs. The encoder takes the input sequence. The decoder takes a target sequence shifted by one step, which during the training process is the true value. The decoder is initialized by passing the last hidden output from the encoder. This method is known as "teacher forcing" during training. After training, the decoder input can be generated from the previous output, allowing to generate forecasts iteratively in an autoregressive fashion. The synthetic data generation and usage of the model is similar to the first example. I've included the training loop and output with shapes to highlight this alternative implementation. The key difference in this architecture is the dual use of LSTMs. In practice I've seen this be more robust to long sequence prediction than simple iterative methods, at a cost of complexity.

When choosing methods for multi-step forecasting, it is imperative to consider the characteristics of your data and the required length of the forecast horizon. Iterative approaches such as the one in the first example are simple to implement, but they are vulnerable to error propagation. Sequence-to-sequence methods can often produce superior performance when modeling long-range forecasts, but at the cost of additional complexity in model definition and training. Direct multi-step forecasting, in which we directly regress the whole forecast sequence in one go, is an option but I have found this to require substantially more data to reliably train.

For further understanding of advanced time series forecasting, I recommend the following resources: the book “Forecasting: Principles and Practice” by Hyndman and Athanasopoulos, “Time Series Analysis” by James D. Hamilton and the documentation of the PyTorch libraries focusing on sequential modeling. Further research of attention-based mechanisms may also improve the predictive capabilities of the model.
