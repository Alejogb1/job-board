---
title: "How to do denoising 1d timeseries using deep learning?"
date: "2024-12-14"
id: "how-to-do-denoising-1d-timeseries-using-deep-learning"
---

alright, so denoising 1d timeseries data with deep learning, huh? i've been down that rabbit hole more times than i can count. it's one of those problems that looks simple on the surface but can get really tricky, really fast. i remember this one project back in '17, working with some noisy sensor data from a ridiculously old piece of industrial machinery. the vibrations were insane, and the raw data was practically useless for any kind of analysis without some serious cleanup. back then, the deep learning landscape wasn't quite what it is today, but even then, the general idea of using neural networks for sequence processing was already solid.

the first thing to understand is that we’re not just throwing a generic neural network at this. we're dealing with temporal data, and that means we need to use architectures that can handle sequences. recurrent neural networks (rnns), particularly lSTMs (long short-term memory) or grus (gated recurrent units), are often the go-to option. they can learn dependencies over time, which is crucial for separating the actual signal from random noise. nowadays, transformers are also a viable alternative, but i've found that for many 1d timeseries problems the overhead they introduce sometimes outweighs the benefits over lSTMs, especially when compute resources are limited.

here's the basic idea: you feed your noisy 1d time series data into the network. the network then learns to predict the clean, noise-free signal. the difference between the network’s prediction and your noisy input is what drives the learning process. this is usually done via some kind of loss function that penalizes the network for poor predictions. the mean squared error (mse) is the most common starting point because it makes calculations simpler. however there are alternatives that work better depending on your case such as mean absolute error (mae) or others if your data has high impact outliers.

now, let's look at some code snippets. i'm going to assume you are using python with tensorflow or pytorch since those are the most widely used tools. i'll keep it as simple as possible:

```python
# example 1: simple lstm based denoising model in tensorflow
import tensorflow as tf
from tensorflow.keras.layers import lstm, dense, input
from tensorflow.keras.models import model

def build_lstm_denoiser(input_shape):
    inputs = input(shape=input_shape)
    x = lstm(64, activation='tanh', return_sequences=true)(inputs)
    x = lstm(64, activation='tanh', return_sequences=false)(x)
    outputs = dense(input_shape[1])(x)
    model = model(inputs=inputs, outputs=outputs)
    return model

# create a dummy input shape example
input_shape = (None, 1) # assumes 1 dimensional time series
model = build_lstm_denoiser(input_shape)
model.compile(optimizer='adam', loss='mse')
model.summary()
```

this first snippet is a tensorflow example. it defines a simple model with two lstm layers followed by a dense layer. pay close attention to `return_sequences=true` in the first lstm layer and `return_sequences=false` in the second one. if you want to build a deep network with more layers, feel free to add more lstm layers, but remember that each one introduces more computational cost and it’s not always the best move. i've seen people go crazy with 5, 6 or more layers and see no increase in quality. sometimes less is more. and for the output layer i chose dense but depending on the nature of the signal that you are working on you might need other type of activation functions. i left it simple as an example.

here's a corresponding example in pytorch:

```python
# example 2: simple lstm based denoising model in pytorch
import torch
import torch.nn as nn
import torch.optim as optim

class lstmdenoiser(nn.module):
    def __init__(self, input_size, hidden_size):
        super(lstmdenoiser, self).__init__()
        self.lstm1 = nn.lstm(input_size, hidden_size, batch_first=true)
        self.lstm2 = nn.lstm(hidden_size, hidden_size, batch_first=true)
        self.fc = nn.linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out

#create a dummy input shape example
input_size = 1
hidden_size = 64
model = lstmdenoiser(input_size, hidden_size)
optimizer = optim.adam(model.parameters())
criterion = nn.mseLoss()
print(model)
```

this pytorch code does exactly the same as the tensorflow one, building an lstm model with two lstm layers and a linear output layer. remember to chose the same input data dimensions that you are going to use in your dataset, in this case a timeseries with one dimension. the main difference between the two snippets lies in how tensorflow vs pytorch handles defining and building models. i chose `adam` as the optimizer in both cases because it is a good general purpose option.

a critical part that is usually overlooked is the data preprocessing step. you can't just feed raw data into these models and expect miracles. i've learned this the hard way. it's essential to scale your data (e.g., using min-max scaling or standardization) so that all features have a similar range. this helps the optimization process and the training speed. also, you might want to window or segment your data into smaller sequences. this will allow the lstm layers to learn short time dependencies and also can solve problems related to memory consumption for long time series.

here's how you could handle the data preprocessing before the training in pytorch:

```python
# example 3: basic data preprocessing
import torch
from torch.utils.data import dataset, dataloader

class timeseriesdataset(dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]
        return torch.tensor(window[:-1], dtype=torch.float), torch.tensor(window[-1], dtype=torch.float)

def scale_data(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# dummy data generation example
data = [i + (torch.rand(1).item() * 2) for i in range(1000)] # create some dummy linear data with some noise

# preprocessing
scaled_data = scale_data(data)
window_size = 20
dataset = timeseriesdataset(scaled_data, window_size)
dataloader = dataloader(dataset, batch_size=32, shuffle=true)

# iteration through the dataloader during training
for batch_x, batch_y in dataloader:
    print(f"batch x shape: {batch_x.shape}, batch y shape {batch_y.shape}")
    break
```

this final snippet illustrates a very important step. first i wrote two helper functions: `scale_data` which scales between 0 and 1 and `timeseriesdataset` which segments the scaled time series into smaller overlapping windows for training. during training you will be using the dataloader which returns batches of training data. the `__getitem__` function in the dataset class is responsible for converting the original data into these batches. during training, the first element of the batch will be given to the model as input and the second element of the batch will be the target that you want to predict. this is only one way to handle the data, you could use other data segmentation techniques that could also work. for simplicity i just showed a simple windowing strategy.

now, the key with these models lies in the hyperparameter tuning. you might need to change the number of layers, the number of units in each layer, the optimizer, the learning rate, the window sizes, the training epochs, and so on. don’t expect to get stellar results with default hyperparameters, that almost never happens. finding the optimal configuration for your specific data is a process of trial and error. there are tools and techniques for this such as bayesian optimization and grid search.

finally, there’s the evaluation part. don’t fall into the trap of evaluating your model on the same data you used for training. that would give you a completely wrong idea of how well it performs in real-world scenarios. split your data into training, validation, and testing sets. train your model on the training set, fine-tune the hyperparameters using the validation set, and finally, test your final model on the testing set to get an accurate idea of its generalization ability. this applies to every machine learning model that exists.

as for recommended resources, i would suggest looking into the book *deep learning with python* by francois chollet if you are using tensorflow/keras. it’s a great place to start understanding the core concepts, it covers a wide variety of subjects and it's very practical and clear in its explanations. if you are more into pytorch, then the *pytorch documentation* is your friend. there’s also a lot of very good courses and materials available from authors such as andrew ng in coursera, where he delves deep into the inner workings of deep learning, specifically sequence models.

it's a bit of a journey but i hope that helps and remember, this stuff takes time and experimentation. don't get discouraged if you don't see results right away. i remember one day i was trying to fix a bug and i was just staring at the code for 30 minutes without finding it. turns out, the problem was that the code was too long, i needed to scroll down a bit more, what a classic! good luck with your denoising endeavors!
