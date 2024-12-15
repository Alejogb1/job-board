---
title: "How to understand variable length backpropagation sequences?"
date: "2024-12-15"
id: "how-to-understand-variable-length-backpropagation-sequences"
---

alright, let's talk variable length backpropagation, because this is a thing, and i've had more than my share of late nights staring at loss curves gone wild because of it. it's one of those concepts that sounds straightforward in theory – "just backpropagate, but like, for a sequence of different lengths" – but the devil is, as always, in the implementation details.

basically, when you're dealing with recurrent neural networks (rnns), or sequence-to-sequence models, you're often not feeding them neatly aligned input sequences of identical length. sometimes you have sentences of varying lengths, time series data with uneven recordings, or, like i experienced back in my early days, sensor data that had missing values in between, causing the sequences to vary in size.

the challenge is that backpropagation, in its most basic form, assumes a fixed computational graph structure. a simple neural network has a clearly defined input layer, hidden layers, and an output layer. the forward pass calculates values layer by layer and then the backward pass calculates gradients with those same layers in a reversed order. when we are in a situation where we have different sequence lengths, this fixed structure gets thrown out of the window, as the depth of your network "unrolled" changes per sample. this is no longer so straightforward.

so how do we get around that? the key idea is to handle each sequence independently but still train the same model parameters. instead of processing the entire dataset in one massive computation graph, we do mini-batches, but now, also need to manage sequence lengths within them. the way to deal with it is padding and masking.

first, *padding*. because everything within a batch needs to have the same size, we pad each individual sequence with a neutral value (often zero) until they all reach the length of the longest sequence in that batch. for example, say i had three sequences of lengths 5, 3, and 7. i'd pad the first sequence with 2 zeros and the second one with 4 zeros to make them all length 7.  this makes it possible to batch them into a tensor of size *batch\_size x max\_sequence\_length*.

second, *masking*. the padding is useful to have all the inputs with the same dimensions but it introduces a problem, which is that now we are adding useless "information" to the model. we should tell the model that these padded parts are not real data to prevent the gradients from being computed on that part. that is the purpose of masks, a tensor of booleans or integers (0 or 1) that have the same dimensions as the padded input sequence, where a 1 or `true` means the value is not padding and a 0 or `false` means the value was padded. the mask is used during the forward and backward passes so that computations associated with padded values do not contribute to the loss or gradients.

now, let's show how this works with some code snippets, i will show them using pytorch since it is the library that i usually use but they can be done with similar functions on other frameworks. we will be using basic rnn cells to illustrate how masks are used in practice.

```python
import torch
import torch.nn as nn

class maskedrnn(nn.module):
    def __init__(self, input_size, hidden_size):
        super(maskedrnn, self).__init__()
        self.rnn = nn.rnn(input_size, hidden_size, batch_first = true)
        self.fc = nn.linear(hidden_size,1)

    def forward(self,x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = true, enforce_sorted = false)
        output, hidden = self.rnn(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first = true)
        output = self.fc(output)
        return output


input_size = 10
hidden_size = 20
batch_size = 4
max_length = 7

model = maskedrnn(input_size, hidden_size)

# example of a batch with variable length
input_data = torch.randn(batch_size, max_length, input_size)
lengths = torch.tensor([5, 3, 7, 2]) # length of each sequence
# lets make the sequences valid to pass to the rnn cell
masks = torch.arange(max_length).expand(len(lengths),max_length) < lengths[:, None]
input_data = input_data * masks[:,:,None] # zero out padded parts


# forward pass
output = model(input_data, lengths)
print(output.shape) # this will return torch.size([4,7,1])

```

here, you can see how i use `pack_padded_sequence` and `pad_packed_sequence` which is provided by pytorch to automatically handle the masking, this avoids the necessity of doing the operations by hand and simplifies the code, other frameworks such as tensorflow do have similar functions.

now, for the loss function. we can't just average the loss over all the timesteps, that will bias the gradient towards the loss on the padded part, which we know is not meaningful. so, we compute the loss only on non-padded parts and then average this only by the non-masked part of the sequence. here's how it might look:

```python
import torch
import torch.nn as nn

def masked_mse(predictions, targets, masks):
  loss = (predictions - targets)**2
  loss = loss*masks
  loss_sum = torch.sum(loss)
  mask_sum = torch.sum(masks)
  return loss_sum/mask_sum

# Example Usage
predictions = torch.randn(4, 7, 1)
targets = torch.randn(4, 7, 1)

lengths = torch.tensor([5, 3, 7, 2])
max_length = 7
masks = torch.arange(max_length).expand(len(lengths),max_length) < lengths[:, None]

loss = masked_mse(predictions, targets, masks[:,:,None])
print(loss) # single value tensor with the final mean squared error loss

```

as you see in this example, the loss function is explicitly calculating the error only on valid parts of the sequences, and avoiding any contribution of the padding on the final result and gradients.

finally, let's talk about masking when you are not doing a simple loss function but need to do a classification or other different type of operation on the outputs. say that you need to do a classification based only on the last step of the rnn:

```python
import torch
import torch.nn as nn

class laststeprnn(nn.module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(laststeprnn, self).__init__()
        self.rnn = nn.rnn(input_size, hidden_size, batch_first = true)
        self.fc = nn.linear(hidden_size, num_classes)

    def forward(self,x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = true, enforce_sorted = false)
        output, hidden = self.rnn(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first = true)
        # get last step outputs
        last_outputs = output[torch.arange(len(lengths)), lengths -1]
        output = self.fc(last_outputs)
        return output


input_size = 10
hidden_size = 20
num_classes = 5
batch_size = 4
max_length = 7

model = laststeprnn(input_size, hidden_size, num_classes)

# example of a batch with variable length
input_data = torch.randn(batch_size, max_length, input_size)
lengths = torch.tensor([5, 3, 7, 2]) # length of each sequence
# lets make the sequences valid to pass to the rnn cell
masks = torch.arange(max_length).expand(len(lengths),max_length) < lengths[:, None]
input_data = input_data * masks[:,:,None] # zero out padded parts


# forward pass
output = model(input_data, lengths)
print(output.shape) # this will return torch.size([4,5])

```

in this example, i show how to extract the last step outputs using the information provided by the `lengths` tensor, this lets the model output a sequence based on the last step. this is extremely useful in multiple applications like classification or when the output of the rnn is used in other type of network.

there are some caveats you should know. when doing multi gpu training, padding can be tricky, i had a time when the input data was correct and i was still having issues with my training and this happened to be because the padding was being done in different ways between devices. this ended up in the same example having different paddings at the beginning of the network, so be sure that everything that can change during the forward pass is done in the same way between gpus. this will prevent issues that are hard to understand.

in short, variable length backpropagation isn't as intimidating as it sounds once you nail down the concepts of padding and masking. the key is to handle sequences individually within batches and make sure that paddings do not affect the gradients.

if you want to dive deeper, i would recommend looking into the original seq2seq paper and also this book "deep learning with python" by francois chollet, as both of them provide a comprehensive explanation about recurrent neural networks and techniques that are useful to deal with variable length sequences.
oh, and one time, i spent a whole weekend debugging a gradient issue, only to discover i'd accidentally divided by zero in my loss function; it's a good reminder that even us experienced people can miss really simple things. it just proves that the problem was user related not code related after all.
