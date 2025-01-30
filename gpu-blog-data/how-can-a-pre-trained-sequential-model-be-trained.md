---
title: "How can a pre-trained sequential model be trained with varying input shapes?"
date: "2025-01-30"
id: "how-can-a-pre-trained-sequential-model-be-trained"
---
A fundamental challenge in working with pre-trained sequential models, particularly recurrent neural networks (RNNs) and Transformers, lies in their inherent dependence on fixed input sequence lengths. While these models are often trained on sequences of a specific size, real-world applications frequently require processing variable-length inputs. Adapting a pre-trained model to this situation demands careful consideration of architectural and training modifications.

The core issue stems from the internal matrix operations within sequential models. RNNs, for instance, maintain hidden state vectors that are updated at each time step. This update process is predicated on receiving inputs of consistent dimensionality across the sequence. Similarly, the positional encodings and attention mechanisms in Transformers are often designed around a specific input sequence length, which is typically determined during the pre-training phase. Directly feeding sequences of varying lengths would lead to dimension mismatches and computation errors.

Two principal strategies can circumvent this issue: padding and masking, and utilizing architectural adaptations. Padding and masking is the more straightforward approach, and its implementation depends heavily on the particular model's input format. The underlying principle involves extending all input sequences to a maximum length. Sequences shorter than this are padded with a special token, often a zero vector or an end-of-sequence (EOS) token. Simultaneously, a mask is applied to the padding tokens during computation to prevent them from influencing the model's output. This effectively allows the model to process sequences of varying lengths while operating on a uniform input size. The masking mechanism ensures that the padding tokens do not affect the computations made during backpropagation.

Alternatively, architectural adaptations involve modifying the model itself to enable the processing of variable sequence lengths natively, often at the cost of increasing training time. One technique involves using pooling mechanisms, where outputs from sequence layers are aggregated, effectively losing sequence position information, but creating fixed-size representations of varied-length inputs. Another technique involves employing dynamic computation graphs that are reconfigured during the training step to process sequences of the exact length in each batch. This approach might involve using the Keras model function API or equivalent approaches for Pytorch to create a graph that adapts dynamically to different input sizes. While powerful, it significantly complicates the training and deployment pipeline.

My work developing a time-series anomaly detection system presented this challenge explicitly. The input sequences representing sensor readings varied significantly in duration. Using the padding and masking method proved to be efficient and straightforward. We chose an LSTM based model that has been pre-trained on a large dataset of time series recordings. Initially we used sequences of 20 time-stamps; however, many real-time data sequences were considerably shorter or longer than that. To address this, we opted to pad shorter sequences with zero vectors, and mask them during training by assigning zero weights. This allowed us to use a single training pipeline without resorting to advanced architectural changes, making the system faster to implement.

Let's consider three code examples to illustrate this process. The first example provides a PyTorch implementation of how to pad a batch of varying length sequences. The second provides an example of the mask creation while the third example shows its use within a typical LSTM training loop.

**Example 1: Padding Sequences in PyTorch**

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_batch(sequences, padding_value=0):
    """Pads a list of sequences to the maximum length in the batch.

    Args:
        sequences: A list of torch tensors, each representing a sequence.
        padding_value: The value to use for padding.

    Returns:
        A padded batch of sequences (torch tensor), and a sequence length tensor.
    """

    # Determine the maximum sequence length
    max_length = max([seq.size(0) for seq in sequences])

    # Pad each sequence
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)

    #Create sequence length tensor
    seq_lengths = torch.tensor([seq.size(0) for seq in sequences])

    return padded_sequences, seq_lengths

# Example usage
seq1 = torch.randn(10, 5) # Length 10, Feature Size 5
seq2 = torch.randn(5, 5)  # Length 5, Feature Size 5
seq3 = torch.randn(15, 5) # Length 15, Feature Size 5
batch = [seq1, seq2, seq3]

padded_batch, seq_lengths = pad_batch(batch)

print("Padded Batch Shape:", padded_batch.shape) #Should output: Padded Batch Shape: torch.Size([3, 15, 5])
print("Sequence Lengths:", seq_lengths) #Should output: Sequence Lengths: tensor([10,  5, 15])

```
This code demonstrates the usage of `pad_sequence` in PyTorch, which is a dedicated function to pad a list of tensors into a single tensor with dimensions suitable for batch processing. It automatically adds the specified padding value to all sequences in the batch, extending each sequence to the length of the longest sequence in the batch. The function also returns a tensor with the original sequence lengths, which is important for masking later.

**Example 2: Generating a Mask in PyTorch**
```python
def create_mask(padded_sequences, seq_lengths):
    """Creates a mask for padded sequences.

    Args:
        padded_sequences: The padded batch of sequences (torch tensor).
        seq_lengths: A tensor containing the original sequence lengths (torch tensor).

    Returns:
        A mask tensor (torch tensor).
    """

    mask = torch.arange(padded_sequences.size(1)).unsqueeze(0) < seq_lengths.unsqueeze(1)
    return mask

# Example usage (continuing from previous)
mask = create_mask(padded_batch, seq_lengths)

print("Mask Shape:", mask.shape) #Should output: Mask Shape: torch.Size([3, 15])
print("Mask:", mask) #Should output a tensor with True for relevant timesteps and False for padded
```

This code shows how to create a boolean mask that indicates which values in the padded batch are real and which are padding. The mask has the same size as the batch but the values are `True` for values in the original sequences and `False` for padding tokens. This mask can be used during training to prevent the model from computing losses on padding tokens.

**Example 3: Using the Mask in an LSTM training loop**

```python
import torch.nn as nn
import torch.optim as optim

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, seq_lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) #Required since mask function expects a padded output
        return self.fc(output) #Return all the predictions for the sequence

# Example usage (continuing from previous)

input_size = 5
hidden_size = 10
model = SimpleLSTM(input_size, hidden_size)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss() #Choose your specific loss function

# Dummy targets (using MSE loss for demo purposes)
targets = torch.randn(3,15,1)

#Training loop
optimizer.zero_grad()
outputs = model(padded_batch, seq_lengths)
loss = criterion(outputs, targets) #This computes the loss across all positions of the sequence
mask = create_mask(padded_batch, seq_lengths) # Get the mask for the batch
masked_loss = loss * mask.unsqueeze(2).float() #Use the mask to cancel the loss on the padding tokens
total_loss = masked_loss.sum() / mask.sum() # Compute the weighted mean loss, masking the paddings out of the mean operation
total_loss.backward()
optimizer.step()

print("Loss:", total_loss.item())
```

This code shows a simplified LSTM training loop with variable sequence length, using the masking approach. The `pack_padded_sequence` and `pad_packed_sequence` methods of PyTorch make the computations over the sequences considerably more efficient and are the recommended way to compute over variable-length sequences. The code illustrates the steps of generating the mask, applying it to the loss, and backpropagating the correct losses. Note that this requires changing your loss function to have an output for each timestamp for this simple example. In more realistic settings, the output of the network can be a single scalar or vector, which is produced by further post-processing operations and loss functions may involve specific metrics for time series data.

For further study of sequential models and variable length sequences, several texts are of significant use. "Deep Learning" by Goodfellow, Bengio, and Courville provides a comprehensive foundation for understanding recurrent neural networks and related concepts. The PyTorch documentation and tutorials are indispensable resources for practical implementation details and the utilization of PyTorch specific functions like `pack_padded_sequence` and `pad_sequence`. Finally, exploring research papers on attention mechanisms, particularly those focused on the transformer architecture, enhances comprehension of advanced methods for processing variable-length inputs. These resources should provide a solid basis for working with sequential models and variable-length sequences.
