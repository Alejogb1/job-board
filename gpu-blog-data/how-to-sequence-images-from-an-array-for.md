---
title: "How to sequence images from an array for LSTM input?"
date: "2025-01-30"
id: "how-to-sequence-images-from-an-array-for"
---
In my experience building time-series forecasting models, I've frequently encountered the challenge of properly formatting image data for consumption by Long Short-Term Memory (LSTM) networks. The critical issue lies in transforming a collection of static images, each potentially multi-dimensional (e.g., color RGB), into a sequential format amenable to LSTM's temporal processing. Unlike other network types that process data in a single step, LSTMs require input in the form of a sequence, where each element is conceptually related to its predecessor. This necessitates careful consideration of data dimensions and their arrangement.

The core of the issue is transforming an unordered set of images into a 3-dimensional tensor suitable for LSTM input. Specifically, the dimensions are typically `(batch_size, time_steps, features)`. Here, `batch_size` represents the number of independent sequences processed in parallel; `time_steps` denotes the sequence length (number of images used in a sequence); and `features` are the pixel data representation of a single image. A naive approach of merely reshaping each image individually and concatenating them could lead to a loss of the temporal relationship, which is what LSTMs are designed to understand. Therefore, a preprocessing step is essential to correctly structure the data.

To illustrate, consider a situation where I have a set of 50 greyscale images, each of size 64x64 pixels, representing frames from a short video clip.  My objective is to train an LSTM network to learn patterns in this sequence, perhaps for motion recognition or future frame prediction. The raw image data sits within an array-like structure where each element corresponds to a single frame.

Here's my typical approach, encompassing three practical scenarios:

**Scenario 1: Single Sequence Input**

In the simplest case, when treating the entire dataset as one long temporal sequence, I can structure my data as follows. Note, for illustrative purposes,  I’m assuming usage of a Python library like NumPy for array manipulation.

```python
import numpy as np

# Assume 'image_dataset' is a list of 50 numpy arrays, each of shape (64, 64)
image_dataset = [np.random.rand(64,64) for _ in range(50)] # example dataset, replace with actual images

def prepare_single_sequence(images):
    num_images = len(images)
    image_height, image_width = images[0].shape
    
    # Stack the images into a single numpy array of shape (50, 64, 64)
    stacked_images = np.stack(images, axis=0) 
    
    # Reshape to (50, 64*64). This gives us 50 time steps, each with 4096 features
    reshaped_images = stacked_images.reshape(num_images, image_height * image_width) 

    # Reshape for LSTM input. (Batch Size, Timesteps, Features) where batch_size = 1
    lstm_input = reshaped_images.reshape(1, num_images, image_height * image_width)
    
    return lstm_input


lstm_input_single = prepare_single_sequence(image_dataset)

print(f"Shape of LSTM input (single): {lstm_input_single.shape}")
# Expected Output: Shape of LSTM input (single): (1, 50, 4096)
```

In this first example, `prepare_single_sequence()` takes the list of images. First, all image arrays are stacked, resulting in a 3D array `(50,64,64)`.  Then, this array is reshaped such that all the pixels within each image become a feature vector, transforming the structure to `(50, 4096)`. Finally, another reshaping is applied to create an array that signifies a single sequence, resulting in `(1, 50, 4096)`. This is the expected input format for an LSTM layer accepting a single batch of data. The  `stack` operation ensures each image maintains its original spatial information, but the reshape transforms them to feature vectors compatible with the LSTM’s input.

**Scenario 2: Multiple Sequences with Overlap**

For tasks where the entire data does not form a single long sequence but instead several overlapping sequences (e.g., where temporal windows are analyzed), further preprocessing is needed. For instance, I may want to provide the LSTM with sequences of length 10 with an overlap of 5 frames.

```python
def prepare_multiple_sequences(images, seq_length, overlap):
    num_images = len(images)
    image_height, image_width = images[0].shape
    
    sequences = []
    for i in range(0, num_images - seq_length + 1, seq_length - overlap):
        
        seq = images[i:i+seq_length]
        stacked_seq = np.stack(seq, axis=0)
        reshaped_seq = stacked_seq.reshape(seq_length, image_height * image_width)
        sequences.append(reshaped_seq)

    lstm_input = np.stack(sequences, axis=0) # stack to get (Batch Size, Timesteps, Features)

    return lstm_input

seq_length = 10
overlap = 5
lstm_input_multi = prepare_multiple_sequences(image_dataset, seq_length, overlap)
print(f"Shape of LSTM input (multiple, overlapping): {lstm_input_multi.shape}")
#Expected output: Shape of LSTM input (multiple, overlapping): (9, 10, 4096)

```

Here,  `prepare_multiple_sequences()` introduces the concepts of `seq_length` and `overlap`, parameters that define how the data is windowed. In the loop, a sequence `seq` is created, corresponding to the `seq_length`. Once all sequences are generated, they are stacked together, forming the desired 3D structure. This procedure allows the network to learn over multiple subsequences, useful for identifying local temporal patterns. The stride during sequence construction is controlled by `seq_length - overlap` which in this case results in an overlapping sequence creation, enhancing data representation.

**Scenario 3: Multiple Sequences Without Overlap**

Another common approach is segmenting the data into non-overlapping sequences. This is particularly relevant when input data is explicitly divisible into distinct time windows, such as individual video clips from a larger dataset.

```python
def prepare_multiple_non_overlapping_sequences(images, seq_length):
    num_images = len(images)
    image_height, image_width = images[0].shape

    sequences = []
    for i in range(0, num_images, seq_length):
        if i + seq_length > num_images: # ensure we don't go out of bounds
            break
        seq = images[i:i+seq_length]
        stacked_seq = np.stack(seq, axis=0)
        reshaped_seq = stacked_seq.reshape(seq_length, image_height * image_width)
        sequences.append(reshaped_seq)
    
    lstm_input = np.stack(sequences, axis=0)  

    return lstm_input

seq_length = 10
lstm_input_non_overlap = prepare_multiple_non_overlapping_sequences(image_dataset, seq_length)
print(f"Shape of LSTM input (multiple, non-overlapping): {lstm_input_non_overlap.shape}")
# Expected output: Shape of LSTM input (multiple, non-overlapping): (5, 10, 4096)

```

`prepare_multiple_non_overlapping_sequences()` is similar to the overlapping version but advances the sequence start point with the stride equal to the `seq_length`. This effectively divides the images into a series of non-overlapping, sequentially ordered subsequences. This approach avoids redundancy between input sequences and promotes the modeling of independent events represented by individual sequences. The use of the `if` statement prevents a malformed sequence from being generated when number of images is not divisible by the specified sequence length.

Throughout my projects, I've found careful preprocessing of time-series images significantly impacts the performance of LSTM-based architectures. The choice between single, overlapping, or non-overlapping sequences depends on the nature of the problem. Consider temporal dependencies and potential redundancy for optimal network performance. Furthermore, techniques like padding for sequences of varying lengths may be needed and have been left out for clarity. I recommend reviewing material on deep learning with recurrent neural networks, paying particular attention to handling time-series data. Resources on convolutional LSTMs and sequence-to-sequence models would offer further insights. Further, exploration of frameworks like TensorFlow or PyTorch provide additional tutorials on working with time-series data, particularly in the context of image sequences. These frameworks can offer alternative methods for this conversion that can often be more efficient, particularly for large datasets.
