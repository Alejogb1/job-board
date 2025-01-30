---
title: "How do I create train and test datasets using PyTorch's DataLoader?"
date: "2025-01-30"
id: "how-do-i-create-train-and-test-datasets"
---
In my experience, correctly partitioning data into training and testing sets is foundational for effective model development in PyTorch; a failure to do this rigorously undermines the entire learning process. The `DataLoader` class itself does not handle dataset splitting. Instead, it’s responsible for efficient data loading, batching, and shuffling. Therefore, the division into train and test sets must occur *before* you instantiate your `DataLoader` objects. I will detail a common method using PyTorch's `random_split` function and provide practical examples.

The core idea revolves around having an underlying dataset, often represented by a class inheriting from `torch.utils.data.Dataset`, which provides access to your data and labels. This dataset holds the entirety of your available information. To separate it into training and testing subsets, the most straightforward approach involves `torch.utils.data.random_split`. This function takes a dataset and lengths as input and returns new dataset objects corresponding to those specified lengths. Crucially, it shuffles the original dataset and splits it into non-overlapping parts, ensuring that no data point appears in both training and testing sets. This randomness is important to avoid biases in how your model is trained.

Once you have your training and testing datasets, you create separate `DataLoader` objects for each. These dataloaders iterate over your datasets, providing batches of samples that are fed into your neural network during training and evaluation. Different `DataLoader` settings, such as batch size, shuffle parameters, or the number of worker threads, can be customized for the training and testing stages. For example, shuffling is typically enabled during training but is often unnecessary during testing or validation.

Here are three examples demonstrating how this is done, starting from a synthetic dataset to a real-world type problem.

**Example 1: Synthetic Dataset**

Let's assume we have a simple synthetic dataset represented by `TensorDataset` from PyTorch. This approach is helpful in scenarios where you're prototyping or experimenting with the pipeline without loading data from files.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# Create synthetic data (100 samples, each with 10 features)
data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,)) # Binary labels

# Create a TensorDataset
dataset = TensorDataset(data, labels)

# Calculate the sizes of the training and testing splits
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Randomly split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example usage (loop through the train_loader)
for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
    print(f"Train Batch {batch_idx}: data shape {batch_data.shape}, labels shape {batch_labels.shape}")

# Example usage (loop through the test_loader)
for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
    print(f"Test Batch {batch_idx}: data shape {batch_data.shape}, labels shape {batch_labels.shape}")
```

In this example, I first created synthetic data and corresponding binary labels. I then wrapped this data into a `TensorDataset`. Afterwards, I used `random_split` to divide the data into 80% for training and 20% for testing. Two distinct `DataLoader` objects were created, each configured with a batch size of 32.  Shuffling was enabled for training to prevent any ordering dependencies, but disabled for testing to maintain the original data order. The print statements at the end demonstrate looping through the dataloaders and inspecting the batch shapes.

**Example 2: Custom Dataset with Image Data**

Now, suppose we have a custom dataset that involves loading images from a directory. This is a more realistic scenario. Here, a custom class inheriting from `Dataset` would be more appropriate. We will also handle some basic image transforms using `torchvision.transforms`.

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg','.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB') # Convert to RGB to standardize channels
        # Assuming images with binary labels encoded in file names as "image_label.ext"
        label = int(os.path.basename(image_path).split("_")[1].split(".")[0]) # Extract the label from filename
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Replace with the actual path to image data.
# Images should be in the format 'image_{label}.jpg' or 'image_{label}.png'
image_folder = './data/images' # Assume this directory exists and contains images.

# Create the custom dataset
dataset = ImageDataset(root_dir=image_folder, transform=transform)

# Calculate sizes and split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4) # Added num_workers
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Example usage
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Train Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")

for batch_idx, (images, labels) in enumerate(test_loader):
    print(f"Test Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
```

This more advanced example utilizes a custom `ImageDataset` class. It initializes with a directory containing image files. Each `__getitem__` function opens an image, extracts a label (assuming labels are embedded in the filename), applies transforms, and returns both image and label. The crucial part again, is that the `random_split` function divides the dataset into training and testing subsets. The `DataLoader` is configured with additional `num_workers` to use multiple threads to load data, speeding up the training. I have included the normalization as it's common for image processing.

**Example 3: Sequential Data with Padding and Masking**

Finally, consider sequential data, like text, where sequences can have variable lengths. Padding and masking become important. This example shows how to handle that in the dataset class.

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        numericalized_text = [self.vocab.get(token, 0) for token in text] # Use 0 as OOV
        return torch.tensor(numericalized_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def pad_collate(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    padded_inputs = pad_sequence(inputs, batch_first=True)
    return padded_inputs, torch.stack(labels) # Stack labels to create a batch

# Example text and labels
texts = [
    ["hello", "world"],
    ["this", "is", "a", "test"],
    ["deep", "learning", "is", "fun"],
    ["python", "programming"]
]
labels = [0, 1, 1, 0]

# Example vocabulary
vocab = {"hello": 1, "world": 2, "this": 3, "is": 4, "a": 5, "test": 6, "deep": 7, "learning": 8, "fun": 9, "python":10, "programming": 11}

# Max sequence length for padding
max_len = 10

# Create the dataset
dataset = TextDataset(texts, labels, vocab, max_len)

# Split the dataset
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create the DataLoaders using a custom collate_fn
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=pad_collate)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=pad_collate)

# Example usage
for batch_idx, (inputs, labels) in enumerate(train_loader):
    print(f"Train Batch {batch_idx}: inputs shape {inputs.shape}, labels shape {labels.shape}")

for batch_idx, (inputs, labels) in enumerate(test_loader):
    print(f"Test Batch {batch_idx}: inputs shape {inputs.shape}, labels shape {labels.shape}")
```

In this text data example, I defined a `TextDataset` which expects a list of tokenized sequences. Crucially, I utilize a `pad_collate` function as the `collate_fn` of the `DataLoader`. This function pads the sequences within a batch to the same length before they are fed into the model. Once more, the dataset is split into training and testing sets via `random_split`.

**Resource Recommendations**

For further learning on this topic, I recommend studying the official PyTorch documentation on `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, and `torch.utils.data.random_split`. Additionally, reading guides and tutorials on custom dataset creation and handling different data modalities, such as images and text, is highly beneficial. Practical examples on specific problem types, for example those involving natural language processing, can be found in research papers and other open-source machine-learning repositories. Studying these implementations provides a great way to understand best practices and adapt them to your own use cases. Finally, understanding the conceptual difference between a `Dataset` object (holding raw data) and a `DataLoader` (iterating over batched, processed data), and how they fit within the overall training pipeline, is paramount.
