---
title: "How can FID scores be calculated for GAN models comparing images from two different directories?"
date: "2025-01-30"
id: "how-can-fid-scores-be-calculated-for-gan"
---
The Fréchet Inception Distance (FID) score, while generally understood in the context of GAN performance evaluation, often presents a practical challenge when implemented for comparing images residing in distinct directory structures, as opposed to neatly packaged NumPy arrays. I’ve encountered this hurdle repeatedly in my work with generative models, and a robust solution requires careful attention to data loading and feature extraction pipelines.

Fundamentally, FID calculates the distance between two multivariate Gaussian distributions, each representing the feature space of real and generated images, extracted through an Inception-v3 network. The score quantifies this distance, essentially indicating how well the generated images match the distribution of the real images; a lower FID score implies better image quality and closer alignment with the target distribution. Directly applying FID to directory-based image sets involves constructing two suitable data loaders, extracting their respective feature distributions, and then calculating the score using those features.

Let's break down the process:

**Data Loading and Preprocessing:**

The primary concern is setting up data loading that handles images from separate directories consistently. This involves leveraging Python’s `os` module for directory traversal and PIL (Pillow) for image handling. We need functions to recursively gather all image files within each directory and create generators or iterators for feeding these to our preprocessing pipeline. Images should be uniformly resized to the input resolution of the Inception-v3 model (299x299 pixels) and normalized to the expected range (usually between -1 and 1). I prefer creating custom data loaders for precise control over this process. For efficiency, we should load batches of images rather than individual ones and employ multiprocessing to speed up the loading and preprocessing phase.

**Feature Extraction:**

Once the images are loaded, we use the Inception-v3 network, pre-trained on ImageNet, to extract features. We’re interested in the activations of a penultimate layer of the network, which serve as the representation of images within a high-dimensional feature space. These activations are what we'll model as a multivariate Gaussian. A crucial optimization here involves running the images in batches through the Inception network on GPU if available, significantly accelerating the extraction process. It's also advisable to disable gradient calculations as we only need forward propagation and not backpropagation. We accumulate these activations into numpy arrays, one for each directory.

**FID Calculation:**

With the feature vectors extracted for both the real and the generated images, the next step involves calculating the Fréchet distance. This requires calculating the means and covariance matrices of the two Gaussian distributions formed by the respective feature vectors. The FID score itself is computed using the means, the trace of the square root of the product of covariance matrices, and some intermediate matrix multiplication. Numerical stability is vital for this part, and one should be mindful of how these matrices are calculated.

**Code Example 1: Data Loading Function:**

```python
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                   self.image_paths.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
           image = self.transform(image)
        return image


def create_dataloader(directory, batch_size, num_workers=4, resize=299):
  transform = transforms.Compose([
      transforms.Resize(resize),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
  dataset = ImageDataset(directory, transform=transform)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  return dataloader

# Example Usage:
# real_dataloader = create_dataloader("path/to/real_images", batch_size=64)
# generated_dataloader = create_dataloader("path/to/generated_images", batch_size=64)

```

*Commentary:* This code snippet introduces a `ImageDataset` class inheriting from PyTorch’s `Dataset`, designed for reading images from a directory. It automatically traverses subdirectories for compatibility. It loads and applies appropriate transforms to each image. The `create_dataloader` function creates and returns a PyTorch Dataloader ready for image loading in batch processing. The inclusion of the normalization transform makes the data compatible with pre-trained Inception-v3.

**Code Example 2: Feature Extraction Function:**

```python
from torchvision.models import inception_v3
import torch
from tqdm import tqdm

def extract_features(dataloader, device, use_gpu):
    inception_model = inception_v3(pretrained=True, progress=False, transform_input=False).to(device)
    inception_model.eval()
    
    if use_gpu:
        inception_model = inception_model.cuda()
    
    feature_vectors = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            if use_gpu:
                images = images.cuda()
            features = inception_model(images)
            # Use the layer output right before classification.
            features = features.detach().cpu().numpy() 
            feature_vectors.append(features)

    feature_vectors = np.concatenate(feature_vectors, axis=0)
    return feature_vectors

# Example Usage:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# real_features = extract_features(real_dataloader, device, use_gpu=torch.cuda.is_available())
# generated_features = extract_features(generated_dataloader, device, use_gpu=torch.cuda.is_available())
```

*Commentary:* The `extract_features` function loads a pre-trained Inception-v3 model, places it on the GPU if available, and then loops through each batch of images in the provided `DataLoader`. It disables gradient tracking using `torch.no_grad()` to ensure efficient forward propagation. Crucially, it detaches the feature activations from the computational graph and moves it back to CPU for numpy based computation. The function finally concatenates feature outputs to create a single array and returns the extracted features as a numpy array.

**Code Example 3: FID Calculation Function:**

```python
import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(real_features, generated_features):
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    mu2 = np.mean(generated_features, axis=0)
    sigma2 = np.cov(generated_features, rowvar=False)

    diff = mu1 - mu2

    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Example Usage:
# fid_score = calculate_fid(real_features, generated_features)
# print(f"FID Score: {fid_score}")
```

*Commentary:* This function takes the extracted feature vectors for real and generated images. It computes the mean (mu) and covariance matrix (sigma) for each set. It then calculates the square root of the product of these covariance matrices using `scipy.linalg.sqrtm`. Handles the complex return type of the function by taking its real part and completes the FID score calculation using the calculated parameters. It returns the final FID score which will be a single numeric value representing the similarity of two distribution of feature vectors.

**Resource Recommendations:**

For further study beyond this discussion, I would recommend consulting resources that delve into the mathematical underpinnings of the Fréchet Distance, exploring its relationship to the Wasserstein distance, and papers focusing on the Inception-v3 network architecture. Further reading about efficient data loading techniques in PyTorch, particularly for image datasets, is also crucial for creating robust solutions. Investigate papers published on GAN evaluation metrics beyond just FID and consider implementations that use Tensorboard to track FID score over training epochs. These sources will solidify your understanding of GAN evaluation and equip you with the knowledge needed for more specialized evaluations in future projects. A deeper understanding of multivariate statistics and numerical computation would also be beneficial.
