---
title: "How do I export a WAV file from a PyTorch tensor?"
date: "2025-01-30"
id: "how-do-i-export-a-wav-file-from"
---
Exporting a waveform represented as a PyTorch tensor to a standard WAV file requires a bridging step involving audio processing libraries, since PyTorch itself primarily handles numerical computations and does not inherently manage audio file I/O. I’ve encountered this frequently during my work on neural audio synthesis where model output is typically a floating-point tensor and requires conversion to an audio format for playback or further processing.

The fundamental challenge lies in transforming a tensor's numeric representation of an audio signal into a structured, encoded audio file. A WAV file, in its essence, is a RIFF (Resource Interchange File Format) container that stores the raw audio samples in a linear pulse-code modulation (PCM) format, along with metadata describing sample rate, bit depth, and channel configuration. PyTorch tensors, on the other hand, are generic multidimensional arrays designed for mathematical operations. Therefore, converting between these two requires carefully handling format specifications and ensuring that the PCM audio data in the WAV file corresponds correctly to the numerical representation in the tensor.

The core process involves three primary stages: Firstly, you must ensure the PyTorch tensor has the correct shape, data type, and range expected for PCM audio. Secondly, the raw PCM samples need to be converted from the tensor's numerical data type (usually floating point) to an integer representation suitable for the WAV file format. Thirdly, this integer representation must be written into a correctly structured WAV file. Libraries such as `scipy.io.wavfile` or `soundfile` are instrumental in handling the WAV file creation and data packing aspects. These libraries abstract the intricacies of RIFF header generation and data block organization, allowing us to focus on the signal processing aspects of the conversion. The key is to understand how to prepare the tensor to comply with these libraries’ data format expectations.

Let me illustrate this with several Python code examples. The first one will use `scipy.io.wavfile`:

```python
import torch
import scipy.io.wavfile

def tensor_to_wav_scipy(tensor, sample_rate, filepath):
    """Converts a PyTorch tensor to a WAV file using scipy.io.wavfile.

    Args:
        tensor (torch.Tensor): Audio tensor with shape (channels, samples) or (samples).
                                    Expected to have values between -1 and 1.
        sample_rate (int): Sample rate of the audio signal.
        filepath (str): Path to save the WAV file.
    """
    if tensor.ndim == 2: # Check for multiple channel audio
        tensor = tensor.permute(1, 0) # Make samples the primary dimension
    elif tensor.ndim != 1:
        raise ValueError("Tensor must have 1 or 2 dimensions.")

    # Ensure tensor is in the correct range for int16 PCM
    scaled_tensor = (tensor * 32767).clamp(-32767, 32767).to(torch.int16)
    
    # Convert the tensor to a numpy array for scipy
    audio_data = scaled_tensor.cpu().numpy()
    
    scipy.io.wavfile.write(filepath, sample_rate, audio_data)

# Example Usage:
sample_rate = 16000
duration = 2 # seconds
num_samples = sample_rate * duration
# generate a test signal (a sine wave for clarity)
frequency = 440
time = torch.arange(num_samples) / sample_rate
sine_wave = torch.sin(2 * torch.pi * frequency * time).float()
tensor_to_wav_scipy(sine_wave, sample_rate, "sine_wave_scipy.wav")

# multi channel test signal
channels = 2
multi_channel_sine = torch.stack([sine_wave, sine_wave * 0.5 ], dim = 0)
tensor_to_wav_scipy(multi_channel_sine, sample_rate, "multi_sine_scipy.wav")
```

In this first snippet, we begin by importing the necessary libraries, `torch` for tensor operations and `scipy.io.wavfile` for writing the WAV file. The function `tensor_to_wav_scipy` takes a PyTorch tensor, a sample rate, and a filepath as input. The crucial step here is scaling and clamping the tensor from the range [-1, 1] (which is a common range in neural audio work) to the range [-32767, 32767] and then converting it to int16, as required by `scipy.io.wavfile` when writing a 16-bit WAV file. Notice how I used a `.cpu()` call; this is to move the tensor to CPU memory before converting it to a NumPy array since `scipy.io.wavfile` expects a NumPy array. The function manages both mono and stereo signals. The example demonstrates basic sine wave generation and its storage to a WAV file using `scipy.io.wavfile`. We demonstrate single channel and multi-channel waveform storage.

Now, let's consider an alternative approach using the `soundfile` library:

```python
import torch
import soundfile as sf

def tensor_to_wav_soundfile(tensor, sample_rate, filepath):
    """Converts a PyTorch tensor to a WAV file using soundfile.

    Args:
        tensor (torch.Tensor): Audio tensor with shape (channels, samples) or (samples).
                                    Expected to have values between -1 and 1.
        sample_rate (int): Sample rate of the audio signal.
        filepath (str): Path to save the WAV file.
    """
    if tensor.ndim == 2:
        tensor = tensor.permute(1, 0) #Samples first
    elif tensor.ndim != 1:
        raise ValueError("Tensor must have 1 or 2 dimensions.")

    audio_data = tensor.cpu().numpy()
    sf.write(filepath, audio_data, sample_rate, subtype='PCM_16')


# Example Usage:
sample_rate = 16000
duration = 2 # seconds
num_samples = sample_rate * duration
# generate a test signal (a sine wave for clarity)
frequency = 440
time = torch.arange(num_samples) / sample_rate
sine_wave = torch.sin(2 * torch.pi * frequency * time).float()

tensor_to_wav_soundfile(sine_wave, sample_rate, "sine_wave_soundfile.wav")

#multi channel test signal
channels = 2
multi_channel_sine = torch.stack([sine_wave, sine_wave * 0.5], dim=0)
tensor_to_wav_soundfile(multi_channel_sine, sample_rate, "multi_sine_soundfile.wav")

```

The second function, `tensor_to_wav_soundfile`, implements the same conversion logic using `soundfile`. `soundfile` offers a different interface for writing WAV files, which in this context, is more direct. Here, the crucial scaling to the appropriate integer range is handled within the `sf.write` function via the `subtype='PCM_16'` parameter implicitly. The main difference from the scipy method is that we do not have to convert to int16 explicitly. We do still have to move the tensor to the CPU using the `.cpu()` call prior to conversion to numpy array. It is important to note that the `soundfile` library can automatically scale floating-point data to the specified subtype; this avoids the manual scaling step required in `scipy.io.wavfile` in this example. Again we show both mono and stereo signal storage.

Finally, let us consider the case where we need a 24-bit representation, instead of 16 bit which is the default case for both `scipy` and `soundfile`:

```python
import torch
import soundfile as sf
import numpy as np

def tensor_to_wav_soundfile_24bit(tensor, sample_rate, filepath):
    """Converts a PyTorch tensor to a 24-bit WAV file using soundfile.

    Args:
        tensor (torch.Tensor): Audio tensor with shape (channels, samples) or (samples).
                                    Expected to have values between -1 and 1.
        sample_rate (int): Sample rate of the audio signal.
        filepath (str): Path to save the WAV file.
    """
    if tensor.ndim == 2:
        tensor = tensor.permute(1, 0) #Samples first
    elif tensor.ndim != 1:
       raise ValueError("Tensor must have 1 or 2 dimensions.")

    scaled_tensor = (tensor * (2**23 - 1)).clamp(-2**23 + 1, 2**23 - 1).to(torch.int32)
    audio_data = scaled_tensor.cpu().numpy()
    sf.write(filepath, audio_data, sample_rate, subtype='PCM_24')


# Example Usage:
sample_rate = 16000
duration = 2 # seconds
num_samples = sample_rate * duration
# generate a test signal (a sine wave for clarity)
frequency = 440
time = torch.arange(num_samples) / sample_rate
sine_wave = torch.sin(2 * torch.pi * frequency * time).float()

tensor_to_wav_soundfile_24bit(sine_wave, sample_rate, "sine_wave_soundfile_24bit.wav")

#multi channel test signal
channels = 2
multi_channel_sine = torch.stack([sine_wave, sine_wave * 0.5], dim = 0)
tensor_to_wav_soundfile_24bit(multi_channel_sine, sample_rate, "multi_sine_soundfile_24bit.wav")
```

The primary change here is in the scaling. To write a 24-bit PCM file, we have to scale the floating-point tensor (values -1 to 1) to integers in the range of -2^23 to 2^23 -1. This range is achieved by scaling by 2^23 -1. Because the WAV file format is integer-based, the data type has to be converted to int32 to accommodate the 24 bit integer representation. This also demonstrates that we can readily modify the format of the created files using the same underlying libraries. As before we show single and multi-channel versions of this example.

In summary, converting a PyTorch tensor to a WAV file fundamentally involves the proper scaling, clamping, and formatting of the tensor's data to match the specifications of the target WAV file. Whether using `scipy.io.wavfile` or `soundfile`, understanding the interplay between the numerical representation in the tensor and the data layout requirements of the WAV format is critical. Choosing the correct bit depth (16, 24, etc.) is also a parameter that is set at the time of file storage.

For further exploration, consider consulting documentation on the following: the RIFF file format specifications, and the API documentation for `scipy.io.wavfile`, and `soundfile`. Further research on digital audio signal processing and data types in general can also prove beneficial. Additionally, looking into the `torchaudio` library, while it does not offer the same low-level control as the previously mentioned libraries, may also be instructive for higher-level audio data manipulation.
