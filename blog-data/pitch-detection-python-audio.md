---
title: "pitch detection python audio?"
date: "2024-12-13"
id: "pitch-detection-python-audio"
---

Alright so pitch detection in Python audio you say huh been there done that got the t-shirt and probably a few extra gray hairs to boot Let me tell you this is not as straightforward as it might seem at first blush especially if you’re trying to get anything resembling accuracy and robustness That’s not to say it’s rocket science but it does require a bit of understanding of signal processing and the quirks of audio

I’ve wrestled with this problem a lot I recall this one time back in my college days I was working on this interactive music project think something akin to a really janky version of Guitar Hero and the pitch detection aspect nearly brought me to tears Seriously it was a nightmare of false positives and wildly inaccurate readings I tried everything from basic autocorrelation to some pretty convoluted fast Fourier transform FFT implementations I was even briefly considering using those weird hardware pitch trackers that some guitar effects pedal people use But my budget was limited back then and my stubbornness kept me going eventually leading me down a more pure software-based path

Let's talk code though because that’s where the rubber meets the road in my experience Starting with something simple like the autocorrelation method is not always the best but it’s an ok place to start especially for very clean signals Here’s a basic implementation that uses NumPy:

```python
import numpy as np

def autocorr_pitch_detection(audio_signal, sample_rate, min_freq=50, max_freq=1000):
    audio_signal = np.asarray(audio_signal)
    n = len(audio_signal)
    correlation = np.correlate(audio_signal, audio_signal, mode='full')
    correlation = correlation[n-1:]
    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)
    sub_correlation = correlation[min_lag:max_lag]

    max_corr_index = np.argmax(sub_correlation)
    lag = min_lag + max_corr_index

    if lag == 0:
        return 0
    
    pitch = sample_rate / lag
    return pitch
```
This script computes the autocorrelation of a given audio signal and then locates the peak that corresponds to the pitch period it also takes min and max frequency as parameters to provide some sort of constraint and speed up the detection a bit. Its a start I will admit.

Autocorrelation is great for simple pure tones and relatively noise-free audio but it can fall apart rapidly with more complex sounds for example human voice with lots of harmonics or polyphonic music The biggest issue is it tends to latch onto the loudest periodic element in the signal not necessarily the fundamental pitch we want It’s kind of like trying to find a specific grain of sand on the beach when a whole bunch of them are trying to look like the same grain and that leads us to the next more robust method

The next method is the average magnitude difference function AMDF it is another time domain technique AMDF is less prone to latch on to harmonics than autocorrelation It’s essentially a variation that looks at the difference between the signal and a delayed version of itself to find periods

Here’s what an AMDF implementation might look like

```python
import numpy as np

def amdf_pitch_detection(audio_signal, sample_rate, min_freq=50, max_freq=1000):
    audio_signal = np.asarray(audio_signal)
    n = len(audio_signal)
    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)
    amdf_values = np.zeros(max_lag - min_lag)
    for lag in range(min_lag, max_lag):
      difference = np.abs(audio_signal[lag:] - audio_signal[:-lag])
      amdf_values[lag-min_lag] = np.sum(difference) / (n -lag)

    min_amdf_index = np.argmin(amdf_values)
    lag = min_lag + min_amdf_index

    if lag == 0:
      return 0

    pitch = sample_rate / lag
    return pitch
```
It is important to be aware that the way the difference is calculated is very important This implementation calculates the absolute difference and summs those values and the final result is a minimum not a maximum as seen in the autocorrelation example That is because in AMDF we are interested in detecting the lag with minimum difference not maximum correlation Also like the last method this one can be improved with some preprocessing of the signal which we will explore later down this response

Now for the gold standard at least in most situations FFTs which brings us to frequency domain techniques We use FFT to move our signal from the time domain to the frequency domain this means we will see what frequencies the audio signal is composed of This gives us more information to work with but also makes the detection more complex and resource intensive In this frequency domain we can then look for peaks that could correspond to pitches which usually the lowest one is our target

Here is a basic example that finds the frequency with the most power and returns it which might be ok for a basic test scenario

```python
import numpy as np
from numpy.fft import fft

def fft_pitch_detection(audio_signal, sample_rate, min_freq=50, max_freq=1000):
  audio_signal = np.asarray(audio_signal)
  n = len(audio_signal)

  fft_signal = np.abs(fft(audio_signal))
  freqs = np.fft.fftfreq(n, 1 / sample_rate)

  min_freq_index = np.argmin(np.abs(freqs - min_freq))
  max_freq_index = np.argmin(np.abs(freqs - max_freq))

  sub_fft_signal = fft_signal[min_freq_index:max_freq_index]
  max_freq_index_sub = np.argmax(sub_fft_signal)
  freq = freqs[min_freq_index+max_freq_index_sub]
  return np.abs(freq)
```
This method uses the FFT function to analyze frequencies in our audio. It’s crucial to remember that FFT will return complex numbers and so we need the magnitude using np.abs before locating the biggest peak of frequency This method is very dependent on the quality of the signal and might require some additional signal preprocessing before being used effectively as the noise can easily overshadow the actual fundamental frequency.

There are always more improvements that can be made to each of these detection methods like doing some data smoothing or calculating more robust peaks and frequencies but that is beyond the scope of this response

Now look while these methods are good for a starting point real-world audio data is often noisy and messy You're not usually dealing with a perfect sine wave you've got harmonics background noise reverberations all kinds of weirdness And that is why no single pitch detection algorithm is the silver bullet that will magically solve every problem There's a lot more to it than just picking one method and running with it

My experience has taught me a couple of things that can improve your results drastically First and foremost preprocessing is key Filtering can reduce noise and make the actual signal more prominent We can use a low pass filter to get rid of high frequencies that might be causing problems and or a high pass filter to remove very low frequencies that might be masking our desired results Also windowing can help isolate short segments of the signal for more accurate analysis Especially when dealing with complex audio or when the pitch changes across time Also using overlaps of samples can improve accuracy at the cost of more processing power

Another trick I’ve learned is that not all the available methods are created equal Some methods perform better in specific scenarios so a hybrid approach might be more beneficial you can also use a combination of methods as some methods are faster but less accurate so you might use them to get a general idea and use a more accurate but costly method to fine tune the result Also when you are designing a real system you might not need to calculate pitch for the entire audio sample you might divide it into segments and work on them independently or just when the user is speaking or playing music that is to say you should consider the context in which you are using your detection system to design a more efficient workflow

Also consider that pitch is not always a single number and the fundamental frequency is not always a perfect representation of how we perceive pitch There is a whole lot of research done in this area and it is important to be aware of that It all boils down to what you are trying to achieve at the end of the day

For further reading I'd highly recommend checking out books like "Fundamentals of Musical Acoustics" by Arthur H Benade or "Speech and Audio Signal Processing" by Ben Gold and Nelson Morgan for more of a hardcore dive into the mathematical foundations These are classics for a reason trust me On the signal processing front "Digital Signal Processing" by Alan V Oppenheim is also essential reading though it’s quite heavy on theory its still worth reading if you are trying to delve deeper into it You also have articles like "A comparative study of pitch detection algorithms" from Brown and Puckette This will give you a good overview of some of the other methods I have not covered here

And you know what's the biggest thing? Experimentation Don’t be afraid to tweak parameters try different approaches and see what works best for your particular situation The best advice I can give is that you have to be patient debugging audio code can be like trying to catch smoke with your hands

Oh one final thing I read somewhere that when the audio processing fails I should just blame the user for making all of the noise that helps with debugging ( just kidding )

Anyway good luck with your pitch detection project and remember if you get lost in the weeds you’re not alone we’ve all been there keep grinding and it will work out eventually
