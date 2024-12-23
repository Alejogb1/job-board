---
title: "How can SDP telephone events be switched between 16000 and 8000?"
date: "2024-12-23"
id: "how-can-sdp-telephone-events-be-switched-between-16000-and-8000"
---

Alright,  Switching SDP (Session Description Protocol) telephone events between 16000 Hz and 8000 Hz sampling rates is something I've dealt with more than once, and it’s not always as straightforward as flipping a switch. The core challenge lies in ensuring that both the audio data itself and the associated SDP signaling are correctly modified to maintain call quality and compatibility. This involves adjusting the media format information within the SDP payload and potentially resampling the actual audio stream if it's being generated dynamically.

To begin, let's unpack what we're actually manipulating. SDP, as defined in RFC 4566, describes multimedia sessions, including audio. A crucial part of the SDP is the ‘m’ line, which specifies the media type (audio), transport port, and a list of *RTP payload types*. For telephone events, often termed *DTMF tones* or *in-band signaling*, we typically see these events being carried as RTP payload types associated with specific encoding parameters. The sampling rate is a fundamental parameter in these encoding definitions.

Now, if we assume a scenario where the original SDP advertises telephone events at 16000 Hz, and we need to switch to 8000 Hz, it’s critical that we handle this consistently across both sides of the call. The SDP, which describes the capabilities of each endpoint, needs to be updated first. This update ensures the receiving end correctly interprets the incoming data. We must adjust the parameters associated with the specific payload type that carries the telephone events. A common payload type used for this is dynamic payload type numbers in the range 96-127, and the *fmtp* attribute will be where much of the action takes place, in particular the *rate* and potentially the *ptime* and *maxptime* parameters.

Let me describe a scenario from my past. I worked on a legacy VoIP system that initially used 16000 Hz for all audio, including DTMF, which was using a dynamic payload type, lets say payload type 97, and advertised using *telephone-event/16000*. We then needed to integrate with a system that only supported 8000 Hz for DTMF signals. This required a change to the SDP, in addition to the audio resampling, which in turn required significant care.

Here's how we managed it, and the code examples will illustrate this further:

First, we would generate a *new SDP* for offering 8000Hz telephone events and the corresponding *answer SDP* would contain the necessary parameters. Let's assume payload type 97 for 16000 Hz had previously been defined like so:

```sdp
m=audio 1234 RTP/AVP 97
a=rtpmap:97 telephone-event/16000
a=fmtp:97 0-15
```

We would need to change the *rtpmap* line and potentially adjust the transport parameters of the new answer to:

```sdp
m=audio 1234 RTP/AVP 97
a=rtpmap:97 telephone-event/8000
a=fmtp:97 0-15
```

This change to the 'a=rtpmap' attribute defines the encoding parameter of that payload type, and would be applied to the *offer sdp* message going out and subsequently in the *answer sdp*. Note that the *fmtp* attribute remained the same as the range of codes it defines are independant of the rate.

However, just changing the SDP might not be enough. Depending on how your system processes the DTMF tones, you might need to actually resample the audio. Let me explain: Suppose your system encodes the telephone events as a sequence of audio samples at 16000 Hz, this will produce a very short sequence. When you transmit that sequence, with a change in the SDP, as if it was samples at 8000 Hz this will be replayed at half the speed and would therefore sound like very slow and low pitch DTMF tones at the receiving end.

To address this, the actual audio samples must be resampled from 16000 Hz to 8000 Hz before it's encapsulated into RTP packets and sent using the new SDP. The resampler must perform proper decimation to avoid aliasing. This part is crucial. There are plenty of resampling techniques available, and many audio processing libraries can do this for you efficiently. Here is an example in python (using the librosa library) that performs resampling of audio data before sending it out, which will illustrate the approach:

```python
import librosa
import numpy as np

def resample_dtmf(audio_data, original_rate, target_rate):
    """Resamples audio data containing DTMF tones from original to target rate.

    Args:
      audio_data: A numpy array of audio samples.
      original_rate: The sampling rate of the input audio data (e.g., 16000).
      target_rate: The desired sampling rate for the output audio data (e.g., 8000).

    Returns:
        A numpy array of resampled audio samples.
    """
    if original_rate == target_rate:
        return audio_data
    resampled_audio = librosa.resample(y=audio_data, orig_sr=original_rate, target_sr=target_rate)
    return resampled_audio

# Example usage:
original_audio_16k = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) # some fictional data.
resampled_audio_8k = resample_dtmf(original_audio_16k, 16000, 8000)
print(f"Original data: {original_audio_16k} Resampled data: {resampled_audio_8k}")
print(f"Original sample rate : 16000, Resampled sample rate : 8000")
```

The example above shows how the raw audio data is resampled, before sending via RTP. The resampled audio is what is packaged into RTP packets and sent over the network according to the *new* SDP description.

Finally, if you are not generating the audio from samples as the previous example described but rather you are generating the tones on the fly it's worth remembering that you should be generating those tones, from scratch, at the target sample rate for example a generator would generate a tone at 8000 Hz if that is what was needed. Here is an example that demonstrates that:

```python
import numpy as np
import math

def generate_dtmf_tone(digit, sample_rate, duration_seconds):
    """Generates a single DTMF tone of the specified digit.

    Args:
      digit: The DTMF digit (e.g., '1', '2', 'A').
      sample_rate: The sample rate in Hz (e.g., 8000).
      duration_seconds: The duration of the tone in seconds.

    Returns:
        A numpy array representing the audio samples of the tone.
    """

    frequencies = {
        '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
        '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
        '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
        '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633),
    }

    if digit not in frequencies:
        raise ValueError("Invalid DTMF digit.")

    low_frequency, high_frequency = frequencies[digit]
    num_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

    tone = 0.5 * (np.sin(2 * np.pi * low_frequency * t) + np.sin(2 * np.pi * high_frequency * t))
    return tone

# Example usage:
sample_rate = 8000
duration = 0.1  # seconds
tone_1 = generate_dtmf_tone('1', sample_rate, duration)
print(f"Generated a tone of digit '1', at sample rate {sample_rate}")
```

In essence, switching telephone events between 16000 Hz and 8000 Hz involves not only updating the SDP, but also potentially performing resampling of the underlying audio samples if necessary, ensuring consistent and correct behavior on both ends of a call.

For deeper understanding, I'd recommend "Voice over IP Technologies" by Mark A. Miller for a solid foundation on VoIP concepts, and RFC 4566, the official SDP specification, for detailed information on SDP syntax and semantics. "Digital Signal Processing" by Alan V. Oppenheim is a classic resource for understanding audio resampling techniques. Remember, careful planning and a thorough understanding of these core concepts will allow you to make the necessary modifications to switch between sample rates without impacting call quality or functionality. This is a practical problem I've seen numerous times in the field, and hopefully, these tips will serve you well.
