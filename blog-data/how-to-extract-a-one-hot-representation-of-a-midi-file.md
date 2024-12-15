---
title: "How to extract a one hot representation of a MIDI file?"
date: "2024-12-15"
id: "how-to-extract-a-one-hot-representation-of-a-midi-file"
---

alright, so you're after a one-hot encoding of midi data, huh? i've been down this rabbit hole myself a few times. it's a pretty crucial step if you're planning on feeding that music data into any machine learning model. it's not usually straightforward, there are a few gotchas you need to be aware of, which i'll lay out based on my past experiences.

first off, let's talk about what a one-hot encoding actually is in this context. with midi, we're dealing with a sequence of events—notes starting and stopping, changes in velocity (how hard a note is struck), controller changes, and a lot more. we need to transform this sequential data into a format a machine learning algorithm can understand, hence the need for one-hot encoding, essentially creating a vector where a '1' indicates the presence of a particular event, and a '0' indicates its absence, think of it as a binary 'is it on or off' system for all potential midi events.

now, this gets a bit tricky because midi can represent quite a lot of things. we could choose to only encode note on/off events for a simplified case or get into a fuller representation including things like pitch bend, program change etc. let's assume we're aiming for a comprehensive approach.

so, what are we encoding? well, a basic approach might include at the very least:
* note pitch (ranging from 0 to 127)
* note velocity (ranging from 0 to 127)
* note on/off status (this could be a single boolean or split into two, which is cleaner)
* time step (we'll have to discretize this a little for practical purposes).

remember back in my student days when i was working with a generative music model? i tried to skip this step for some reason and things got super weird. the model just outputted chaotic sequences. it was only when i sat down and actually implemented a proper one-hot encoding did things become somewhat usable, so yeah, don't skip this step.

let's see some python code examples to get a handle on things. assuming you have `mido` installed (if not, `pip install mido` will sort it out), here is a basic way to convert a midi file into a sequence of note-on events, along with the pitch and time, which is then transformed into a one-hot encoding.

```python
import mido
import numpy as np

def midi_to_one_hot(midi_file, time_step=0.02):
    mid = mido.MidiFile(midi_file)
    messages = []
    time_accumulated = 0
    for track in mid.tracks:
        for msg in track:
            time_accumulated += msg.time
            if msg.type == 'note_on' or msg.type == 'note_off':
                messages.append([
                    time_accumulated,
                    msg.note,
                    msg.velocity if msg.type == 'note_on' else 0,
                    int(msg.type == 'note_on')
                ])

    time_steps = np.arange(0, max(x[0] for x in messages), time_step)
    vocabulary_size = 128 * 2 + 2  # note pitch and on/off plus start/end

    one_hot = np.zeros((len(time_steps), vocabulary_size))

    for time, note, velocity, on_off in messages:
      time_idx = np.argmin(np.abs(time_steps - time))
      one_hot[time_idx, note] = 1
      one_hot[time_idx, 128 + (note)] = velocity if on_off else 0 # velocity if on otherwise zero, it can be expanded later to include the off events
      one_hot[time_idx, 128 * 2 + on_off] = 1 # note on/off bit
    
    return one_hot, time_steps

midi_file_path = 'your_midi_file.mid' #put your midi file here
one_hot_matrix, time_array = midi_to_one_hot(midi_file_path)
print(f"shape of one hot matrix: {one_hot_matrix.shape}")
```

in this example, we are only encoding the note-on events and their parameters, and also separating the on off status, but it is simple to also encode the note off events, by extending the on/off parameter of the message array. the velocity for note-off messages is set to 0 in this example which simplifies things, you can later add velocity to the off notes too, but for simplicity we will stick to the current one.

a few points about that code block:

*   `mido` makes it easy to load and process midi files.
*   we're discretizing time using `time_step`, which affects the resolution. a smaller time step leads to more timesteps for the model to train on and more memory to use.
*   i've also included the on/off status in the encoding, which is important if you want to preserve timing and expressiveness.
*   remember, this encoding is still relatively basic, and it doesn't include all possible midi events, a lot more could be added (like pitch bend, program change, etc)

now, let's get a bit more ambitious. what if you want to encode chords and durations? it starts to get a bit trickier because we're moving beyond simple one-note events. to do this we can create custom "events" based on the on notes.

```python
import mido
import numpy as np

def midi_to_one_hot_chord(midi_file, time_step=0.02):
    mid = mido.MidiFile(midi_file)
    messages = []
    time_accumulated = 0
    note_on_dict = {}

    for track in mid.tracks:
        for msg in track:
            time_accumulated += msg.time
            if msg.type == 'note_on':
                if msg.note not in note_on_dict:
                    note_on_dict[msg.note] = [time_accumulated, msg.velocity]
            elif msg.type == 'note_off' and msg.note in note_on_dict:
                start_time, velocity = note_on_dict.pop(msg.note)
                messages.append([start_time, time_accumulated, msg.note, velocity])
                
    time_steps = np.arange(0, max([x[1] for x in messages]), time_step) #use the end time for discretization
    vocabulary_size = 128 * 2 + 128 # note pitch, duration, and velocity for all notes

    one_hot = np.zeros((len(time_steps), vocabulary_size))
    for start_time, end_time, note, velocity in messages:
        start_time_idx = np.argmin(np.abs(time_steps - start_time))
        end_time_idx = np.argmin(np.abs(time_steps - end_time))
        duration = end_time_idx - start_time_idx
        one_hot[start_time_idx, note] = 1
        one_hot[start_time_idx, 128 + (duration % 128)] = 1
        one_hot[start_time_idx, 128*2 + velocity] = 1

    return one_hot, time_steps
midi_file_path = 'your_midi_file.mid' #replace this
one_hot_matrix_chord, time_array_chord = midi_to_one_hot_chord(midi_file_path)
print(f"shape of one hot matrix with chord encoding: {one_hot_matrix_chord.shape}")
```

here's what's changed:
* i now store the on note events and their timestamps. then, when an off note event arrives for a corresponding note, i calculate the duration based on the difference between the timestamps.
* we are now also encoding the duration of the note by getting the difference between the starting and end timestamp indexes and mod it by 128.
* it's now creating one-hot vectors for start time, note, duration and velocity, which gives you a richer representation than just single notes.

we're still working with discrete time steps, so there's always a trade-off between resolution and computational complexity here. but, this is a good example of how you can expand the encoding to get more intricate music information.

one of the biggest challenges i had was managing variable-length sequences. midi files can be different lengths, and you might need a model that can handle this. you've got a couple of routes you can take: padding with zeros to create sequences of equal length or using recurrent neural networks with masking techniques. the padded sequence approach is usually the simplest way to start, if the model is not recurrent (in case it is, the masking could provide an additional enhancement and not make the model learn 'false data'). another method is to cut all midi files to the same length by either truncating or zero-padding them and to just use the midi file data until a certain fixed limit.

so, now that we've gone through the encoding process itself, let's talk about the data. the quality of your encoded data will directly impact your model's performance. things to watch out for:
*   ensure your midi files are well-formed and contain what you expect them to, a common mistake is that some midi files do not contain any notes and then the process will break if not managed properly.
*   normalize your data if needed – sometimes you need to scale your velocity values or apply other transforms before they go into the model.
*   decide if you want to include or exclude metadata events and other midi specific messages, it will heavily depend on what you intend to achieve, for a simple note generative model you do not need other specific messages, but if you want to generate midi as close to the input as possible it is needed.
* and another very important step, inspect your encoded data to make sure it looks okay. if you are having issues it's easier to find it if you perform some sort of data inspection.

finally, a little extra, a basic midi file event visualizer which creates the one hot vector in the process, it has the code from the two previous examples (using the simple note on/off encoding method). this is how i usually debugged problems on my initial models, a simple visualization will tell you a lot of potential issues:

```python
import mido
import numpy as np
import matplotlib.pyplot as plt

def midi_to_one_hot_visual(midi_file, time_step=0.02):
    mid = mido.MidiFile(midi_file)
    messages = []
    time_accumulated = 0
    for track in mid.tracks:
        for msg in track:
            time_accumulated += msg.time
            if msg.type == 'note_on' or msg.type == 'note_off':
                messages.append([
                    time_accumulated,
                    msg.note,
                    msg.velocity if msg.type == 'note_on' else 0,
                    int(msg.type == 'note_on')
                ])

    time_steps = np.arange(0, max(x[0] for x in messages), time_step)
    vocabulary_size = 128 * 2 + 2  # note pitch and on/off plus start/end

    one_hot = np.zeros((len(time_steps), vocabulary_size))

    for time, note, velocity, on_off in messages:
      time_idx = np.argmin(np.abs(time_steps - time))
      one_hot[time_idx, note] = 1
      one_hot[time_idx, 128 + (note)] = velocity if on_off else 0
      one_hot[time_idx, 128 * 2 + on_off] = 1

    plt.figure(figsize=(15, 5))
    plt.imshow(one_hot.T, aspect='auto', cmap='viridis', origin='lower')
    plt.xlabel('time step')
    plt.ylabel('one hot vector index (note/velocity/on/off)')
    plt.title('one hot midi visualization')
    plt.colorbar()
    plt.show()
    
    return one_hot, time_steps

midi_file_path = 'your_midi_file.mid' #replace this
one_hot_matrix, time_array = midi_to_one_hot_visual(midi_file_path)
print(f"shape of one hot matrix: {one_hot_matrix.shape}")
```

the visualization should give you a way to check if the encoding is correct, and you can even start adding the chords to this visualization.

as far as resources go, i'd recommend looking into the midi specification documents directly. they can be a bit dry, but it is the ultimate source of truth. books like "fundamentals of music processing" edited by meinard müller can give you a good understanding of music processing techniques. papers on sequence-to-sequence models and recurrent neural networks can also help you understand how to use this encoded data properly, you can search for those on google scholar or specific research paper indexes.

i hope this extended explanation has helped, i know it's a bit of information, but believe me, a lot of things can go wrong with this process, and going slowly at each step is crucial to achieve a working pipeline. let me know if you have other questions, i've been there and done that, probably a lot more times than i'd like to *play back*.
