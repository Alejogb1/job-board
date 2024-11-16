---
title: "Building a Responsive AI Storyteller App"
date: "2024-11-16"
id: "building-a-responsive-ai-storyteller-app"
---

dude so i just saw this awesome talk about storyteller this app that makes little audio stories for preschoolers it's wild  the whole thing's built with typescript and this library the dude made called model fusion which is like a total boss move  basically it takes a voice recording  like you saying "benny saw something weird in the forest" and *bam* it spits out a whole two-minute story with sound effects and everything  

the setup's pretty straightforward the goal is to make this thing super responsive so kids don't get bored waiting right? it's a client-server app react on the front end fastify on the back end that's what makes everything super speedy. the speaker even showed a tiny little recording screen  just a button that says "record topic"  simple as can be. you hit it you talk and *whoosh* off it goes to the server  another visual cue that stood out was the way he showed the different stages updating the ui you know showing that things are happening even if it's not finished yet. he also showed this really cool loading screen that updates in real time showing the parts of the story that are generating, and a little loading bar that would move along as it went. the fact that you could start listening even as other parts are being generated is really clever.

one of the biggest challenges was speed  getting that story generated super fast  they talked a lot about responsiveness right?  like you don't want a kid waiting two minutes for a story that's supposed to be two minutes long that would just be a disaster!  the main idea behind their solution was all about parallelization and streaming so  they don't wait for one thing to finish before starting another  it's like an assembly line for stories


here's where it gets really interesting the whole process happens in stages  first openai whisper transcribes your voice input  this is crazy fast  like 1.5 seconds for a short phrase  then gpt-3 turbo instruct kicks in to create a story outline  think of it like a skeleton for the story  this takes around 4 seconds and that's it's the magic part


let's talk code for a second because this is where the real fun is  imagine how you might implement a simplified version of the transcription part you could use something like this:


```javascript
// simplified transcription using a mock API call
async function transcribeAudio(audioBuffer) {
  try {
    const response = await fetch('/api/transcribe', {  //mock api call, replace with actual endpoint
      method: 'POST',
      body: audioBuffer,
    });
    const data = await response.json();
    return data.transcript; // the transcribed text
  } catch (error) {
    console.error('Transcription failed:', error);
    return ''; // return empty string if something goes wrong
  }
}


const audioBlob = await fetch('/your-audio-file.wav').then(res => res.blob()); // Fetch audio from a source
const transcript = await transcribeAudio(audioBlob);
console.log("Transcription Result", transcript);

```

this code snippet uses a mock api call to a '/api/transcribe' endpoint. you’d replace this with your actual openai whisper setup naturally. The important part is the asynchronous nature of fetching the transcription and handling potential errors.

then the fun really begins it all happens in parallel!  gPt-3 turbo instruct makes the story outline another gpt-3 call creates the title stability ai's stable diffusion generates an image all at the same time!  the speaker emphasized the importance of consistency here using the whole story for the image generation prompt to make sure the image actually matches the story  this isn't a simple task; it involves some natural language processing (NLP) techniques.  


getting the image is pretty cool too:

```javascript
// simplified image generation using a mock API call
async function generateImage(story) {
  try {
    const response = await fetch('/api/generate-image', { // mock api call, replace with actual endpoint
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ story }),
    });
    const data = await response.json();
    return data.imageUrl; //url of the generated image
  } catch (error) {
    console.error('Image generation failed:', error);
    return ''; // return empty string if something goes wrong
  }
}


const story = "Benny saw something strange in the forest";
const imageUrl = await generateImage(story);
console.log("Generated Image URL:", imageUrl);
```
here we make a post request to a mock api `/api/generate-image` passing the story as a json object.  you'd replace the fetch call with your actual call to stability ai obviously. This illustrates how the process would handle creating the image in parallel, similar to the previous example.


but the real magic is in the audio generation the dude used gpt-4 to structure the story with dialogue and speakers  then it sends that structured information to a speech synthesizer (11 labs or similar) this is where it gets super interesting because this part is slow like a minute and a half slow  so they solved this using streaming! they don’t generate the whole audio at once instead they generate and send bits of audio as they’re ready this is what's called "streaming" and allows the user to start listening immediately   


here’s a slightly more conceptual example to show the streaming idea:

```python
# conceptual example of streaming audio generation
# this isn't runnable code but illustrates the idea

def generate_audio_stream(story_structure):
    for segment in story_structure:
        audio_chunk = generate_audio_for_segment(segment) #this is the part that calls the synthesis api
        yield audio_chunk # this yields each chunk, letting it be processed by the client before the next one is generated


#in your client, you would iterate over this generator:
for chunk in generate_audio_stream(story_structure):
    play_audio_chunk(chunk) # plays the audio chunk
```

this pseudo-code shows a generator function, `generate_audio_stream`, which yields chunks of audio one by one.  This simulates how the server sends data in pieces, enabling the client to begin playing even before the entire story is synthesized.

finally  for the voices they used gpt-3.5 turbo to describe each speaker (like "a grumpy old bear")  then they used these descriptions to find matching voices from a pre-embedded set  this is super clever because it saves a ton of time  imagine trying to synthesize voice for every character on the fly  that would be a nightmare.

the resolution is pretty clear they built a crazy fast responsive story generator for kids using parallelization streaming and clever prompt engineering and the use of various ai models. this whole thing isn't just some basic app—it's a masterclass in building highly efficient and interactive ai powered applications.  the guy totally nailed the challenge of responsiveness  and the whole thing sounds super fun.   you can check out his code on github if you are interested, i think it's github.com/lrl/storyteller and github.com/lrl/modelfusion  but honestly even just watching that talk was worth it  the whole thing is so well done that i could not recommend it more highly.  it's inspiring and funny and also pretty mind blowing.
