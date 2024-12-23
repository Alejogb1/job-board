---
title: "What steps are involved in using AI tools like Claude and Suno to create jingles for marketing campaigns?"
date: "2024-12-03"
id: "what-steps-are-involved-in-using-ai-tools-like-claude-and-suno-to-create-jingles-for-marketing-campaigns"
---

 so you wanna make jingles with AI right super cool  I've been playing around with Claude and Suno lately and it's a wild ride let me tell you

First things first you gotta have a solid idea  not just any jingle will do you need something catchy memorable something that sticks in people's heads like that one annoying song you can't get out of your head yeah that one  Think about your target audience what vibe are you going for are we talking upbeat pop or something more mellow and chill  You know the whole branding thing 

Once you've got that killer concept you can start feeding it to the AI  I usually start with a simple prompt like "create a short catchy jingle for a new eco-friendly cleaning product target audience is millennials who care about sustainability musical style upbeat pop"  See I'm giving it all the details  the more information you give it the better results you'll get it's like training a puppy you gotta be specific

Claude is amazing for the lyrical parts  it's really good at generating creative text  Suno on the other hand is all about the music  it can generate melodies harmonies rhythms pretty much anything you can imagine musically  I've had some great success pairing them together Claude gets the words flowing Suno makes them sing


Here's where it gets fun the iterative process  you're gonna get some output that's awesome some that's  and some that's like what even was that  Don't worry it's all part of the process  You gotta tweak your prompts experiment with different keywords styles  It's a bit like sculpting you're chiseling away at the rough draft until you get something you love

Example 1 - Claude for Lyrics

```python
# Hypothetical Claude interaction  not real code just showing the interaction style
prompt = "write lyrics for a jingle about a new sustainable coffee brand targeting young adults lyrics should be upbeat and focus on ethical sourcing"
claude_response = """
(Verse 1)
Sunrise brew ethically true
Beans grown with love for me and you
Sustainable sips happy hips
Taste the goodness earth gives

(Chorus)
[Brand Name] coffee so right
Ethical choice morning bright
Wake up and thrive feeling alive
[Brand Name] coffee oh so alive
"""
```

You can see here I'm giving Claude a very descriptive prompt  It knows the target audience the vibe even the kind of imagery I want in the lyrics.  This is key to getting good results from these AI models. For deeper understanding on prompt engineering you could check out a book on natural language processing (NLP)  many good ones cover prompt design strategies  look into the work on large language models by researchers like Bengio and Hinton their papers will be invaluable

Next  I'd take those lyrics and feed them to Suno  

Example 2 - Suno for Melody (Conceptual)

This part isn't really code  Suno works more like a web app or a plugin you give it the lyrics and some parameters like tempo key and style and it generates a melody I can't easily show you the code of Suno's internal workings because I don't have access  it's proprietary stuff  but I can give you a pseudocode representation of the process

```python
# Conceptual representation of Suno's functionality not actual code
lyrics = claude_response # from previous example
parameters = {
  "tempo": 120,
  "key": "C Major",
  "style": "Upbeat Pop",
  "instruments": ["acoustic guitar", "ukulele"]
}

# Suno's internal processes (not accessible)
melody = suno_generate_melody(lyrics, parameters)


```

See I'm setting parameters  tempo key style and even the instruments  These parameters heavily influence the output and refine it more  Think of it as directing a band  you need to specify the tempo how fast or slow they should play  the key the overall tone and feel  and of course the instruments are essential


Finally  you'll want to combine the output. Now this is where things get REALLY interesting because you'll need some audio editing software like Audacity GarageBand or even professional DAWs like Logic Pro or Ableton Live. I am not going to go into detail on audio editing this is another whole rabbit hole  it's about arranging mixing mastering that stuff  but it's essential in getting a professional level jingle.

Example 3 - Combining and Refining (Conceptual)

Again no real code here this is just a flow  you'd be using audio editing software


```python
#Conceptual representation  not executable code
lyrics = claude_response
melody = suno_generate_melody(lyrics, parameters) # from previous example

# Audio editing software workflow
combined_track = combine_audio(lyrics, melody) # combining lyrics and melody
refined_track = add_effects(combined_track) # adding reverb echo other effects
mastered_track = master_audio(refined_track) # final polishing volume etc

# Export to a standard audio format like WAV or MP3
export_audio(mastered_track, "final_jingle.wav")

```

So basically  it's a back and forth until you're happy  you might need to go back to Claude for lyric adjustments then back to Suno for melody tweaks and then back to your audio editor for final polish  Think of this process as agile development but for audio  it's iterative and experimental

Remember to check for copyright and royalty free music if you use any samples or sounds  that's very important  there are tons of resources online offering royalty-free stuff  it's always good practice  to look into creative commons licenses and make sure you're using things legally

So there you have it  a glimpse into AI jingle creation  It's more than just typing prompts into a box  it's a creative process that requires experimentation iteration and a good understanding of music theory and audio engineering.  If you want to dig deeper look at books on audio engineering like "Mixing Secrets for the Small Studio" and for a broader view of AI music generation try searching for research papers on generative adversarial networks (GANs)  and  reinforcement learning in the context of music composition.  Happy jingling
