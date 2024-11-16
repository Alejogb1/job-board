---
title: "Designing Usable AI Products: Structure & Familiarity"
date: "2024-11-16"
id: "designing-usable-ai-products-structure--familiarity"
---

dude so this ben hilac talk right totally blew my mind  it was all about building ai products that don't totally suck  like you know  the whole point wasn't just "ai is cool" it was "how do we make ai actually usable for normal people not just us geeky types"

first off the setup was pure genius he started by talking about "unbounded products"  these aren't your average apps stuck on a screen  think robotics spacex even the apple vision pro stuff  these things exist in the real world  and that changes everything  he even cracked a joke about software roaming the streets of san francisco and getting attacked by angry mobs  i nearly choked on my coffee

he laid out three main sections past present future  classic storytelling 101 right  the past was all about how products used to be super simple click and tap  but multi touch changed all that  suddenly we had smartphones and all this crazy interaction  then he transitioned to the present focusing on AI products

one key visual cue was this crazy star cluster visualization of early dawn prototypes  it was wild  it showed how far they'd come to something much more structured and user-friendly  later he talked about bad examples like that vercel chatbot demo with ephemeral ui elements vanishing mid conversation  it was like watching a house of cards fall apart before your eyes  another visual cue was the amazing before and after shots of the design changes  it's amazing how a simple layout change can drastically improve user experience


here's where things got technical and interesting he talked about two key design patterns for ai products

**1  structure structure structure** this wasn't just about making things pretty  it's about making sense of chaos  he used the example of dot a journal app  the way they structure their entries by date and user really makes sense it's intuitive  think of this code example which shows basic python  it's a simplified version but gets the idea across

```python
# simple journal entry structure
class JournalEntry:
    def __init__(self, date, text, user):
        self.date = date
        self.text = text
        self.user = user

entries = [
    JournalEntry("2024-10-27", "had a great coffee today", "ben"),
    JournalEntry("2024-10-27", "worked on a new ai project", "sarah"),
    JournalEntry("2024-10-28", "finished that ai project", "sarah")
]

# display entries for a given user
def show_user_entries(user):
    for entry in entries:
        if entry.user == user:
            print(f"{entry.date}: {entry.text}")

show_user_entries("sarah")

```

this simple structure makes it easy to manage and display entries by user and date which is like dot but in code


**2  familiarity matters**  he hammered home that users love familiar things  don't make them learn a whole new interaction paradigm  make your ai product feel like something they already know  like a spreadsheet or a search engine  this is crucial  the apple tv app in vision os  similar to the tvos one  he emphasized that that wasn't laziness its about usability

another code example  this time a little js to show how familiar interactions can be implemented even in an ai context  imagine a simple slider to adjust something within an ai response  this is cleaner than text based adjustments

```javascript
// simple slider to adjust AI response parameter
const slider = document.getElementById("mySlider");
const output = document.getElementById("output");
output.innerHTML = slider.value; // Display the default slider value

slider.oninput = function() {
  output.innerHTML = this.value;
  // here you'd make an api call to your ai model
  // with the updated slider value.  imagine sending 'confidence' level
  // to refine the ai's response
  fetch('/updateAI',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({confidence:this.value})
  }).then(response => response.json())
    .then(data => {
        //update the ui with the new ai response
    })
}
```

then you got the present section  it was a whirlwind tour of good and bad ai product design  he showed examples of perplexity which he said structured it's responses like a search engine  he even called out a vercel chatbot he didn't like because its ui was all over the place  its a key takeaway  structure your ai outputs like a proper program not just a freeform chat


the resolution was pretty clear  it's all about bringing structure and familiarity to ai  he even suggested using "ranked presets" to make interactions even smoother  more presets more options to cater to individual user needs


finally  he threw in this amazing idea of sparse autoencoders for generating a massive number of presets  it's a concept  think of it as being able to infinitely refine ai behavior like fine tuning millions of parameters without the prompt engineering headache  this really is future thinking  here's a very conceptual code example  it's not runnable but shows the essence


```python
# conceptual sparse autoencoder for ai preset generation (not runnable)
# imagine a vast latent space representing all possible ai behaviors
# we use sparse autoencoders to discover meaningful dimensions
# then we generate presets along those axes

class SparseAutoencoder:
  # ... complex internal structure to learn and encode/decode from latent space ...
  def generate_presets(self, num_presets, latent_space_dimensions):
    # ... generate presets along latent space axes ...
    # ... each preset is a set of weights to tune ai parameters
    presets = []
    for i in range(num_presets):
      # ... generate a unique point in latent space, decode to get ai parameter values ...
      presets.append(self.decode(generate_random_latent_vector(latent_space_dimensions)))
    return presets
```

basically  the talk was a masterclass in usability  he showed how to move beyond simple "chatbots" into creating truly useful and intuitive ai products  it's not just about creating ai it's about creating good design  and the secret sauce was structured familiarity and a user-centric approach  totally worth watching again and again  this ben hilac guy knows his stuff
