---
title: "How can NotebookLM assist in high-fantasy worldbuilding?"
date: "2024-12-03"
id: "how-can-notebooklm-assist-in-high-fantasy-worldbuilding"
---

Okay so you want to use NotebookLM for worldbuilding right super cool idea I've been messing around with LLMs for this kind of thing lately its pretty awesome  I'm thinking high fantasy so lots of magic mythical creatures crazy geography  NotebookLM is perfect for this because it lets you build on your ideas iteratively  you know like a real notebook you keep adding to and refining  unlike some other models that just give you a one-shot thing  

The way I'd approach this is to start with a core concept  maybe a specific magical system or a key character or even just a feeling you want the world to evoke  say you're going for a dark and gritty high fantasy world with a focus on necromantic magic then you'd start with something like this in NotebookLM


```python
# Initial world concept
world_concept = {
  "name": "Aethelgard",
  "tone": "dark and gritty",
  "magic_system": "necromancy",
  "key_characters": [],  # To be added later
  "geography": {
      "main_landmass": "The Mournlands",
      "climate": "cold and harsh",
      "resources": ["obsidian", "iron ore", "rare herbs"]
  },
  "notes": "Explore the ethical implications of necromancy, potential rebellions against necromancer rule"
}

# Add a character
world_concept["key_characters"].append({
  "name": "Lysandra",
  "role": "powerful necromancer queen",
  "motivation": "maintain control over Aethelgard",
  "flaws": "obsessive and paranoid"
})


print(world_concept)
```


So you see  it's all dictionary-based  super clean  easy to read and modify  you could even load this from a JSON file if you wanted to get really fancy  I’m thinking you could also easily integrate a version control system like git so you can keep track of all your changes which is super handy for collaborative worldbuilding or just to avoid accidentally deleting something crucial. The book "Version Control with Git" by Jon Loeliger is a great resource if you are not familiar with Git.  

Next you'd use NotebookLM's prompts to expand on this  maybe ask it to generate descriptions of locations within The Mournlands   This is where the iterative part is really important  you wouldn't just accept the first thing it gives you  you'd refine the prompts experiment with different phrasing  maybe even incorporate some of the responses back into the `world_concept` dictionary



```python
import openai #you will need your own key and install the library 

openai.api_key = "YOUR_API_KEY"


def generate_location_description(location_name, world_concept):
    prompt = f"""
    Describe a location called {location_name} in the world of {world_concept["name"]}.  
    Consider the overall tone of the world ({world_concept["tone"]}), the magic system ({world_concept["magic_system"]}), and the resources available ({world_concept["geography"]["resources"]}).
    The climate is {world_concept["geography"]["climate"]}.
    """
    response = openai.Completion.create(
        engine="text-davinci-003", # or other suitable model
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7, # Adjust for creativity vs. coherence
    )
    return response.choices[0].text.strip()

location_name = "Black Mire"
description = generate_location_description(location_name, world_concept)
world_concept["geography"][location_name] = description

print(world_concept)
```

This code snippet shows how easy it is to integrate the model  Its just a few lines to make a call to the OpenAI API and you get back a description You'll need to look up the OpenAI API documentation or perhaps check out their quickstart guide to get this running though  remember to get an API key


See how I added the generated description directly to the `world_concept` dictionary?  That’s how you build your world organically layer by layer   You can keep adding characters storylines plot points magical items anything you can think of using the same approach  And remember to experiment with different prompts  try being very specific sometimes  and other times be more vague and see what interesting results you get


For complex things like relationships between characters or the history of your world you might need a more structured approach   Maybe something like this



```python
#Character relationships
characters = {
  "Lysandra": {
      "relationships": {
          "Kael": {"type": "rival", "description": "Ambitious general who opposes Lysandra's rule"},
          "Seraphina": {"type": "advisor", "description": "A skilled necromancer, loyal but secretly plotting"},
          },
  },
  "Kael": {
      "relationships": {
          "Lysandra": {"type": "rival", "description": "He secretly wants to overthrow Lysandra"},
          "Seraphina": {"type": "ally", "description": "Shares his ambition"},
          },
  },
   "Seraphina": {
      "relationships": {
          "Lysandra": {"type": "advisor", "description":"She is manipulating Lysandra"},
          "Kael": {"type": "ally", "description":"She is helping him"},
          },
  },
}


print(characters)
```

This is again a dictionary-based structure it makes it easy to represent intricate relationships between multiple characters  I recommend looking at graph database concepts for managing large and complex relationships because this gets unwieldy quickly  "Graph Databases" by Ian Robinson is a good starting point


Remember  this is just a starting point  NotebookLM gives you the flexibility to really explore your worldbuilding ideas in a way that’s fun and intuitive  The key is to be iterative  experiment and don't be afraid to scrap things that aren't working  The beauty of this approach is you can constantly refine and restructure as your world evolves  It's like a living document that grows and changes with your imagination  Oh and don't forget to backup regularly  you don't want to lose all your hard work


For the technical side you might want to check out some papers on  prompt engineering for LLMs  there's a lot of research on how to craft effective prompts to get the best results from models like the ones in OpenAI’s API   Also looking into different LLMs and comparing their strengths and weaknesses might be useful depending on your needs  some excel at creative writing while others are better at factual information  Experiment and find the best fit for your style  And finally  if you want to go really deep  consider learning about graph databases and knowledge representation techniques to handle the complexity that will undoubtedly build up as your world grows  it'll help you keep things organized and avoid getting lost in your own creation   Good luck have fun building your fantasy world  let me know how it goes  I'd love to see what you come up with
