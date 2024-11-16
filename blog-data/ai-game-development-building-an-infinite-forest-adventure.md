---
title: "AI Game Development: Building an Infinite Forest Adventure"
date: "2024-11-16"
id: "ai-game-development-building-an-infinite-forest-adventure"
---

dude you are NOT gonna believe this generative ai project i just watched it's bananas

so basically this jeff fellow built a whole game—a little forest adventure thing—where *everything* is ai-generated  we're talking scenes characters descriptions the whole shebang  the goal is to find your way home before you run out of courage and vigor  it's simple but the execution? mind-blowing

the setup was pretty straightforward  he wanted a game with infinite replayability each playthrough totally unique  no two games alike  that's the dream right?  

so first he needed the scenes the guts of the game  he started with openai's completion endpoint  you know the deal  you give it a prompt it spits out text  his prompt? a monster  it was super detailed basically dictated the json structure he wanted  plus all the scene details what happens when you first visit a location how it affects your stats all that jazz  this is the kind of prompt that makes you appreciate a good prompt engineer:

```json
{
  "sceneId": 1,
  "description": "You stumble upon a clearing bathed in the soft glow of twilight.  A gentle stream meanders through tall grass.  A weathered wooden sign reads 'Beware the Whispering Woods'",
  "firstVisitEffect": {
    "vigor": -5,
    "courage": 10
  },
  "repeatVisitEffect": {
    "vigor": -2,
    "courage": 5
  }
}
```

and this is where things get REALLY interesting  the prompt was a behemoth  it worked okay but it was clunky and expensive  so he fine-tuned a model  this is where you take existing data—in this case 50 examples of his scene json—and train an ai model on it  the magic?  he drastically simplified the prompt for the fine-tuned model  took out all the json structure details  just gave it descriptive text  he basically bet that the model would learn the structure from the training data  and guess what? it worked like a charm  a dollar or two later he had a way more efficient and reliable system

```python
# simplified prompt for fine-tuned model
prompt = "a dark and mysterious forest scene with a creepy old house in the distance the air is heavy with fog and the wind howls through the bare branches of the trees"

# hypothetical response from fine-tuned model (note: this isn't actual json)
response = """you approach an imposing house hidden in fog, its dark silhouette barely visible through the dense mist. the wind whips around you, carrying with it the scent of damp earth and decaying leaves. a sense of unease settles over you.
"""
```

that was step one  next up images  he used leonardo ai a tool that not only generates images but also lets you create custom image models  this is crucial for consistency  you don't want wildly different art styles  you want a cohesive look  and that's where the real challenge began


he used the scene descriptions as prompts for leonardo but it wasn't as simple as copy paste  training an image model requires a specific type of consistency  all images need similar elements like perspective and scale   and the content can vary  it's like walking a tightrope between consistent style and creative variety

```python
# hypothetical leonardo ai prompt
prompt = "a dark forest path winding through tall trees with strange glowing mushrooms nearby fog obscures the distance a lone figure sits in the distance gazing towards a mysterious light in the trees"

# we can't show a leonardo output here, obviously, it would be an image
```

he trained a couple of models before hitting the sweet spot he generated tons of images  and man they were good  they all had this distinct forest vibe  the same kind of path the same overall feel but each one was unique  it was a perfect blend of style and variation a testament to the power of fine-tuning and careful prompt engineering  

the final step?  putting it all together he built a simple server that handles the ai pipeline  it requests a scene from openai gets the json validates it sends the description to leonardo for image generation and then assembles everything for the game   it was like watching a well oiled machine  a really cool pipeline

but here's the kicker  generating a scene took 10-30 seconds way too long for a smooth gaming experience  so he implemented caching  he pre-generated a bunch of scenes  and as players use them he replenishes the cache  smart  efficient and it kept the game running smoothly

the game itself is simple  you start at a lamppost  wander the forest  your vigor and courage affect your speed and visibility  the scenes are crazy cool  and you never see the same one twice  it’s a beautiful loop


his closing thoughts were great too he mentioned higher resolution images (ai upscaling is the key there), more creative prompts (user-selected themes, weather conditions tied to location for immersive gameplay), and how this process could be used for other projects  it's a whole system that he could extend and improve even further  it's a fantastic example of generative ai in action

so yeah  it's pretty awesome  it's more than just a summary it's a peek into the mind of a creative coder tackling a complex problem with ai and coming out with something truly special  it's a story of clever design thoughtful implementation and a whole lot of fun  and maybe a touch of caffeination.  totally worth checking out
