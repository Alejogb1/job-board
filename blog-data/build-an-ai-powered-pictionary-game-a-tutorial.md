---
title: "Build an AI-Powered Pictionary Game: A Tutorial"
date: "2024-11-16"
id: "build-an-ai-powered-pictionary-game-a-tutorial"
---

dude you will not believe this video i just watched it's like the coolest thing ever this guy joseph built this game paint wtf and it's basically ai pictionary but on steroids  it uses openai's clip and gpt-3 and in its first week it had 120000 players seriously  the whole thing's a masterpiece of multimodal madness and i'm gonna break it down for you  get ready for some serious tech talk interspersed with some good old-fashioned crazy

so the setup is this  joseph wanted to make a game where people draw stuff based on prompts  but not just any prompts  gpt-3 generates these wacky prompts like "a giraffe in the arctic" or "a bumblebee that loves capitalism"  then users get a microsoft paint-like interface  it’s super basic just a canvas you know like the real deal  they draw their interpretation of the prompt hit submit and then the magic happens

the core of the magic is clip  clip is a model that can understand both images and text its job is to compare the user's drawing to the original gpt-3 prompt the drawing gets turned into an image embedding and the prompt becomes a text embedding clip then compares these embeddings using cosine similarity if the similarity score is high it means the drawing is a good match to the prompt simple right

but here's where it gets really interesting one of the visual cues i loved was this image of a raccoon driving a tractor  users submitted tons of versions some were pretty realistic others were super abstract and you could see how clip judged them based on how close they matched the prompt's description it's pretty wild how accurately it assessed those drawings you know this is not just about drawing skills it is about understanding the prompt and conveying it through your drawing

another key moment was when joseph talked about the "open-set understanding" of clip  it means clip isn't just matching against predefined categories like "cat" or "dog" it actually understands the *concept* of the prompt even when the drawings are wild and abstract that's the power of multimodal learning he showed an example of a bumblebee that loves capitalism that’s not your everyday prompt you see how abstract those drawings got  

now the resolution  the game went viral  because it's fun easy to understand and uses ai in a creative way people love seeing their drawings judged by an ai and competing on a leaderboard it's engaging  but also it showed some real-world limitations of clip like how users would sometimes add extra things to their drawings or how clip can't always understand abstract concepts perfectly   it’s a real-world example of building with large language models

let's get into the code snippets dude this is where the fun really starts joseph showed a python script using opencv and an open-source inference server called inference from roboflow the whole thing's ridiculously simple  first we've got some basic image processing this guy was working straight on his webcam

```python
import cv2
import inference

# Initialize the inference server for image processing
inference_server = inference.InferenceServer()

# Load a pre-trained model (this example uses rock-paper-scissors, but you can swap it out)
model = inference_server.load_model("rock-paper-scissors")

# Function to process a frame from the webcam
def process_frame(frame):
  # Perform object detection (or other inference tasks)
  results = model.predict(frame)
  # process the results
  return frame

# Open the default camera
cap = cv2.VideoCapture(0)
while True:
    # read frame
    ret, frame = cap.read()
    #process the frame
    frame = process_frame(frame)
    # show frame
    cv2.imshow('webcam', frame)
    #break if q pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

this is just basic webcam access and model loading  imagine how much more you can do with this setup its all about real time processing


then he loaded clip that's where the multimodal magic happens we are working with image and text embeddings

```python
from inference.models import clip

# Initialize CLIP
clip_model = clip.CLIP()

# GPT-3 prompt (example)
prompt = "a grumpy cat wearing a tiny hat"

# Get the text embedding
text_embedding = clip_model.embed_text(prompt)

# Get image embedding from a frame
image_embedding = clip_model.embed_image(frame)

# Calculate cosine similarity
similarity = cosine_similarity(text_embedding, image_embedding)

print(f"Similarity score: {similarity}")
```

here we're getting the embeddings for both text and image  simple and elegant

and finally the similarity calculation he even has a live demo with two volunteers drawing gorillas gardening with grapes in real-time

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assume text_embedding and image_embedding are already calculated (from previous code snippet)

# Calculate cosine similarity and adjust the range
similarity_score = cosine_similarity(text_embedding, image_embedding)
normalized_similarity = (similarity_score - similarity_score.min()) / (similarity_score.max() - similarity_score.min()) * 100


print(f"Similarity Score: {normalized_similarity:.2f}%")

```

this code calculates the similarity and converts it to a percentage for easy display  super simple but super effective  the whole thing is crazy impressive

so yeah that's paint wtf in a nutshell  it's not just a game it’s a demonstration of the power of multimodal ai the potential for creative applications is insane  and the code is surprisingly straightforward  it was seriously inspiring and made me want to build my own ai-powered thing immediately also it highlighted the challenges of working with user-generated content and how ai can help us moderate and improve these experiences   go check out the video yourself dude you’ll love it
