---
title: "How to Empower Learners with Multimodal AI"
date: "2024-11-16"
id: "how-to-empower-learners-with-multimodal-ai"
---

dude so steph's talk at the ai engineering summit was like totally mind-blowing  it was all about how multimodal ai is gonna change education forever and honestly it blew my mind  she started by laying down the harsh reality like only 70% of ten-year-olds can actually grasp a simple story  and covid just made things way worse  a huge learning gap opened up especially in math and reading  and it's not just kids adults need serious upskilling too  so that's the setup  a total wake-up call for the ed-tech world

the whole thing was super relatable  she mentioned how like 70% of generative ai users are gen z  so kids are already diving headfirst into this stuff  another visual cue was a graph showing this huge statistic which really hammered home how deeply integrated ai is becoming in young lives  and she even showed a cute little video of kids using her platform cognates  that was probably my favorite part  seeing those kids  just having fun programming  it was super inspiring

one key idea was ai literacy  it's not just about using ai tools it's about understanding how they work  steph showed how kids who used cognates became way more skeptical of ai's "smarts"  initially they thought ai was super smart but after learning how to train models themselves they realized it wasn't magic it was all about data and algorithms  it was a total shift in perspective from awestruck to critically curious

another killer concept was  co-creating with ai  she talked about designing an ai friend that helps kids and parents learn to code together  not just giving answers but suggesting ideas providing prompts  helping them brainstorm  think of it as a coding partner a helpful sidekick guiding them through tricky parts  she showed some really cool quotes from parents and kids raving about this collaborative learning approach  it really changed how they approached programming  

the resolution was a powerful call to action  she emphasized ai tinkering for everyone  not just passively using ai but actively engaging with it understanding its limitations and exploring its possibilities  she showed some pretty awesome demos using google's gemini api  like  drawing a scale with weights and the ai predicting the outcome  it was insane  then there was another one about solving math problems where the ai acts as a tutor offering step by step guidance  that blew my mind even more

here are some code snippets that connect with the ideas she presented  the first one is a simple scratch-like example of how kids might learn to program a robot to play hide and seek

```python
# simple hide and seek game logic in python

import random

robot_position = random.randint(1,10) # robot's random starting position
player_position = 0 # player starts at 0

while player_position != robot_position:
  player_position = int(input("enter your position (1-10): ")) # player's move
  if player_position < robot_position:
    print("robot is further away")
  elif player_position > robot_position:
    print("robot is closer")
  else:
    print("robot found you!")

# expand this to include robot movement , obstacle avoidance, etc
```

this is a super basic example but it shows how you can get kids engaged in basic programming concepts  making it a game makes it way more fun and less intimidating

next up let's look at a more advanced example incorporating custom image classification using a library like tensorflow  this ties into steph's discussion on training custom models  imagine the kids using their own images of unicorns and narwhals

```python
# basic image classification in tensorflow/keras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define image data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# generate training and testing data
training_set = train_datagen.flow_from_directory('training_data', target_size=(64, 64), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('test_data', target_size=(64, 64), batch_size=32, class_mode='categorical')


# define and compile the model (a simple CNN)
model = keras.models.Sequential([
  keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(2, activation='softmax') # 2 classes: unicorn, narwhal
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# train the model
model.fit(training_set, epochs=10, validation_data=test_set)


# evaluate the model
loss, accuracy = model.evaluate(test_set)
print(f"Accuracy: {accuracy}")

#make predictions on new images
```

this is a really simplified example  a real project would involve much more data preprocessing and potentially a more sophisticated model architecture  but it shows how kids can start exploring concepts of model training prediction and accuracy

finally  let's look at a tiny bit of code that shows how an ai could provide hints in a coding environment

```python
# simple ai hint generation for a coding project

user_code = """
  def my_function():
    #incomplete code block
  """

ai_hints = {
  "incomplete code block": ["try adding a 'return' statement"," consider using a for loop", "maybe you need an if/else statement"]
}

problem_area =  find_problem(user_code) # a hypothetical function to identify code issues

if problem_area in ai_hints:
  print ("ai hint: ", random.choice(ai_hints[problem_area]))
else:
  print("no hints available")


# more sophisticated versions can use language models to analyze code and generate more tailored hints
```

this is incredibly simplified  but it shows how an ai could analyze code and provide relevant hints  imagine having an ai buddy that gives you suggestions as you code that’s  like having a personal tutor right there in your code editor

so yeah  steph's talk was a real eye-opener  it's not just about ai in education it’s about empowering the next generation to become ai literate and creators  not just consumers  it's about making ai accessible fun and engaging for everyone  from little kids to adults  she totally nailed it
