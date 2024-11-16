---
title: "Build an Age-Guessing Program with Machine Learning"
date: "2024-11-16"
id: "build-an-age-guessing-program-with-machine-learning"
---

dude you will *not* believe the video i just watched it was like a rollercoaster of code and hilarious explanations i'm still chuckling  it was all about building this super cool  program that basically guesses your age based on your face  i mean who *doesn't* want a robot that can tell how old you are right?  the whole thing is built on this idea of  machine learning which is basically teaching a computer to learn without you explicitly telling it everything which is way cooler than it sounds


so the setup was pretty straightforward the guy—let’s call him professor awesome because that’s totally what he was—starts by saying how much he hates typing his age every time he signs up for something online  i feel that bro  major first-world problem right there  then he drops this bomb: he’s gonna build a program to do it for him  and not just any program a *machine learning* program using python and a bunch of libraries i nearly choked on my coffee  


one of the first key moments was when he showed this huge dataset of faces  i’m talking thousands of pictures people of all ages  it was seriously impressive it looked like a hollywood casting call gone wild he was using a dataset called "the age dataset" which you can probably find on google datasets search if you're curious  that's where the magic happens the computer needs lots of examples to learn from  it's like showing a kid a million pictures of cats before you ask them to identify a cat—they'll get it eventually  


another super important bit was when he started talking about convolutional neural networks  cnns for short  think of it like this  imagine you're looking at a picture of your friend  you don’t look at every single pixel individually right you sort of see shapes and patterns your brain groups things together and that's exactly what a cnn does  it's like a super-powered image filter  it focuses on specific features like eyes noses mouths wrinkles all the things that indicate age  professor awesome even showed a diagram with layers and filters and stuff it was pretty wild  


here’s a tiny snippet of code that shows a basic cnn structure in python using tensorflow  it's simplified for clarity obviously a real one is way more complex


```python
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1) # Output layer for age prediction
])

model.compile(optimizer='adam', loss='mse') #mse is mean squared error the loss function

# training part omitted for brevity – this would involve loading and feeding the image data
```

basically  it's creating a model that takes an image as input and predicts a number which represents the age  the `conv2d` layers are the convolutional layers doing the pattern recognition  `maxpooling2d` reduces the image size while keeping important features  `flatten` converts the processed image into a 1d array for the dense layers and finally the dense layer predicts the age  i know it looks scary but trust me it's not that bad once you break it down


he also emphasized something called "training the model"  this part was visually represented as a graph showing the accuracy increasing over time it was like watching a plant grow its so satisfying to watch the accuracy go up  basically the computer is looking at all those faces and figuring out the relationship between facial features and age  it's like a really smart game of  "guess the age" but on a much larger scale   it wasn't perfect at first  it made some hilariously wrong guesses especially with pictures of babies – one time it guessed a baby was 87  i almost died laughing


another crucial point was model evaluation he talked about metrics like mean squared error  mse and r-squared  mse basically tells you how far off the model's predictions are on average and r-squared tells you how well the model fits the data  it's like grading your model on a test  the lower the mse the better  and the closer r-squared is to 1 the better the fit he showed these metrics visually  it was like looking at a report card for the computer and it got a pretty good grade


i'm not kidding there was a part where he messed up a piece of code and the whole thing crashed  it was so relatable i wanted to give him a high five  it reminded me of when i spent three hours debugging a simple for loop only to realize i misspelled a variable name  debugging is a programmers life blood  it’s like a game of where’s waldo  but waldo is a tiny typo in a million lines of code


here's another piece of code showing a simple function to calculate mse


```python
import numpy as np

def mse(y_true, y_predicted):
  """Calculates the Mean Squared Error between true and predicted values.

  Args:
    y_true: A numpy array of true values.
    y_predicted: A numpy array of predicted values.

  Returns:
    The Mean Squared Error (MSE).
  """

  return np.mean(np.square(y_true - y_predicted))


#example usage
true_ages = np.array([25,30,35,40])
predicted_ages = np.array([27,32,33,42])
error = mse(true_ages, predicted_ages)
print(f"The mean squared error is: {error}")


```

this function takes the true ages and the ages predicted by our model and calculates the average squared difference between them  you can use this to evaluate the accuracy of age prediction models  it's important to remember that lower mse means better accuracy because it indicates that the predictions are closer to the actual values



and finally the resolution  professor awesome’s age-guessing program wasn't perfect but it was pretty darn impressive  it showed how machine learning can be used to solve seemingly simple problems in a fun and interesting way  he even tried it on his own picture and the result was surprisingly accurate  he concluded by saying that this is just a small example of what’s possible with machine learning  it’s a field with endless possibilities  he even showed  a glimpse of how this could extend to other applications like medical diagnosis or security systems  


one last code example this one shows a simple function that just uses the model we created earlier to predict an age from a picture – note you'll need to preprocess your image to fit the input shape of the model


```python
import tensorflow as tf
from PIL import Image
import numpy as np

def predict_age(image_path, model):
  """Predicts the age from an image using a pre-trained model.

  Args:
      image_path: Path to the image file.
      model: The pre-trained age prediction model.

  Returns:
      The predicted age.
  """
  img = Image.open(image_path).resize((150,150))
  img_array = np.array(img) / 255.0 #normalize pixel values
  img_array = np.expand_dims(img_array, axis=0) #add batch dimension
  prediction = model.predict(img_array)
  return int(round(prediction[0][0]))

#example usage (assuming you have a model loaded as 'model')
predicted_age = predict_age("my_face.jpg", model)
print(f"Predicted age: {predicted_age}")
```

this shows how you'd use the model we defined previously to make a prediction given an image path  remember you’ll need to install the necessary libraries  like tensorflow and pillow `pip install tensorflow pillow`  the code handles resizing and normalization of the image  and returns the rounded predicted age


so yeah that was my wild ride through the world of age-guessing machine learning it was way more fun than i thought it would be  i highly recommend checking out videos like this if you want to get into machine learning   it’s not as scary as it sounds and honestly pretty darn cool
