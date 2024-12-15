---
title: "How to Predict a single image with a pre-trained load_model in Keras and how to get the label?"
date: "2024-12-15"
id: "how-to-predict-a-single-image-with-a-pre-trained-loadmodel-in-keras-and-how-to-get-the-label"
---

alright, so you're looking at loading a keras model, feeding it a single image, and getting the predicted label. i've been there, done that, got the t-shirt (and probably a few debugging scars too). it sounds straightforward but there are a few places where you can stumble if you're not careful. i'll walk you through the process, share some things i learned the hard way, and offer some code examples.

first off, let's talk about the model itself. assuming you've got a keras model saved – either as a `.h5` file or using the savedmodel format – you'll use `keras.models.load_model()` to bring it back into memory. this is pretty standard, and hopefully, you’ve already got this part sorted. the important thing here is that the model you load is architecturally identical to the one you trained. if there are any differences in layers, or even their activation functions, things are gonna go south very quickly. i once spent a whole day debugging a model that wouldn't predict correctly and it turned out, i was loading the model from a different training run. it's a lesson i won't soon forget. double check your filenames and paths, seriously.

now, for the image preprocessing, this is where things tend to get a little messy. your model was trained with some specific input shape and scaling. if you feed it an image without doing the proper pre-processing, the prediction will be garbage, plain and simple. usually, you will need to resize your image to the same size the training images had, and you need to make sure you are using the same type of scaling you used during training (e.g., dividing by 255, or using some other normalization). if your original training set was 128x128 then the input should also be 128x128. if the channels are in the wrong format, say it's in `bgr` when it should be `rgb` then things are going to fail, also.

the common way people load images and do the resizing is using the `image` class from `keras.utils` or `tensorflow.keras.preprocessing` like this:

```python
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np


def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # batch dimension
    img_array = img_array / 255.0 #scaling
    return img_array
```
in this code block, i am using `load_img` to load the image, `img_to_array` to convert it into a numpy array and finally, i am adding a batch dimension because keras expects the input to have it. it's a common pitfall to forget this batch dimension and the model will choke on it. the dividing by 255 is because most image pixel values are between 0 and 255 and most models are trained using values between 0 and 1.

another tip, if the model was trained using image augmentations, don't apply these to your validation images. augmentation is there to help the model generalise better in training, but is not meant to be used in prediction.

now let's do the prediction part. once you have your preprocessed image, just use `model.predict()` function like this:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# load the model
model = load_model('path_to_your_model.h5')

def predict_image(image_path, target_size):
    preprocessed_image = preprocess_image(image_path, target_size)
    predictions = model.predict(preprocessed_image)
    return predictions
```
here, i'm simply passing the preprocessed image to the `predict` function. this function will return an array containing the probabilities (or logits if the model doesn't have a final activation).

finally, for getting the label, this depends a lot on how the model was structured and trained. if your model is a standard classification model with a softmax activation in the final layer, then `predictions` will be a vector where each element is the probability of the image belonging to the respective class. the class with the highest probability is your prediction. you can use `np.argmax()` to find the index of the highest probability, which corresponds to your predicted label.

```python
import numpy as np

def get_predicted_label(predictions, class_names):
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name, predicted_class_index


if __name__ == '__main__':
    image_path = "path/to/your/image.jpg"
    target_size = (128, 128)  # same as training
    class_names = ['cat', 'dog', 'bird'] # example, this is model dependant
    predictions = predict_image(image_path, target_size)
    predicted_label, predicted_class_index = get_predicted_label(predictions, class_names)
    print(f"predicted label: {predicted_label} with class index: {predicted_class_index}")

```

the above code i’ve wrapped into a `get_predicted_label` function, and i’m also assuming you have the `class_names` array in the same order as the output layer in your model. it's a critical step that you keep your mappings in order so you know what class each output index represents. i’ve seen it happen so many times where someones label names get messed up. usually when someone saves the class names and index in different files, then it's very easy to mess it up. i recommend that, you use the class names only as an indexable list, as a best practice.

in summary, here is the flow:
* load the model with `load_model`.
* load the image and preprocess it to be the same as the training images, this includes resizing and scaling.
* perform a prediction using `model.predict()`.
* use argmax to obtain the class index and if needed, obtain the label based on the class index.

a few things that you should watch out for, apart from all the things i've mentioned. keep in mind that the model might be expecting the input to be a single input or a batch. if you add the batch dimension to your input and the model is expecting a single input you are not going to have a good time. another point to keep in mind is the type of input it might be expecting for example, floating point or integer, make sure you match that. and one more thing, if your model does multiple predictions and you are expecting just one, you need to select the right output. also, if you have a complex pipeline, you might need to extract intermediate layers instead of using the output layers, this can help you debug if the model is predicting correctly in some parts, but not in the outputs.

for further reading, i recommend, 'deep learning with python' by francois chollet, this book gives excellent examples of working with keras. also, anything by andrew ng on coursera is a great source of information. i am also a fan of the papers 'imageNet classification with deep convolutional neural networks', the original paper for alexnet, to give you a historical context of deep neural networks used in image processing. i do not recommend blog posts or youtube videos as they are not that reliable. you need to go to the source or use a good book.

the most important advice i can give you is to always double-check and triple-check that your input image is pre-processed in exactly the same way the training images were. a tiny difference can lead to completely wrong predictions, so start with a few known images, those that you know the class. that's the fastest way to debug.

i hope that this detailed explanation helps. let me know if you have more questions.
