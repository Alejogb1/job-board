---
title: "How can I preprocess data into tensors?"
date: "2024-12-15"
id: "how-can-i-preprocess-data-into-tensors"
---

alright, so you're looking at prepping data into tensors, right? been there, done that, got the t-shirt – probably several, stained with coffee from late nights debugging this exact thing. it's one of those foundational steps that can make or break your whole machine learning pipeline, and trust me, i've seen it break, spectacularly.

let's break it down like we're debugging some nasty code. the core issue here is that machine learning models, especially the deep learning ones, don't understand raw data. they crave numbers, organized into multi-dimensional arrays – tensors. whether it's text, images, audio, or tabular data, it all needs to be transformed into this format. you can't just throw a bunch of strings or jpegs at a neural network and expect magic to happen.

i remember back when i was first getting into this. i had this dataset of sensor readings from a prototype robot arm, just messy csv files full of timestamps and seemingly random numbers. i tried feeding it directly into a simple neural net, not knowing what i was doing, i expected some cool robotic feats. it was a mess. the loss function went bananas and crashed the computer, it was just a lesson in wasted compute power. it took me days to realize that data preprocessing was the missing link.

so, how do we actually do it? well, it really depends on the type of data you're dealing with. let’s talk about common scenarios.

**numerical data:**

this is arguably the simplest. if you have numerical data, like sensor readings, stock prices, etc. you typically need to do some scaling. one common approach is standardization, where you center the data around zero and scale it to unit variance. here's how you might do it using numpy:

```python
import numpy as np

def standardize_numerical_data(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    standardized_data = (data - data_mean) / data_std
    return standardized_data


#example data
numerical_data = np.array([[1.0, 2.0],
                          [1.5, 2.5],
                          [2.0, 3.0],
                          [2.5, 3.5]])
standardized = standardize_numerical_data(numerical_data)
print(standardized)
```

another common technique is min-max scaling, which squeezes your data to a range between zero and one. i've found this helpful when dealing with data that has a fixed or logical lower and upper bound. i used it heavily when handling image pixel intensities, for instance.

**text data:**

text is tricky. computers don't directly understand words like "hello" or "banana". we need to turn it into numerical representation. a common approach is to use tokenization and one-hot encoding.

tokenization splits the text into individual words or sub-word units. then, one-hot encoding creates a vector for each word, where one element is 1 (or higher value based on occurrence counts) and all others are zero.

```python
import numpy as np

def create_vocabulary(texts):
    unique_words = set()
    for text in texts:
       for word in text.split():
          unique_words.add(word)
    return list(unique_words)

def text_to_tensor(texts, vocab):
  tensor_list=[]
  for text in texts:
    vector = np.zeros(len(vocab))
    for word in text.split():
      try:
        idx = vocab.index(word)
        vector[idx] = 1
      except:
        pass #word not found in voc, can also throw an warning
    tensor_list.append(vector)
  return np.array(tensor_list)

# example text
texts = ["this is a sample sentence",
         "another sample this is"]
vocab = create_vocabulary(texts)
tensor_data = text_to_tensor(texts,vocab)
print(tensor_data)
```

this approach can lead to high-dimensional sparse vectors, specially if your vocabulary grows huge. things like word embeddings(like word2vec or glove) create dense vector representations that capture some semantic meaning. embeddings often result in much more compact tensors. and if you're dealing with sequences of text, padding the tensors with zeros to make all sequences the same length, is a common practice when you start working with recurrent neural nets. i once spent a week debugging a text classification model only to realize the cause was just different sentence lengths creating incompatible input shapes. it was infuriating. i ended up writing a custom preprocessor that handles this automatically, lesson learned.

**image data:**

images are often represented as multi-dimensional arrays, each pixel having red, green, and blue values. typically images need to be resized to a uniform size and normalized. normalization often involves scaling pixel values to a 0-1 range by dividing by 255. other things like data augmentation (rotating, scaling, and shifting the image during training), is also used to increase data diversity and improve model robustness.

```python
from PIL import Image
import numpy as np

def image_to_tensor(image_paths, target_size):
    tensors = []
    for image_path in image_paths:
      try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        tensors.append(img_array)
      except FileNotFoundError as e:
          print(f"error processing file {image_path}: {e}")
          continue
    return np.array(tensors)


#example paths to image files.
image_paths = ["image1.jpg", "image2.jpg"]
# dummy images for the example, create image files locally
img1 = Image.new('RGB', (50, 50), color = 'red')
img1.save("image1.jpg")
img2 = Image.new('RGB', (50, 50), color = 'blue')
img2.save("image2.jpg")

target_size = (64,64)
image_tensors = image_to_tensor(image_paths,target_size)
print(image_tensors.shape)
```
**general approach:**

the idea in all these cases is the same, transform the raw data into numerical representation and then structure it into an array that your model can consume. a good approach is to start simple, get your basic tensor preprocessing working, and then gradually add more complex techniques, as you see fit. don't prematurely optimize if you don't have to. that’s what i’ve learned. once you get the data in the form of tensors, you often have to shuffle your data set before training. and sometimes, data needs to be pre-splitted into train, validation, and test sets, this helps avoid overfitting and gives a more accurate measure of the model performance. one more tip, always plot the histograms and distributions of your data before you actually start training and after processing to get a grasp on what is actually happening. i once missed an outlier which was creating a total mess on my training until i saw it in the histogram. it is a good practice.

where to learn more? i've found that the book 'deep learning' by goodfellow, bengio, and courville is a fantastic resource for understanding the math behind a lot of these techniques. also, the 'hands-on machine learning with scikit-learn, keras & tensorflow' by aurelien geron is a practical book and really helps in implementing these things. for text specific pre-processing, i would recommend 'natural language processing with python' by bird, klein, and loper, it's a gold mine of information.

data pre-processing into tensors isn't glamorous work, but its an absolutely essential step in any machine learning task, you can think of it as the foundation to any data-driven model. don't skimp on it, take your time, understand what each step is doing, and you'll save a lot of debugging headaches later on. its like setting your car's tire pressure right, ignore it and the ride is bumpy. the tires will even fall out. and remember the first rule of machine learning... garbage in, garbage out. you can have the most elegant models but if your input tensors are poorly processed, you will get junk results.
