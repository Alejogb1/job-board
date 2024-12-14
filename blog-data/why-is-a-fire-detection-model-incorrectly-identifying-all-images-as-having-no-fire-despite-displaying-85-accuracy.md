---
title: "Why is a Fire detection model incorrectly identifying all images as having no fire despite displaying 85% accuracy?"
date: "2024-12-14"
id: "why-is-a-fire-detection-model-incorrectly-identifying-all-images-as-having-no-fire-despite-displaying-85-accuracy"
---

ah, i've been there, staring at a seemingly 'good' model that's completely useless in practice. let's unpack this fire detection fiasco. an 85% accuracy figure sounds impressive, but it’s totally misleading. it’s the classic case of accuracy paradox and class imbalance rearing their ugly heads. i've seen this happen countless times, and trust me, it's more common than one might think.

when i first started dabbling in computer vision, i remember working on a project to classify images of different types of flowers. the initial model, boasting a similar high accuracy, fell flat on its face when confronted with a garden of only, let’s say, tulips. turns out, my dataset was heavily skewed towards roses; most images were roses and the model learnt to guess ‘rose’ all the time. i spent a whole weekend debugging that thing and i learnt more from that mistake than all the tutorials i followed. it taught me the accuracy metrics were almost meaningless by themselves. 

so, here is the thing, in your fire detection scenario it's very likely that the model is just predicting "no fire" for *everything*. the 85% accuracy means it is correct 85% of the time. that is fine if you have a dataset that 85% of the images have no fire in them; if your dataset is 90% no fire and 10% fire and the model always predict "no fire" then it would be correct 90% of the time, that accuracy says very little about how good the model is at *actually* detecting fire. think of it this way. if you trained a model to predict if there were a bear inside your closet and you used a dataset where 99.9% of the images had not a bear, your model can achieve a 99.9% accuracy saying always 'no bear'. is it good? certainly not!

the problem is class imbalance. your "no fire" class probably heavily outweighs the "fire" class in your training dataset. your model has learned a bias towards the majority class, the "no fire" one. it has simply found the easiest route to maximizing accuracy, even if it is practically useless. i have been burnt by this one more than once myself, literally and figuratively speaking.

now, let's dive into how this looks like and how to tackle it. first, let's talk about the metrics that are more informative for imbalanced datasets, accuracy is just not the one here. you need things like precision, recall, f1-score, and roc-auc. these metrics consider both false positives and false negatives, giving a much better insight into how the model performs, particularly on the minority class (in this case, "fire"). for example: precision is the proportion of correct positive predictions; recall is the proportion of actual positives correctly predicted. the f1-score is the harmonic mean of the precision and recall. and ROC curves, let me tell you, are a lifesaver.

to get a grip of these metrics i highly recommend looking into the "pattern recognition and machine learning" by christopher bishop. the book has an excellent section on these, it will give you a strong mathematical background. also, i found the "hands-on machine learning with scikit-learn, keras & tensorflow" by aurélien géron also helpful, that one is more practical and has a chapter with these problems on imbalanced datasets.

here's a snippet in python with scikit-learn demonstrating the difference between accuracy and these other metrics. it assumes your model is already trained and ready for evaluation, in the example is using a random prediction just to showcase the concept, not to be used in real world models.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# example of real labels (0 = no fire, 1 = fire)
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

# example of model predictions (a model always predicting no fire)
y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"accuracy: {accuracy:.2f}")

# calculate precision, recall, and f1-score
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
print(f"precision: {precision:.2f}")
print(f"recall: {recall:.2f}")
print(f"f1-score: {f1:.2f}")


# calculate roc-auc score, note that y_pred should contain the probabilities rather than a single value, it needs to change if you want to use this function

# i have included this just for completeness sake

y_pred_prob = np.array([0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2]) # simulated probabilities, replace this with your model output
try:
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    print(f"roc-auc score: {roc_auc:.2f}")
except ValueError:
    print("roc-auc score: cannot be calculated because a class is present only in one label (y_true or y_pred)")
```

this script shows that even with a high accuracy, precision, recall and f1-score are equal to 0, indicating a terrible model. it's also very instructive to analyze the confusion matrix, you will find that you are getting lots of true negatives and no true positives at all.

next, we need to talk about how you fix it. there are several strategies.

1.  **data augmentation:** generate more images of "fire" by applying transformations like rotations, flips, changes in brightness and contrast to the existing "fire" samples. this is a basic but usually very useful strategy for dealing with imbalanced datasets.

    for instance, with the `imgaug` library it can be done like this. it is important to note that `imgaug` has some deprecated functions. the modern way to use it is using `augmenters` classes, instead of `imgaug.augmenters.SomeFunction` do `iaa.SomeFunction` like shown in the next example. note that `image` below should be a numpy array with the image you want to transform.
```python
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image

# load image
image = Image.open('fire_image.jpg')
image = np.array(image)


# define the augmentation sequence
augmentation_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-20, 20)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.ContrastNormalization((0.75, 1.5)),
])

# augment the image
augmented_image = augmentation_seq(image=image)

# the augmented_image can be saved to disk or used by your pipeline to feed data to your model

```

2.  **oversampling or undersampling:** in oversampling we duplicate examples from the minority class. in undersampling we remove examples from the majority class. the most popular oversampling technique is smote, synthetic minority over-sampling technique, it creates new samples by interpolating between existing ones.

    here is an example with `imblearn`:

```python
from imblearn.over_sampling import SMOTE
import numpy as np

# generate dummy data
x = np.array([[1, 2], [1, 1], [3, 1], [0, 0], [0, 1], [1, 0], [0, 2], [0, 3]])
y = np.array([1, 1, 1, 0, 0, 0, 0, 0])

# create smote instance
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)
print(x_resampled)
print(y_resampled)

# in this example you can see the original 3 examples of the class 1 became 8
# it is synthetic data, not the original examples
# y_resampled contains now 8 samples of class 1 and 5 samples of class 0
```

3.  **weighted loss function:** when calculating loss, you can give higher importance to the minority class. this makes the model focus more on correctly classifying the fire samples. this can be done in many deep learning frameworks like tensorflow or pytorch. this is, in my opinion, the most important strategy and should be the first one to try if you have enough data. for instance, in tensorflow:
```python
import tensorflow as tf

# example class weights, the "1" (fire) class has a bigger weight
class_weights = {
    0: 1.0,
    1: 5.0
}

def weighted_loss(class_weights):
    def loss(y_true, y_pred):
        weights = tf.reduce_sum(tf.gather(list(class_weights.values()), tf.cast(y_true, tf.int32)), axis=1)
        loss_values = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_true, tf.float32), logits=y_pred)
        return tf.reduce_mean(loss_values * weights)

    return loss

# replace 'model.compile' with this when training your model
# model.compile(optimizer='adam', loss=weighted_loss(class_weights), metrics=['accuracy'])
```

these three are among the most used techniques to deal with class imbalances. you should try the 3 and tune them to improve your results.

so, to summarize, your 85% accuracy is probably a case of class imbalance. evaluate your model with metrics beyond accuracy like precision, recall and f1-score, augment your data, oversample your minority class and use a weighted loss function and things should start looking better. these are my first-hand tips. i hope this helps you get your fire detector up and running and avoid future ‘accuracy paradox’ situations. it happened to me more than once, and it's something that just sticks with you. i hope it doesn't 'burn' you as much as it burned me. it was a joke, i promise i'm not funny! good luck with your model.
