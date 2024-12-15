---
title: "How is fully supervised training performed from annotation results of weakly supervised training?"
date: "2024-12-15"
id: "how-is-fully-supervised-training-performed-from-annotation-results-of-weakly-supervised-training"
---

alright, let's talk about this. it's a common situation, actually. you start with weak supervision, get some initial results, and then, hopefully, you want to leverage those results for fully supervised training. i've been down this road more times than i care to count, and it's always a bit of a dance.

basically, what we're doing is taking the noisy, imperfect labels produced by our weakly supervised approach and using them to guide a more robust, fully supervised training process. it's like trying to learn how to build a house using a set of instructions written by someone who wasn't quite sure what they were doing but, at least they gave you an idea of how to begin. you get the basic frame up and then go back with the real blueprint to add more structure and details.

so, the core idea is to treat the weak labels as *pseudo-labels*. these pseudo-labels are then used to train a model in the same way you would train it if you had actual ground truth labels, the main difference here is you'll probably want to take precautions when you do.

let's say, for instance, you were building a text classifier, like identifying the sentiment of a tweet. with weak supervision, you might have used hashtags as a proxy for sentiment. a tweet with #happy might be considered positive and one with #sad, negative. pretty straightforward. this is fast, easy to implement, requires little effort and is usually good enough for a first try, but it's definitely not perfect. there might be ironic uses of hashtags, or maybe the tweet is only partially related to the hashtag's general sentiment. those are things you need to think about and take into account.

now, we use those weak labels to train an initial model. this initial model will likely have moderate performance because it is trained with noise, but it gives us something to work with. the next step is to use this initial model to predict labels on the same dataset that we used for the weak supervision, the dataset which lacks ground truth. these predictions, or rather, the probabilities associated with those predictions are what we are going to use as pseudo labels.

but hold on. using them directly without consideration is not the best option. if a model is very confident with a classification that is wrong the probability assigned will be very high. it would be ideal if we could somehow filter this noise out.

here's where things get interesting. instead of directly using the raw predictions from the weakly supervised model as our new training labels we are going to use the predicted probabilities, but we need to be careful because not every classification will be equally good. you are likely going to want to consider the probability assigned to each class when calculating loss.

one very useful trick is to threshold the predicted probabilities. for example, you could set a threshold where any prediction with a probability less than 0.8 (for example) is discarded. this gets rid of the really noisy stuff that the weak model has labeled incorrectly. it can be used by using these probabilities to compute the loss based on whether this threshold is met, here's some pseudo code.

```python
def compute_loss(model, inputs, weak_labels, threshold=0.8):
    outputs = model(inputs)
    probs = torch.softmax(outputs, dim=1) #assuming classification
    max_probs, predicted_labels = torch.max(probs, dim=1)
    loss = 0
    for i in range(len(weak_labels)):
        if max_probs[i] > threshold:
            loss += loss_function(outputs[i], weak_labels[i]) #if the threshold is met then calculate loss using weakly labels
    return loss
```

this example uses `torch.softmax` which implies that we are doing classification, this snippet is just an example of how you could calculate the loss and is more generic than specific. notice that we only calculate the loss if `max_probs[i]` is higher than the `threshold`. the other case when `max_probs[i]` is smaller than `threshold` is ignored, it effectively means that you are not considering low-confidence labels for updating your model.

i have found that if you do not consider how confident the weakly supervised model is on the assigned labels, it is more than likely that you will just be training your model to perform like your weakly supervised model, which is probably not what you want in the end.

another common approach is to use a technique known as label smoothing. instead of treating a label as 100% correct, you introduce a small probability of incorrectness. this softens the one-hot encoded labels (which means that instead of just having 1 on the assigned label position, you are going to assign a probability which is less than 1 on that position and distribute the remaining probability across the rest of the classes) and makes the model less prone to overfitting on potentially incorrect pseudo-labels. it helps also when your model becomes overconfident on a category, making them slightly less sure about that label.

here's another example on how to implement label smoothing using a very basic approach

```python
def smooth_labels(labels, num_classes, smoothing_factor=0.1):
    smoothed_labels = torch.full_like(labels, smoothing_factor / (num_classes - 1), dtype=torch.float)
    smoothed_labels.scatter_(1, labels.unsqueeze(1), 1 - smoothing_factor)
    return smoothed_labels
```

in this example, `labels` is a tensor containing the pseudo labels. `num_classes` is the total number of classes and `smoothing_factor` determines the amount of smoothing applied to labels. the `scatter_` method is used to assign the 1 - `smoothing_factor` to the correct label position. this ensures that each class now has a probability which is different than zero. for the assigned label the probability will be `1 - smoothing_factor`, and for the rest of the labels its will be `smoothing_factor / (num_classes - 1)`.

another technique which helps is to use consistency regularization. here you apply small perturbations to your inputs and ask the model to have consistent predictions. this regularizes the model, making it more robust to small variations and less dependent on spurious patterns from the pseudo-labels.

for example, you could augment an image with minor rotation or add small amounts of noise, and expect to have almost the same predictions. this ensures that the model is learning the proper features which are not dependent on small input changes.

here is another example on how you could use data augmentation on your training, based on the python package `torchvision` (which, if you are dealing with images, you'll probably end up using eventually, if you haven't already). this snippet is not going to run directly. it requires you to have defined your model and have the input data in the proper format, but it is a good example of how this could be used.

```python
from torchvision import transforms
def train_step(model, optimizer, inputs, labels):
    aug_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    ])
    optimizer.zero_grad()
    augmented_inputs = aug_transforms(inputs)
    outputs_normal = model(inputs)
    outputs_augmented = model(augmented_inputs)
    loss_normal = loss_function(outputs_normal, labels)
    loss_augmented = loss_function(outputs_augmented, labels)
    total_loss = loss_normal + loss_augmented
    total_loss.backward()
    optimizer.step()
    return total_loss
```

what we are doing here is, for every batch, apply augmentation to the inputs and calculate the loss both from the augmented and normal input. the augmentation we are doing is minor rotation, slight translation and minor color changes. this helps on making the model not overfit the specifics of the original image, and more robust to image transformations, improving generalization on unseen images.

the process is iterative. you could use those better labels to train a new better model, and then do it again. it's like bootstrapping yourself to a better result. just keep in mind, if your starting weakly supervised labels are garbage, you are going to end up training your model to be equally as garbage. and that's not a good place to be. it's like trying to teach a dog to play the piano if it can barely sit, you first need to teach it how to sit. and even then, the piano part could be challenging, or maybe not so much if the dog is *that* good.

the success of this process relies on two critical things. first, the quality of your initial weak labels (even if they're noisy, they need to point in the general direction of the correct answers), and second, your ability to handle the noise present in these pseudo-labels. it's about mitigating the risk of the model simply learning the biases of the initial (weak) model.

i recommend you take a look at "semi-supervised learning with deep neural networks" by oliver rippel, and the works of professor andrew ng. also, pay attention to research papers that have "pseudo-labeling" or "consistency regularization" in the title or abstract. they're all goldmines. this should get you started. it's an ongoing field, but knowing the core concepts and common techniques puts you in a good place to tackle this problem. good luck.
