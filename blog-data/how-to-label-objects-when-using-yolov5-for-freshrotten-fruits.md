---
title: "How to label objects when using YoloV5 for fresh/rotten fruits?"
date: "2024-12-15"
id: "how-to-label-objects-when-using-yolov5-for-freshrotten-fruits"
---

alright, so you're diving into object detection with yolov5, specifically for fresh and rotten fruit labeling. that's a common scenario, and i've definitely been down that road myself, many times. let me walk you through how i typically handle this, based on past experiences and some common pitfalls i've seen.

first things first, the core of this is your dataset. yolov5, like most object detection models, learns from the labels you provide. a good dataset is the foundation, if it's garbage the results will be garbage too. so lets assume you've already collected the images you want to use, you need to label each fruit, this means that you'll need to draw boxes around the fruits you want to detect. the critical part, where the fresh/rotten bit comes in is in how we assign *classes* to those bounding boxes.

think of classes like categories. you won’t label an object as “just fruit”, because then the model wouldn't know the difference between fresh and rotten. you need distinct classes. in our case, it sounds like you want at least two: `fresh_fruit` and `rotten_fruit`. if you have multiple types of fruit, you will need more classes of course, like `fresh_apple`, `rotten_apple`, `fresh_banana`, `rotten_banana`. you see the pattern. it is possible to use more classes like `partially_rotten_apple` or even some different classes of rotteness, but this adds a layer of complexity. lets go for the simple version, that is easier for everyone.

so, to actually create these labels, you can’t just draw boxes and say 'it’s a fruit'. the yolov5 expects labels to be in a specific format. traditionally, this means a separate text file for each image, with the same name. for example, if you have `apple_123.jpg` as your image, then you will have a `apple_123.txt` with the labels. the text files use a specific format, each line contains the information about one bounding box, including the class it belongs to, relative to the image size.

here's how that line looks like:
```
<class_id> <x_center> <y_center> <width> <height>
```

the `<class_id>` is an integer, assigned to each class according to the order they are in the data file (see below). the `<x_center>`, `<y_center>`, `<width>`, and `<height>` are all *normalized* values, ranging from 0 to 1. this makes the process independent of the image resolution, so if you resize the image the bounding boxes will be correctly scaled.

here's a quick example. suppose you've got an image with a single fresh apple, and the top-left corner of your box is (100, 150) and the bottom-right corner is (300, 350). the image dimensions are, let's say, 600x400.

first, determine `x_center`, `y_center`, `width`, and `height` in pixels. the center of the box is (200, 250), width is (300-100) = 200, and the height is (350-150) = 200. now we normalize:

`x_center = 200 / 600 = 0.333`
`y_center = 250 / 400 = 0.625`
`width = 200 / 600 = 0.333`
`height = 200 / 400 = 0.5`

then, if `fresh_fruit` is class id 0, the corresponding line in your text file will be:
```
0 0.333 0.625 0.333 0.5
```

so that's the core part of the annotation. now how do you do it easily? you do not want to manually calculate this for all your images. manual annotations takes too long. luckily, there are several free and open source annotation tools available. i've personally used a variety of them, i've found that `labelImg` is an ok tool for this kind of work, it's simple and to the point, so I would recommend it. it's a gui tool, it allows you to draw the boxes on the images and the labels in the format you need, that’s basically all it does, but it does it well. other alternatives include `cvat`, or `makesense.ai`. the best depends on your particular setup and needs, but they all do the same job, they create the txt files.

the crucial part is that when you label, you have a `data.yaml` or `classes.txt` (this depends on what version you're using) file that defines the classes. yolov5 needs this file to know what class id 0, 1, 2 or whatever number means.

here's a basic example of `classes.txt` for our two class example, `fresh_fruit` and `rotten_fruit`:

```
fresh_fruit
rotten_fruit
```

this means that `fresh_fruit` has the class_id 0 and `rotten_fruit` has the class_id 1. when you annotate you have to select this when drawing each bounding box.

and here is an example of a `data.yaml` file:

```yaml
train: path/to/your/train/images
val: path/to/your/val/images
nc: 2  # number of classes
names: ['fresh_fruit', 'rotten_fruit']
```

`nc` is the total number of classes and the names have to be in the same order as the class ids, as discussed above.

a few things i've learned the hard way:

*   **consistency is key**: be super consistent when labeling. if you label something as a `rotten_fruit`, always label it as `rotten_fruit`. be extra careful when drawing the boxes, try to fit them closely. if you have some borderline cases, where it’s not exactly clear if it's fresh or rotten, make a decision and stick with it throughout the entire dataset. it's like when i was a kid trying to sort my lego blocks, if i started classifying pieces as the “reddish” and then changed to the “orangish-red” the organization was a complete disaster.
*   **data augmentation**: consider using data augmentation. this means flipping, rotating, adding noise, etc. you can do it with yolov5's built in functions when training, and this can be very effective. small datasets can lead to overfitting. in one particular project, i was labeling some very difficult to differentiate objects, adding data augmentation boosted the training a lot. if you cannot add more annotated samples, this is the way to go.
*   **validation set**: split your data into training and validation sets. typically something around 80-20 or 70-30 works. i usually randomize and then create the folders accordingly. the validation set allows you to keep an eye on the generalization during the training process. a model that trains well in the training set but performs bad in the validation means that you are overfitting, which means that your model has learned all the particularities of the training set and will not perform well with unseen data.
*   **check your labels**: after you finished the annotations, it is wise to check a few of the bounding boxes before training your model. i had some issues when the model performed poorly, only to realize that the labeling was wrong. i had boxes out of place, wrong classes, stuff like that. it can save you from wasting time later.

for resources, i'd point you towards the original yolov5 paper and the official documentation. they are the best places to start for in-depth understanding. also, deep learning books that cover object detection in general, like "deep learning with python" by francois chollet, are great for understanding the bigger picture, which can be useful. but the most helpful resource is other people's code, specifically the official yolov5 repo. it includes tutorials and also different jupyter notebook examples on how to train your model. it’s useful to test your dataset, it’s usually the most important part.

so that's the whole process. it might seem like a lot, but once you get a good workflow going it gets easier. just remember the basics: good labels, consistent annotation, proper class definitions, a solid train/val split, and careful review of your labels before training. it’s not rocket science, it's just being organized and methodical, a bit like properly sorting and organizing my tools in my garage so i know where everything is at all times. and, yes, i also label them.

if you're still having issues, provide more details, maybe a snippet of your label files and i will try to help.
