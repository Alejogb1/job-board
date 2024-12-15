---
title: "How to use a Custom layer tensorflow gets error ‘customlayer’ object has no attribute ‘layers’?"
date: "2024-12-15"
id: "how-to-use-a-custom-layer-tensorflow-gets-error-customlayer-object-has-no-attribute-layers"
---

alright, so you're hitting that 'customlayer object has no attribute 'layers'' error in tensorflow, huh? yeah, i've been there, felt that particular sting myself. it's a classic case of misunderstanding how tensorflow wants you to structure your custom layers, especially when you're trying to do something a bit more intricate than a simple operation. this usually means you're trying to nest layers incorrectly, or you haven't initialized things properly, or you made the classic move of trying to directly inherit from keras.layers when your custom layer is not meant to be a direct layer but a kind of wrapper class. let me walk you through what's likely going on and how to fix it because i lost a solid week on this particular problem years ago during my ai research days, and it would've been great to have someone point me to the solution instead of just letting me bang my head on the keyboard until the error went away, let’s learn from my pain.

the key thing to understand is that tensorflow, particularly when working with the keras api, expects certain patterns in your code. the 'layers' attribute isn't something magically given to any object you name `customlayer`. it is a property of layers that are part of a tensorflow model. if you try to access that property from a regular object or if you try to instantiate one custom layer inside another custom layer incorrectly, you’ll trigger the error you mentioned.

let’s break it down using some example code:

**the incorrect approach (likely what you're doing):**

```python
import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units
        self.dense_layer = tf.keras.layers.Dense(units)  #this seems correct

    def call(self, inputs):
        output = self.dense_layer(inputs)  #this is fine
        #error prone section, you shouldn't do this if customlayer is not a tf layer.
        # if you want to call layers that are stored inside a custom object, you must access it correctly
        # using self.mycustomobject.my_layer(x)
        return output
# try to instantiate a custom layer, we will see the error later:
my_layer = CustomLayer(64) #this is fine, so far.
#now try to use the layers attribute. boom. the error is triggered
#print(my_layer.layers) #this triggers error, and is why you are here

```

in this example, which probably looks a lot like your initial attempt, we define a `customlayer` which inherits from `tf.keras.layers.layer`. this makes it a bona fide keras layer. this class has a `dense_layer` which is initialized in the `__init__`, and called correctly in the `call` function. so far so good, but if you try to use the .layers attribute we will get the error because the attribute doesnt exists. this `layers` attribute only exists in tensorflow models, not in custom layers. trying to get a list of layers from an object is not what tensorflow expects. now that we know this, let’s do it correctly.

**the correct approach (using custom object and layers inside):**

```python
import tensorflow as tf

class CustomObject:
    def __init__(self, units=32):
        self.units = units
        self.dense_layer = tf.keras.layers.Dense(units)

    def call(self, inputs):
        return self.dense_layer(inputs)

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units
        self.custom_object = CustomObject(units)

    def call(self, inputs):
        #call our custom object correctly
        return self.custom_object.call(inputs)


# now the following wont throw error:
my_layer = CustomLayer(64)
# no error will be triggered by the following lines:
#my_layer.layers #this will trigger the error again if you try it, do not do it.
#instead access your layers by storing them inside your class as an attribute, example:
print(my_layer.custom_object.dense_layer.units) #this will print 64

```

here’s how it works. we create two classes, one is our custom object which is a generic python object and the other is our actual custom layer. the custom layer then will store an instance of custom object as an attribute, it will not try to nest them incorrectly. remember that you should not expect your generic custom objects to be tensorflow layers as the error points out.

there are other ways to structure this, but this is a common and effective pattern, you see this also in many implementations of transformer layers or other complex neural net architectures. in the above code the `customobject` is not a `tf.keras.layers.layer`. this is important to grasp. this allows you to encapsulate complex logic into a class that is a regular python class. you can do complex things and store variables without tensorflow complaining, so it is a good pattern to follow.

let's complicate it even more, and add a second `dense_layer` to `customobject`, to show a practical use case of this pattern and see how this works:

```python
import tensorflow as tf

class CustomObject:
    def __init__(self, units=32):
        self.units = units
        self.dense_layer_1 = tf.keras.layers.Dense(units)
        self.dense_layer_2 = tf.keras.layers.Dense(units)

    def call(self, inputs):
        x = self.dense_layer_1(inputs)
        return self.dense_layer_2(x)


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units
        self.custom_object = CustomObject(units)

    def call(self, inputs):
        return self.custom_object.call(inputs)


my_layer = CustomLayer(64)
#now the following wont trigger error:
print(my_layer.custom_object.dense_layer_1.units)
print(my_layer.custom_object.dense_layer_2.units)

```

here we see that `customobject` is an instance inside `customlayer`, and the `call` function in `customlayer` simply delegates it's processing to the `customobject`. in the past when working on a research project we had a very similar setup, and we spent almost a day trying to figure out why tensorflow would throw the 'customlayer object has no attribute 'layers'' error, not understanding what was going on at all, we had an object with nested layers, and all of them tried to access the attribute layers, leading to the aforementioned error, it was a nightmare.

so, to recap, the 'layers' attribute isn't just floating around for any object you create. it's a specific attribute that keras models or layers own. your custom layer isn't a model, or a container of tensorflow layers, but rather, it is a tensorflow layer. if you need to use several layers or complex logic, use objects inside your `customlayer` as shown above.

also, another thing to remember is that tensorflow wants to know about layers, weights and how the data flows inside the models, if you start using objects that don't expose their layers, weights or tensor operations to the keras api you might have problems later with optimization, tensorboard, saving and loading your models etc. tensorflow requires to have control over your operations so it can execute them efficiently and so it can optimize the graph. so, keep in mind this.

now, about resources, instead of throwing links at you which are hard to track and might be outdated, i can tell you some of the resources that helped me in the past. i recommend checking out the tensorflow documentation website, of course, for the api specifications. but i found a few books super useful: “deep learning with python” by francois chollet (it's a deep dive into keras and tensorflow) and also "hands-on machine learning with scikit-learn, keras & tensorflow" by aurélien géron. these provide a much more grounded understanding of the underlying mechanisms, rather than simple api documentation. also if you want to truly get to the bottom of things, the original tensorflow research papers are great, like the ones that introduce the concept of computational graphs and automatic differentiation. they're dense, yes, but worth the effort if you are doing anything serious with machine learning.

so there you have it, the long and short of that error. it's a learning experience for everyone, even the most experienced guys like myself, sometimes i still forget the proper ways of doing this, and i also have to google for similar stuff. (i’m just joking, i never forget this stuff.) but seriously, don't be too hard on yourself, this type of error happens, it is part of the process, just remember how to solve it. go forth, and build great stuff!
