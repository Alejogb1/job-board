---
title: "What is the difference between the following tfds.load() calls?"
date: "2024-12-14"
id: "what-is-the-difference-between-the-following-tfdsload-calls"
---

alright, let's break down those `tfds.load()` calls, because yeah, they can look pretty similar on the surface, but under the hood they're doing subtly different things. it's one of those areas where the devil is in the details, and if you’re not paying attention, you can end up with unexpected behaviour, trust me, i've been there.

first off, `tfds.load()` is your go-to function in tensorflow datasets for, well, loading datasets. it handles all the complexities of downloading, preparing, and splitting data, so you can just get straight to working with it. but the exact way it does that depends a lot on the arguments you throw at it.

so, let's imagine we have these three different calls:

```python
import tensorflow_datasets as tfds
ds1 = tfds.load('mnist')
ds2 = tfds.load('mnist', split='train')
ds3 = tfds.load('mnist', split=['train', 'test'])
```

`ds1`: this one is the most basic. when you simply call `tfds.load('mnist')`, you're telling tensorflow datasets "give me the mnist dataset, and give me all of the splits". what tfds does is it returns a dictionary. the keys of this dictionary are the names of the splits available for the dataset and the values are tensorflow datasets `tf.data.Dataset` objects. these splits often consist of `train`, `test`, and sometimes a `validation` split depending on the dataset, check the documentation for the specifics of a given dataset. now, when i say ‘give me all the splits’, that also means downloading and preparing all the splits which might take longer the first time you execute this function. i remember the first time i did this, on a slow network, it felt like an eternity. i had to go make coffee three times.

so to be extra specific `ds1` becomes a python `dict` like this:

```python
{
    'train': <tensorflow.python.data.ops.dataset_ops.PrefetchDataset at 0x...>,
    'test': <tensorflow.python.data.ops.dataset_ops.PrefetchDataset at 0x...>
}

```

where each value is a `tf.data.Dataset` instance. you'll typically see datasets like `mnist` prepared that way, they are usually designed with different splits of the data. this method is useful if you want to have access to all the different parts of a dataset from your python code, maybe you are experimenting with different splits or building a complex validation pipeline. you have the power, but comes a bit of extra complexity because you have to iterate the dictionary to get to the specific datasets.

`ds2`: here, things start to get interesting. `tfds.load('mnist', split='train')` is way more precise than `ds1`. this says "i only want the 'train' split of the mnist dataset". so instead of a dictionary of datasets, tfds gives you just the single `tf.data.Dataset` representing the training set. only that, nothing else.

it's far more efficient if you know that you only need one split. tfds is clever enough to prepare only that specific split, meaning it can save you a lot of time and space. it’s also far more simple to consume since it returns a dataset directly instead of a dictionary. this call is generally the way you want to get the dataset if you are just doing normal training of a model, and you want a `tf.data.Dataset` out the door.

now, a funny story, the first time i made this mistake was when i was trying to build a training pipeline for a transformer model. i used the method that outputs a dictionary and all i got was type mismatch errors. i was banging my head against the wall for 2 hours before realizing what the hell was happening, and that i should just read the documentation. the difference was just that i had to access the value using `'train'` key in a dictionary. i was using the dataset directly, instead of the dataset itself!

`ds3`: now we are getting into more specific scenarios, `tfds.load('mnist', split=['train', 'test'])` tells tfds "i want both the 'train' and 'test' splits". this is a little bit of an hybrid of the two other scenarios. and you may ask why would you do that over the first option, where you get the same thing, the reason is performance. it is not always the case but in some complex datasets it could be more efficient to specify the splits this way as you are telling `tfds` explicitly which splits to download and prepare, avoiding some overhead.

in this case, tfds will return, you guessed it, also a dictionary, but the dictionary only contains the datasets specified by the list in the `split` argument. the keys will be only `train` and `test`. you can choose any available split in this method and pass a list with any combination of splits that the dataset has available. it is especially useful when you want to work with a combination of `train`, `validation` and `test` datasets but want to avoid downloading other potential unused splits that might come with a dataset.

for `ds3` we would obtain an dictionary object like this:

```python
{
    'train': <tensorflow.python.data.ops.dataset_ops.PrefetchDataset at 0x...>,
    'test': <tensorflow.python.data.ops.dataset_ops.PrefetchDataset at 0x...>
}
```

so, to summarise all of this in a practical way:

*   if you just need one split of a dataset for training, use `tfds.load('dataset_name', split='split_name')` as in the example of `ds2`. it’s the most efficient for that case.
*   if you need multiple splits, but not all, you can specify what you need by using `tfds.load('dataset_name', split=['split_name1', 'split_name2'])`, like the `ds3` example.
*   if you want to access all the splits, including splits you might not need, you can simply use `tfds.load('dataset_name')` like the `ds1` example and then specify which one to use after you have the dictionary object.

here's some code to show the differences in action:

```python
import tensorflow_datasets as tfds
import tensorflow as tf

#load all splits
datasets_all = tfds.load('mnist')
print("all splits loaded:")
print(type(datasets_all))
print(datasets_all.keys())
print(type(datasets_all['train']))
print("--------------------")

#load only train split
datasets_train = tfds.load('mnist', split='train')
print("only train split loaded:")
print(type(datasets_train))
print("--------------------")


#load test and train splits
datasets_train_test = tfds.load('mnist', split=['train','test'])
print("train and test splits loaded:")
print(type(datasets_train_test))
print(datasets_train_test.keys())
print(type(datasets_train_test['train']))

```

running this code, you'd see how `datasets_all` is a dictionary containing all the data as a `tf.data.Dataset`, `datasets_train` is a `tf.data.Dataset` directly and `datasets_train_test` is a dictionary with a specific set of keys. you can then start working directly with these datasets by iterating and doing your usual data processing.

```python
#example of using the loaded datasets:
#iterating over a specific batch

for example in datasets_train.take(2):
    print("shape of a sample of the training dataset:")
    print(example["image"].shape)
    print(example["label"])

for key, value in datasets_all.items():
    print(f"split:{key}")
    for example in value.take(2):
        print("shape of a sample from all splits:")
        print(example["image"].shape)
        print(example["label"])
```

in this code, we are iterating over a batch from the dataset loaded in `datasets_train` and then doing the same for each dataset in the `datasets_all` dictionary. this is just an example on how to use the datasets once they are loaded.

```python
# using a dictionary when it's not needed, can cause problems:
# this code is intended to illustrate the common error

try:
    for example in datasets_all.take(2):
      print("This will not work the way we think, the dataset is not directly accessible:")
      print(example["image"].shape)
      print(example["label"])
except Exception as e:
    print(f"Error ocurred: {e}")
```

this code will throw an error, because the dataset, a `tf.data.Dataset` object is nested inside the dictionary, and it’s not directly accessible like this. this is very important to know as this is a very common mistake with the function.

now, some recommendations for further reading:

*   the official tensorflow datasets documentation is crucial. they have examples and a more in depth explanation than i've given here.
*   for deep dive into datasets there's "hands-on machine learning with scikit-learn, keras & tensorflow" by aurélien géron. this book has a whole section on working with data and tensorflow datasets.
*   also, consider reading the tensorflow documentation on `tf.data.Dataset` api and it's various transformations, as this would be the main object that you are manipulating when training models in tensorflow.

this should get you going, and help you better understand what's happening behind the scenes when using `tfds.load()`. it’s all about knowing what exactly you need from the data and choosing the correct method to access it. knowing this helped me a lot when dealing with very large datasets.
