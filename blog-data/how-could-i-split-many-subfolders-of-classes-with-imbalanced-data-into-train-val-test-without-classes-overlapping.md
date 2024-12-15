---
title: "How could I split many subfolders of classes with imbalanced data into train val test, without classes overlapping?"
date: "2024-12-15"
id: "how-could-i-split-many-subfolders-of-classes-with-imbalanced-data-into-train-val-test-without-classes-overlapping"
---

alright, i see the pickle you're in. dealing with imbalanced datasets across a bunch of subfolders – been there, done that, got the t-shirt. it's a common headache in the ml world, especially when things are organized in a particular way. let me walk you through how i've handled this in the past, and it should point you in the direction you need.

the key here is to ensure, as you pointed out, that you don't get class leakage across your train, validation, and test splits. by that i mean, a class showing in one split shouldn't be present in the other two. this is critical for proper evaluation. you wouldn't want your model to have "seen" the test classes during training. that defeats the whole purpose of independent evaluation.

my first encounter with this was back when i was working on a project involving different types of medical scans. we had a folder for each type, and within that folder were individual patient scans, each labelled with a specific condition. imbalanced class distributions were a daily occurrence. we had a lot more of one condition than others.

the structure was something like this:

```
medical_scans/
├── scan_type_a/
│   ├── patient_1/
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   └── ...
│   ├── patient_2/
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   └── ...
│   └── ...
└── scan_type_b/
    ├── patient_1/
    │   ├── image_1.png
    │   ├── image_2.png
    │   └── ...
    └── ...
```

so here's the approach i used, and that worked decently well:

first, let's import some needed libraries, the pandas library it isn't essential but helps handling the data later if needed.

```python
import os
import random
from collections import defaultdict
import shutil
import pandas as pd
```

now, the core logic of creating our splits. i typically prefer a dictionary-based approach. it keeps the data structured nicely:

```python
def create_splits(root_dir, train_ratio=0.7, val_ratio=0.15):
    """splits subfolders of classes into train/val/test sets, no class overlap.

    Args:
      root_dir (str): root directory containing class subfolders.
      train_ratio (float): proportion of data to use for training.
      val_ratio (float): proportion of data to use for validation.

    Returns:
      dict: dictionary containing train/val/test file lists.
    """
    
    all_classes = os.listdir(root_dir)
    splits = defaultdict(list)

    for class_name in all_classes:
        if not os.path.isdir(os.path.join(root_dir, class_name)):
            continue  #skip non dir files
        
        class_path = os.path.join(root_dir, class_name)
        all_items = [os.path.join(class_path,item) for item in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, item))]
        
        random.shuffle(all_items)
        num_items = len(all_items)
        num_train = int(num_items * train_ratio)
        num_val = int(num_items * val_ratio)
        
        splits['train'].extend(all_items[:num_train])
        splits['val'].extend(all_items[num_train:num_train + num_val])
        splits['test'].extend(all_items[num_train + num_val:])

    return splits

```
this function takes your root directory, where all those subfolders are stored and calculates the train, validation, test splits respecting the input ratios. by randomly shuffle the list first, we ensure that the train, val, test lists will have diverse data avoiding having patients or similar in the same dataset.

to use this, you would call it like this.

```python
root_directory = "medical_scans" #replace with your own path
split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
splits = create_splits(root_directory, split_ratios['train'], split_ratios['val'])

print(f"train set size:{len(splits['train'])} images")
print(f"validation set size:{len(splits['val'])} images")
print(f"test set size:{len(splits['test'])} images")
```

this would give you a dictionary with lists of file paths for each split, now we can move these filepaths to different folders to mimic our split needs, by doing this we avoid altering the original structure.

```python
def move_files_to_splits(splits, output_dir):
    """moves files listed in a splits dictionary into train/val/test folders.

    Args:
      splits (dict): dictionary containing train/val/test file lists.
      output_dir (str): output directory to store split data.
    """
    for split_name, file_paths in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True) #creates dir if does not exist
        for class_path in file_paths:
            class_name = os.path.basename(class_path)
            new_class_path = os.path.join(split_dir, class_name)
            shutil.copytree(class_path,new_class_path)

output_dir = "split_scans"
move_files_to_splits(splits, output_dir)
```
this would recreate the data structure with the files in it respecting the splits.
now, depending on your specific needs, you might need to tweak things a bit further.

another tricky bit i encountered early on was when some subfolders were essentially empty or had a really low amount of data, the split might fail because the division would produce empty lists. so in these cases it's better to filter out any empty classes or re organize the data into a more balanced manner.

one more thing, for dealing with class imbalances during training, i found that using weighted loss functions (like focal loss or class-balanced cross-entropy loss) or sampling techniques really helps, i usually read papers and articles instead of using random blog posts, for example, the paper “focal loss for dense object detection” is essential to understand these topics better. also check “class-balanced loss based on effective number of samples” for further reading on this, they are very illuminating and useful if you are trying to understand the math behind the implementation of these techniques. also if you are a book person "deep learning" from ian goodfellow et al it is a must read for all things related to deep learning.

also, remember to document everything. future you will thank you, trust me. once i spent a whole week trying to figure out why my training wasn't working until i realized i had a typo in a path name, it was not a good week. it's like going to a party dressed as a database admin – nobody gets it, but it's still part of your identity.

anyways, let me know if you have other questions. this is my usual approach, and i have had great results on this, i hope this helps.
