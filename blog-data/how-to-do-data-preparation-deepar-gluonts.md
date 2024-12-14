---
title: "How to do Data preparation DeepAr GluonTS?"
date: "2024-12-14"
id: "how-to-do-data-preparation-deepar-gluonts"
---

alright, let’s talk about prepping data for deepar in gluonts. it's a crucial step, and i've definitely spent my fair share of late nights debugging data issues that cropped up when i thought i was done. the thing about time series, is that the format has to be spot on or you'll get nowhere.

so, first off, gluonts expects your data in a very particular format, which is basically a list of dictionaries. each dictionary represents a single time series. and within that dictionary, you'll find at least two keys: `target`, which is your actual time series values, and `start`, which is the starting timestamp. that’s the bare minimum. if you have additional features, these will be included too.

i’ve seen so many people get tripped up by the timestamp format. it’s got to be a pandas timestamp object, or a string that pandas can interpret into a timestamp. otherwise, gluonts just throws its hands up. i remember one project, some data was coming in as raw unix timestamps; i had to wrangle that into pandas datetime objects before anything would work. a little pain, but lessons learned.

let's look at some examples. if your time series data is just a simple list of numbers and you also have the date for the beginning:

```python
import pandas as pd

data = [10, 12, 15, 13, 16, 18, 20]
start_date = "2023-01-01"

ts_data = {
    "target": data,
    "start": pd.Timestamp(start_date)
}

print(ts_data)
```

this is a pretty basic example. the `start` is now a pandas timestamp. now, you can create a list of such dictionaries for multiple time series. but before moving on lets look into the time frequency. the default, is often daily, but you might be working with hourly data. let’s say the data is hourly and we want to specify this correctly within gluonts:

```python
import pandas as pd

data = [10, 12, 15, 13, 16, 18, 20, 22, 25, 24, 23, 26]
start_date = "2023-01-01 00:00:00" # note the time
freq = "1H" # hourly frequency

ts_data = {
    "target": data,
    "start": pd.Timestamp(start_date),
    "freq": freq
}

print(ts_data)
```

notice the `freq` parameter. this will save you headaches. if not specify it gluonts might guess it incorrectly and you'll be debugging that for a while... trust me! i know this from a project where the data was actually every 15 minutes but the model was training with an hourly frequency. it was a mess. i learned a lot about time series frequencies the hard way that month.

ok, now what about more complex scenarios when you have features (dynamic features)? it is not much harder, just add the features into the dictionary.

```python
import pandas as pd
import numpy as np

data = [10, 12, 15, 13, 16, 18, 20, 22, 25, 24, 23, 26]
start_date = "2023-01-01 00:00:00"
freq = "1H"

#create some dynamic features as an example
n_steps = len(data)
feature_1 = np.sin(np.linspace(0, 10, n_steps))
feature_2 = np.cos(np.linspace(0, 10, n_steps))


ts_data = {
    "target": data,
    "start": pd.Timestamp(start_date),
    "freq": freq,
    "dynamic_feat": [feature_1, feature_2]
}

print(ts_data)
```

notice the new entry `dynamic_feat`. the shape of the features must match the length of the target data and must be a numpy array (or be convertible to). also, the `dynamic_feat` entry must be a list of features, even if you only have one. the feature themselves must have the same shape than the target. if you have more complex features, you should make sure that they are properly aligned with the target values. misalignment can cause the model to not learn properly. in my experience, this is where many unexpected behaviours show up. i was doing a project last year and the dynamic features were lagging 1 hour, the model was basically learning the past by looking in the future. it took me hours to find the problem and the results were bad.

a couple more things to consider. handling missing values: deepar can't deal with NaNs out of the box. you need to either fill them or drop the series with missing values (not recommended if the number of missing values is not that much). i typically prefer to fill them using methods like forward fill or backfill as these have shown good results in multiple occasions. but that depends on the nature of the data at hand. another thing is how to handle outliers. outliers can hurt the training and can really skew the model, particularly deepar with its attention mechanisms are prone to be very sensible to these outliers. removing or clipping outliers is often beneficial.

a final point, don’t treat all your time series the same. some series will have very different scales, and some will be more volatile than others. it is a good idea to normalize or standardize the data. i've found that standardizing each time series independently often helps deepar to learn faster. it prevents the model from becoming biased towards series with larger values or larger standard deviations.

about resources. for a solid understanding of time series fundamentals, i’d recommend reading “time series analysis” by james d. hamilton. it’s a classic for a reason. and if you really want to get into the nitty-gritty of deep learning for time series, look for papers from the neurips, icml, and aistats conferences. you know, real research. that is where i learned how to use gluonts. there are some excellent papers by the original authors. sometimes is not about the most recent paper, rather about the first paper of the method. it is were the core insights and motivations are better described. there is no shortcuts. if you want to have an advanced understanding of a particular machine learning method, you must understand how it started. also, the book "forecasting: principles and practice" by hyndman and athanasopoulos is a very good general purpose forecasting resource.

and if after all that your data still throws errors at you? well, that happens, the nice thing about programming is that it is a never ending puzzle game… sometimes i feel like a detective looking for that one tiny issue preventing the program to run smoothly. i guess that is why i love doing this. just try again, and again... and again… until it works.

oh and before i forget. a programmer's spouse asks, "can you go to the grocery store and buy a gallon of milk? and if they have eggs, get a dozen." the programmer comes back with a dozen gallons of milk.
