---
title: "How to reshape nested json data in a dataframe using python to get a desired output?"
date: "2024-12-15"
id: "how-to-reshape-nested-json-data-in-a-dataframe-using-python-to-get-a-desired-output"
---

alright, so you've got some nested json, and you're trying to wrangle it into a dataframe. been there, done that, got the t-shirt, probably spilled coffee on it too. it's a classic data munging problem. i've spent way too many hours staring at json dumps that look like they were designed by a committee of squirrels before i figure out how to massage them into something usable. let's break this down into some practical steps.

first thing's first, you gotta understand the shape of your json. is it a list of dicts? a single dict with nested dicts? something more horrific? the key is to visualise it in your head, think of it like a tree, the root of the json becomes your main data container, and the branches as the inner json objects you want to access. this is critical before we even touch the code. i've seen folks try to jump straight to pandas without really understanding their json, and it usually ends up a messy disaster. trust me on that one. a decade of experience staring into data messes teaches you a few lessons.

let's assume for the sake of this answer you have a situation where your json is a list of dictionaries, and each dictionary contains some simple keys and also has another dictionary nested inside, this is one of the most common scenarios i have seen.

the pandas library is your best friend for this, for anyone that doesn't know pandas it’s a core library for data analysis in python that gives you the power to manipulate tabular data easily. it can directly read json into dataframes, but when things are nested, well, it needs a little nudge. the trick is to normalise the data first, using the `json_normalize` function of pandas. it flattens the json into a tabular format. this is the key to a tidy dataframe. i spent months trying to do this in pure python loops, writing a function that can handle nested dictionaries manually, until i stumbled upon json_normalize, it felt like i discovered fire again, and learned a big lesson that always check for existing tooling that is already solved for me.

here's a basic example of how to do it:

```python
import pandas as pd

data = [
    {
        "id": 1,
        "name": "item_a",
        "details": {"color": "red", "size": "large", "weight": 10}
    },
    {
        "id": 2,
        "name": "item_b",
        "details": {"color": "blue", "size": "small", "weight": 5}
    }
]

df = pd.json_normalize(data)
print(df)
```

this will create a dataframe but with the "details" dictionary now flattened and keys prepended with the `details.` prefix which looks like this:

```
   id    name details.color details.size  details.weight
0   1  item_a           red        large              10
1   2  item_b          blue        small               5
```

this works when the inner structure is consistent. but life's never that simple, is it? what if you had a different json structure? or you need to handle variable keys inside your nested json. let’s say now you have a list of dictionaries where one of the keys contains a list of dictionaries instead of a single dictionary inside, now you must first iterate through your list of dictionaries and flatten that list into the same level before json_normalize can create your dataframe, because we can only flatten one level at the time, you will need to repeat this process for other levels of nesting.

here is an example code on how to proceed if we have a different nested structure:

```python
import pandas as pd

data = [
    {
        "id": 1,
        "name": "product_x",
        "attributes": [
             {"feature": "screen", "value": "amoled"},
             {"feature": "cpu", "value": "snapdragon"}
         ]
    },
    {
        "id": 2,
        "name": "product_y",
        "attributes": [
            {"feature": "memory", "value": "16gb"},
            {"feature":"storage", "value":"1tb"}
         ]
    }
]
exploded_data = []
for row in data:
    for item in row['attributes']:
        new_row = row.copy()
        new_row.update(item)
        del new_row['attributes']
        exploded_data.append(new_row)

df = pd.json_normalize(exploded_data)
print(df)
```

this flattens the list of dictionaries inside attributes into multiple rows, now we have a new column feature that contains the feature's name and value contains the feature value which looks like this:

```
   id       name   feature      value
0   1  product_x    screen     amoled
1   1  product_x       cpu  snapdragon
2   2  product_y    memory       16gb
3   2  product_y   storage        1tb
```

another case i experienced is when you have a list of dictionaries where inside each dictionary you have a key that its value is a dictionary, and inside that dictionary you have multiple keys with different keys, but the values are always a list with a single dictionary, and you want to access that specific dictionary inside the list. this is very common on api responses that you want to reshape the response into a more usable structure. sometimes you have to spend hours debugging those apis just to find out that an element is nested as a dictionary inside a single element list!

here is how to access the inside values:

```python
import pandas as pd

data = [
    {
        "id": 101,
        "item": "gadget_alpha",
        "configurations": {
            "setting_a": [{"value": "high"}],
            "setting_b": [{"value": "enabled"}],
            "setting_c": [{"value": "25"}],
            "setting_d": [{"value": "option_1"}]
        }
    },
    {
         "id": 102,
         "item": "gadget_beta",
         "configurations": {
            "setting_a": [{"value": "low"}],
            "setting_b": [{"value": "disabled"}],
             "setting_c": [{"value": "10"}],
             "setting_d": [{"value":"option_2"}]
         }
    }
]

exploded_data = []
for row in data:
   new_row = row.copy()
   for key, value in row['configurations'].items():
      new_row[key] = value[0]['value']
   del new_row['configurations']
   exploded_data.append(new_row)

df = pd.json_normalize(exploded_data)
print(df)
```

here we iterate trough the inner dictionary and extract the value of each dictionary inside the list and append it to a new column based on the key that corresponds to the nested list of dictionary inside configurations. the final result is:

```
   id         item setting_a setting_b setting_c  setting_d
0  101  gadget_alpha      high   enabled        25   option_1
1  102   gadget_beta       low  disabled        10   option_2
```

i know that seems like a lot of code, and it’s very common to get lost inside multiple iterations over multiple levels of json objects. in my early days i struggled with writing complex loops when i didn’t understand the json structure, so my advice here is to always print every single step of the way so you can see what you are doing and catch errors before it is too late, i also have spent many hours writing loops that are over-complicated when a single line of code could have handled it.

there's also the question of handling different data types in your json values, sometimes you can have numbers stored as string, or null values that can mess up with your dataframe if not handled correctly. pandas is pretty good at auto-detecting data types. but you can explicitly define the data types for each column by using the `dtype` parameter when creating the dataframe, or you can coerce the data after creation using the `astype` method.

remember to use the `inplace=true` parameter when making changes to your dataframes like `df.rename(columns={'old_name':'new_name'}, inplace=true)` if you are not going to store it into a new variable, if not it will not store the changes and it will cause you a debugging headache. this is another trap that got me multiple times.

finally, if you have exceptionally complex json structures, sometimes it is better to break things down into smaller, manageable steps, you can even write functions that flatten a specific level to keep your code more readable. i usually start simple and when my logic is correct i refactor the code into more reusable functions, this is the secret to readable code. also remember that readable code is also maintainable code. and if you have a teammate or your future self trying to debug the code, you would thank past you for doing that.

if you are looking for more info, the pandas documentation is always a good place to start for the basic usage of pandas functions, for more advanced use cases you might need to go into more specific libraries depending on your specific issue. for a deep dive into structured data, "data structures and algorithms in python" by michael t. goodrich, roberto tamassia and michael h. goldwasser is a fantastic resource. and "fluent python" by luciano ramalho, if you want a comprehensive view on python programming, and it will also teach you how to write a beautiful code.

it's not rocket science, it's just a matter of methodically breaking down the problem, and remember that half the battle is understanding the data. and probably 30% is debugging, 15% is reading documentation and the last 5% is actually writing the solution. and before i forget, what do you call a data scientist who is always right? a data whisperer... i'll show myself out.
