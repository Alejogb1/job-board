---
title: "How to Check to see if a column has part of a string - if it does, I want to return a value to a new column?"
date: "2024-12-15"
id: "how-to-check-to-see-if-a-column-has-part-of-a-string---if-it-does-i-want-to-return-a-value-to-a-new-column"
---

alright, so you’re looking to see if a string exists as a substring within values of a dataframe column, and if it does, then flag it in a new column. i’ve been down that road more times than i care to remember, it's a pretty common data cleaning task. let me walk you through how i usually handle this, and share some war stories from my past.

first off, i'm going to assume you're working in python, likely using pandas. that's where i spend most of my time when dealing with tabular data. if not, the core logic should be transferable but you'll need to adapt the syntax to your specific environment.

the heart of the matter lies in using string methods effectively. pandas provides vectorized string operations, which means you can apply string functions to an entire column of data without looping, which is way more performant, and avoids python’s loops.  if you were working with basic python list and looping you are going to get crazy slow performance.

let's start with a simple example: say we have a dataframe representing customer information, and a column named 'customer_notes' where free form notes exist. we want to flag rows where the note contains the string “vip”:

```python
import pandas as pd

data = {'customer_id': [1, 2, 3, 4, 5],
        'customer_notes': ["normal customer", "this is a vip",
                          "another normal note", "vip vip member",
                          "just a regular joe"]}
df = pd.dataframe(data)

search_term = 'vip'
df['is_vip'] = df['customer_notes'].str.contains(search_term, case=false).astype(int)
print(df)

```

what's happening here?

*   we use `df['customer_notes'].str.contains(search_term)` . the `.str` accessor enables us to use string methods on the column values, which is awesome. the method `contains` does the heavy lifting, and returns `true` if the substring exist, and `false` otherwise.
*   i have added the option `case=false`, which means this operation will be case insensitive. if you don't want that, omit the parameter and it will be case sensitive.
*   finally, `.astype(int)` converts the boolean (`true`/`false`) results to integers (`1`/`0`), which is often useful. you can skip this if you want to keep the boolean result in the column.
*  i am printing the whole dataframe so we can visually inspect the changes.

when i started out, i totally missed the `.str` accessor and ended up iterating through every row, checking the substring. it was slow and terrible! i probably lost a week's worth of computing time to those bad loops, i still have nightmares about it. using vectorized operations from libraries like pandas is one of the biggest lessons i’ve learned over time.

now, this handles a straightforward search for a single substring. what if you want to look for multiple substrings and flag it differently? for instance, let’s say i have product descriptions, and i want to tag descriptions based on keywords like 'red', 'blue', and 'green' and put those color descriptions as a new column:

```python
import pandas as pd

data = {'product_id': [101, 102, 103, 104, 105],
        'description': ["a blue car with red stripes", "a green shirt",
                          "just a regular product", "a very red hat",
                          "no colors"]}
df = pd.dataframe(data)

color_mapping = {'red': 'has_red', 'blue': 'has_blue', 'green': 'has_green'}

def apply_color_flags(row):
  for color, flag in color_mapping.items():
    if row['description'].lower().find(color) != -1:
        return flag
  return 'no_color'

df['color_flag'] = df.apply(apply_color_flags, axis=1)
print(df)

```

in this scenario:

*   i define a mapping `color_mapping` of what color corresponds to the new column value.
*   i use a python function `apply_color_flags`, that we apply through the pandas `apply` method. since we need to check multiple conditions of the same column we need to perform the apply row by row, this is why we pass `axis=1`.
*   inside the function, i lowercase the whole string to handle possible capitalization variations, then use `find` instead of `contains`. in this example i decided not to use the vectorized method and use a more common python style code because the goal was to assign multiple different values to the new column, this function is not as performant as the previous example.
*   if no match is found, i return "no\_color" as a value for the column, you can leave that as a blank string if you like.

this approach allows you to handle different substrings and create new columns based on the matches. i have personally used this code to label sentiment from text and create categories based on keywords. in a former job where i was working on marketing datasets, we used to parse the text of different marketing campaigns, and use it to find which segments of users were more receptive to the text. with this method we could automate the tagging of thousands of campaigns.

here’s one last example; what if you need to deal with more complex search patterns? like maybe we need to extract different numeric values from a string? for example, suppose you have something like ‘product id: 1234 price: 19.99’. and you want to extract both the product id and the price in two different columns. in those cases you can use regular expressions:

```python
import pandas as pd
import re

data = {'product_info': ["product id: 1234 price: 19.99",
                        "product id: 5678 price: 99.50",
                        "product id: 9012 price: 42.00",
                        "no price or product data"]}
df = pd.dataframe(data)


def extract_product_data(row):
    match = re.search(r'product id: (\d+) price: ([\d.]+)', row['product_info'])
    if match:
      return match.group(1), match.group(2)
    return None, None

df[['product_id', 'price']] = df.apply(extract_product_data, axis=1, result_type='expand')
print(df)
```

the `re.search` method searches for the pattern and if finds a match it return the match object, which allows us to group the sub-matches.

*   `r'product id: (\d+) price: ([\d.]+)'` is the regular expression. `(\d+)` means “one or more digits” and `([\d.]+)` means “one or more digits or periods”. the parenthesis will capture the groups, and we extract those with the `.group(1)` for the first group and `.group(2)` for the second group. we can extract more groups by using the corresponding index.
*   if no match is found we return none for both groups.
*   `result_type='expand'` makes sure that the returned tuple will be expanded into two separate columns. otherwise, it will return a single column with the tuple.
*  i am printing the dataframe to see the result of the operation.

regular expressions are a powerful but complex tool. if i was starting out i would be more prone to use the previous methods to handle my string operations, and only use regular expressions when i cannot get my solution with normal string methods.

i remember one time, i spent hours crafting a regular expression to extract data from log files. i got it to work after many trials and errors. but then, two days later, the log format changed, and my regular expression broke. that’s when i learned that sometimes, simpler solutions, like splitting by known delimiters, are way more robust to format changes. but hey, at least it was a good learning experience. i even wrote a blog post about it at some point, i think it was called ‘regex adventures’ or something like that... anyways don't rely on regex too much.

that’s my take on how to approach this problem. i suggest you check out “python for data analysis” by wes mckinney. and if you are interested in a more complex way to tackle this you can check "mastering regular expressions" by jeffrey friedl, i recommend this book a lot but it’s important to emphasize the complexity, you can probably get away with just simple methods 99% of the time.
