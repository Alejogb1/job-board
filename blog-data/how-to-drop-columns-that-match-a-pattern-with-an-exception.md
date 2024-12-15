---
title: "How to Drop columns that match a pattern, with an exception?"
date: "2024-12-15"
id: "how-to-drop-columns-that-match-a-pattern-with-an-exception"
---

alright, so you're looking to ditch some columns from a dataframe based on a pattern, but want to keep certain ones. i've been there, trust me. i remember back when i was first knee-deep in data analysis, i had this huge dataset from some sensor readings – think gigabytes of time series data. the columns were a mess, named like `sensor_1_temperature_raw`, `sensor_1_humidity_raw`, `sensor_2_temperature_raw`, and so on. i needed to get rid of all the `_raw` columns but hold on to the `sensor_id` one and other similar ones so i could make some calculations later. what a headache.

anyway, to tackle this, i generally reach for pandas, it's a life saver for this kind of stuff. so lets start, i think the most straight forward approach involves a little bit of regular expression action combined with a list comprehension. it's pretty efficient and readable once you get the hang of it.

here's the basic idea. we use `df.columns` to get a list of all the column names. then, we iterate through this list and apply a regex to check if a column name matches the pattern. if it does *and* it's not in our exception list, we drop it. simple right?

```python
import pandas as pd
import re

def drop_columns_by_pattern(df, pattern, exceptions):
    """drops columns matching a regex pattern, excluding specified ones.

    Args:
        df (pd.DataFrame): the input dataframe.
        pattern (str): regex pattern to match for column names to drop.
        exceptions (list): list of column names to exclude from dropping.

    Returns:
         pd.DataFrame: a new dataframe with dropped columns.
    """
    columns_to_drop = [col for col in df.columns if re.search(pattern, col) and col not in exceptions]
    return df.drop(columns=columns_to_drop)

# example
data = {'sensor_1_temperature_raw': [25, 26, 27],
        'sensor_1_humidity_raw': [60, 62, 61],
        'sensor_2_temperature_raw': [28, 29, 30],
        'sensor_2_humidity_raw': [55, 56, 57],
        'sensor_id': [1, 1, 2],
        'reading_date': ['2024-01-01', '2024-01-02', '2024-01-03']}

df = pd.DataFrame(data)

pattern_to_drop = r'_raw$'  # columns ending with _raw
columns_to_keep = ['sensor_id','reading_date']

df_cleaned = drop_columns_by_pattern(df, pattern_to_drop, columns_to_keep)
print(df_cleaned)

```

in this example, the pattern `r'_raw$'` targets any column name that ends with `_raw`. i made it specific with the `$` anchor. and the `columns_to_keep` list specifies columns that are exempt from being dropped, like `sensor_id`, and `reading_date`.

a common pitfall is forgetting the regex escaping. if you have any special chars in your column names, you need to escape them with a `\`. for instance, if your pattern was something like `sensor(1)_raw` the code would not work correctly, you would need to make it `sensor\(1\)_raw`. regex is its own rabbit hole, there are great resources out there like 'mastering regular expressions' by jeffrey friedl, that could be beneficial if you need to make complex regexes.

sometimes though, you may want a bit more flexibility. like, what if you have a long list of exceptions, or if your regex pattern needs to handle more complex scenarios. in these situations, i like to build a separate boolean mask instead of using an `and` condition inside the list comprehension directly. it makes the logic a bit cleaner and easier to extend. like this

```python
import pandas as pd
import re

def drop_columns_by_pattern_mask(df, pattern, exceptions):
    """drops columns matching a regex pattern, excluding specified ones, using a mask.

    Args:
        df (pd.DataFrame): the input dataframe.
        pattern (str): regex pattern to match for column names to drop.
        exceptions (list): list of column names to exclude from dropping.

    Returns:
        pd.DataFrame: a new dataframe with dropped columns.
    """
    mask = df.columns.map(lambda col: bool(re.search(pattern, col)) and col not in exceptions)
    columns_to_drop = df.columns[mask].tolist()
    return df.drop(columns=columns_to_drop)

#example
data = {'sensor_1_temperature_raw': [25, 26, 27],
        'sensor_1_humidity_raw': [60, 62, 61],
        'sensor_2_temperature_raw': [28, 29, 30],
        'sensor_2_humidity_raw': [55, 56, 57],
        'sensor_id': [1, 1, 2],
        'reading_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'user_id': [101, 101, 102]
        }

df = pd.DataFrame(data)

pattern_to_drop = r'_raw$'  # columns ending with _raw
columns_to_keep = ['sensor_id','reading_date', 'user_id']

df_cleaned = drop_columns_by_pattern_mask(df, pattern_to_drop, columns_to_keep)
print(df_cleaned)
```

this version uses a `map` to apply a lambda function, which returns `true` if a column matches the pattern and is not in the exception list, creating the boolean mask. this is then used to select and drop columns. personally, i think it looks a bit cleaner.

and remember, pandas dataframes are immutable. that means functions like `drop()` don't modify the original dataframe, instead, they return a new dataframe with the columns dropped. so you either need to assign the result back to your dataframe variable or work with the returned dataframe directly. forgetting that fact has been a time sink in my experience. it took a while, i can say that.

now, for the corner cases: what if some of your `exceptions` are not actually in the dataframe, or if a regex is too complex and you want to test it first ? its better to be prepared for that. we can add some checks and try to predict those cases.

```python
import pandas as pd
import re

def drop_columns_by_pattern_safe(df, pattern, exceptions, dry_run = False):
    """drops columns matching a regex pattern, excluding specified ones, with safety checks.

    Args:
        df (pd.DataFrame): the input dataframe.
        pattern (str): regex pattern to match for column names to drop.
        exceptions (list): list of column names to exclude from dropping.
        dry_run (bool): if true prints columns to drop, otherwise drops columns.

    Returns:
         pd.DataFrame or None: a new dataframe with dropped columns, or None for dry run
    """
    valid_exceptions = [col for col in exceptions if col in df.columns]
    invalid_exceptions = [col for col in exceptions if col not in df.columns]

    if invalid_exceptions:
        print(f"Warning: following exception columns do not exist: {', '.join(invalid_exceptions)}")


    columns_to_drop = [col for col in df.columns if re.search(pattern, col) and col not in valid_exceptions]

    if dry_run:
        print(f"columns that would be dropped:\n{columns_to_drop}")
        return None
    else:
        return df.drop(columns=columns_to_drop)


#example
data = {'sensor_1_temperature_raw': [25, 26, 27],
        'sensor_1_humidity_raw': [60, 62, 61],
        'sensor_2_temperature_raw': [28, 29, 30],
        'sensor_2_humidity_raw': [55, 56, 57],
        'sensor_id': [1, 1, 2],
        'reading_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'user_id': [101, 101, 102]
        }

df = pd.DataFrame(data)

pattern_to_drop = r'_raw$'  # columns ending with _raw
columns_to_keep = ['sensor_id','reading_date', 'user_id', 'non_existing_col']

df_cleaned = drop_columns_by_pattern_safe(df, pattern_to_drop, columns_to_keep, dry_run=True)
if df_cleaned is None:
    df_cleaned = drop_columns_by_pattern_safe(df, pattern_to_drop, columns_to_keep)

print(df_cleaned)
```

in this example i added a `dry_run` parameter, it's a technique i use a lot to debug these kind of processes. if it's set to `true`, the function will just print what columns would be dropped without actually modifying the dataframe. and also i have added the warning about nonexistent columns. this helped me avoid plenty of errors in my past, i really recommend to use it. its also a good practice if you are sharing the code with other members in your team.

for more general pandas best practices, i really enjoyed the book "python for data analysis" by wes mckinney, it gives you the bases to work with dataframes at different levels.

so there you have it, three different ways to drop columns by pattern with exceptions. each of the functions above offers its own advantage depending on the problem's specific scenario. remember to test your pattern thoroughly before applying it to a large dataset. its like that time i almost deleted all the columns containing the word `temperature` because my regex had an error. luckily i saw it before pushing the changes to the server. haha, close one!.

let me know if this helps or if you have other similar problems. i've got plenty of war stories when it comes to data wrangling.
