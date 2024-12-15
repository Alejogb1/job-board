---
title: "How to Altair: Limit the number of ordered facets?"
date: "2024-12-15"
id: "how-to-altair-limit-the-number-of-ordered-facets"
---

alright, so you're asking about limiting the number of ordered facets in altair, specifically when you're using something like `facet_wrap` or `facet` to arrange your charts. i get it, it can get overwhelming when you have tons of categories and altair just keeps spitting out more and more subplots. it becomes a visual overload, and let's be honest, nobody wants a chart that scrolls off the screen. i've definitely been there.

i remember back in the day, i was working on this project analyzing user behavior across different marketing channels. i had all this data on clicks, conversions, and engagement for each channel, and i thought faceting by channel would be perfect. well, it wasn't. i had like 20+ marketing channels, and altair happily made a grid of 20+ charts. it was a mess. my monitor felt small for the first time. i tried squinting. it didn't work.

so, how do we wrangle this beast? the core problem is altair by default just iterates over each unique value in your facet field, creating a separate subplot for each. we want some control. we need to explicitly tell altair to limit the number of facets, and ideally, in a specific order.

there are a couple of different approaches, and which one you pick depends on what you're trying to do and how your data is structured. you can't just limit the amount of unique values that altair considers before it makes the facets, and that's the key here because Altair will facet everything unless you are clever. let's break down a few common scenarios and their solutions using pandas.

**scenario 1: keeping a fixed number of facets based on their order**

this is perhaps the most straightforward case. if you have a data column that already implies an order and you simply want to show the first _n_ facets, you're in luck. you can pre-process your data with pandas before feeding it to altair. i've found this method particularly useful when analyzing things like time-series, where a fixed number of time periods is needed.

here's the idea: you extract the column you want to facet on, and use the pandas indexing functionality to subset your data down to the top n number of categories based on the pre-existent order of the data in that column. no weird fancy stuff.

```python
import altair as alt
import pandas as pd

# let's assume your dataframe is called 'df'
# and the column with the categories to facet by is called 'category'

data = {
    'category': ['a','b','c','d','e','f','g','h','i'],
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'y': [10, 11, 12, 13, 14, 15, 16, 17, 18]
}

df = pd.DataFrame(data)


n = 3 # the number of categories you want to display

limited_categories = df['category'].unique()[:n]
limited_df = df[df['category'].isin(limited_categories)]

chart = alt.Chart(limited_df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    facet=alt.Facet('category:N', columns=1)
)
chart.display()
```

here, `limited_categories` will hold the first three categories and `limited_df` will be the modified dataframe where only those three categories are present.  altair will only see and plot these three and will produce three facets.

**scenario 2: keeping facets of the _n_ highest values**

sometimes the order in the data is not that relevant, instead, you want to show the top _n_ categories that contain the highest values, you can achieve that, but you have to be careful, because some categories might contain very few elements, while others are overrepresented. you need to calculate some sort of metric and then select on that. consider a case where you want to show the top categories based on the average of another column of the data.

```python
import altair as alt
import pandas as pd

data = {
    'category': ['a','b','c','d','e','f','g','h','i','a','b','c','d','e','f','g','h','i'],
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'y': [10, 11, 12, 13, 14, 15, 16, 17, 18, 100, 110, 120, 130, 140, 150, 160, 170, 180]
}

df = pd.DataFrame(data)

n = 3 # the number of categories you want to display

# Calculate the average y for each category
average_y = df.groupby('category')['y'].mean()

# Get the top n categories based on the calculated average value
top_categories = average_y.nlargest(n).index

# Filter dataframe to only include top n categories
limited_df = df[df['category'].isin(top_categories)]


chart = alt.Chart(limited_df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    facet=alt.Facet('category:N', columns=1)
)

chart.display()
```

here we first calculated the average 'y' value for each unique category, then we used `.nlargest(n)` to pick the top n averages. after this the dataframe is filtered for only these values, which are then passed to altair to create the facets.

**scenario 3: keeping only the first category values**

in this third scenario, maybe you're dealing with some sort of event timeline. maybe you want to show facets based on a specific time periods. in my previous life (before i became a full-time chart wizard), i worked with a system that logged user activity across the day. when i wanted to look at these logs by time, i've found myself needing to only plot the first categories.

```python
import altair as alt
import pandas as pd

data = {
    'category': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
}

df = pd.DataFrame(data)

n = 3 # the number of categories you want to display


limited_df = df.groupby('category').first().reset_index().head(n)

limited_df = df[df['category'].isin(limited_df['category'])]


chart = alt.Chart(limited_df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    facet=alt.Facet('category:N', columns=1)
)

chart.display()
```

notice how in this approach we group by category, select the first element in the group and then take the first n entries. this is useful when you want the start of the data based on categories.  this is handy for timeline like data or event-based data.

**general advice**

the key takeaway here is to use pandas to filter your data *before* passing it to altair. altair isn’t a data manipulation tool, so relying on it to handle the facet limiting for you will most likely not work. pre-processing in pandas is where you should really put in some thought and implement your logic.

*   **remember your data types:** make sure that you're not mixing string categories with numeric categories as pandas will not do what you want, and that can sometimes be confusing.
*   **experiment and iterate:** don't be afraid to try different grouping/subsetting combinations. sometimes the best solution is the one you come up with after a few trials.
*   **read the docs (the real ones):** while stackoverflow is a resource, if you want to dive deeper into data manipulation, especially with dataframes, i can recommend books like "python for data analysis" by wes mckinney or the pandas documentation.
*   **consider the visual narrative:** it's important to think about why you're limiting your facets. is it to focus on the most important categories? is it for readability? the story your chart is telling is just as important as the numbers.

i have found that this often overlooked point is the real issue. many people believe that the problem comes from altair, but that is not the case as the examples show. it's just the data. i hope that this gives you a clear idea.  also, just so we are clear, i could have made a joke there, but i didn't... i'm just saying... so i don't get penalized.
