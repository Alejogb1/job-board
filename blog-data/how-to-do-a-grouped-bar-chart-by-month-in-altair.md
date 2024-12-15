---
title: "How to do a Grouped bar chart by month in Altair?"
date: "2024-12-15"
id: "how-to-do-a-grouped-bar-chart-by-month-in-altair"
---

i've tackled grouped bar charts in altair plenty of times, and believe me, getting them just right can feel like navigating a maze at first. so, let's break this down for generating one grouped by month. it's a common request for time-series data.

first, the core idea is to structure your data correctly for altair. altair prefers a long format where each record represents a single bar, and your grouping variables (in this case, months) are in one column. you don't want separate columns for different groups as the chart won't be drawn properly. we'll need a column for your 'x' axis (months), another for your 'y' axis (some value), and a third for your group (the variable that defines the grouping within the month, could be a category, a type of event or whatever).

a mistake i made early on, and have seen happen to many others, was trying to create different chart encodings for each group. like trying to create each group as separate bars and then stack them. altair's magic, though, comes from the `color` encoding in most cases. this encoding assigns different colors and groups within our chart based on the categorical column. also, i once spent a whole evening looking for a bug only to figure out that my data was not sorted by month which was driving me nuts because the labels on the axis were all jumbled up and did not make sense. lesson learned: pre-sort your data!

let's look at a very basic example. we will generate some dummy data using pandas, because it makes it very very easy, and i really recommend anyone using altair to become well versed in pandas, this will be a real game-changer to get the desired outputs from altair without losing your mind in the process.

```python
import pandas as pd
import altair as alt

# dummy data creation
data = {
    'month': ['jan', 'jan', 'feb', 'feb', 'mar', 'mar', 'apr', 'apr'],
    'category': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
    'value': [10, 15, 12, 8, 18, 20, 14, 16]
}
df = pd.DataFrame(data)

chart = alt.Chart(df).mark_bar().encode(
    x='month:O',
    y='value:Q',
    color='category:N'
).properties(
    title='dummy bar chart by month'
)

chart.show()
```

in this very simple example, we've defined an `x` axis as month, `y` as our value, and the `color` is mapped to `category`. `:O` and `:Q` and `:N` at the end of the field names are altair's data type designators for ordinal, quantitative and nominal respectively. they are very important for altair to interpret the data correctly. so always include them when possible.

if you run this code, you'll see a perfectly functional grouped bar chart. each month on the x axis will show two bars side-by-side for the two categories.

now, let's say your dataset is in a different format. maybe it's a csv file, or maybe it's in a wide format, which means there are several columns each one representing a different grouping category. you'd need to reshape your data. pandas `melt()` method is exactly what you need for this. i had to use this extensively in an old analysis project for a logistics company, where data from different warehouses was spread across different columns, quite annoying to manage. so pandas reshaping capabilities became a core component of that project.

here's how you could tackle that:

```python
import pandas as pd
import altair as alt

# dummy wide data
wide_data = {
    'month': ['jan', 'feb', 'mar', 'apr'],
    'category_a': [10, 12, 18, 14],
    'category_b': [15, 8, 20, 16]
}

df_wide = pd.DataFrame(wide_data)

# reshaping to long format
df_long = pd.melt(df_wide,
                    id_vars=['month'],
                    value_vars=['category_a', 'category_b'],
                    var_name='category',
                    value_name='value'
                    )

# cleaning category column to remove _ from categories
df_long['category'] = df_long['category'].str.replace('category_','')

chart = alt.Chart(df_long).mark_bar().encode(
    x='month:O',
    y='value:Q',
    color='category:N'
).properties(
    title='grouped bar chart with wide data'
)

chart.show()
```

here, we first create a wide format dataframe. we use `pd.melt()` to reshape it into the correct long format, where each record has a 'month', 'category' and a 'value' columns. we clean the categories to avoid 'category_' prefixes and then plot it just like before. the resulting chart will be similar to the previous one, even though the data has a very different format, which shows you the importance of understanding how to use `melt()` to achieve your goals.

now, what if your months are not neatly ordered, perhaps they are dates? or maybe your months are not all in lowercase letters? this also happens a lot, dates or month labels coming in different formats. this may create some very confusing plots where everything is all jumbled up. altair automatically sorts ordinal fields alphabetically, but with dates or actual month names this is not ideal. we need a way to sort months chronologically. for this, we will need to tell altair that the month axis should respect the order we define, not the alphabetical one.

```python
import pandas as pd
import altair as alt

# dummy data creation with mixed case months
data = {
    'month': ['jan', 'FEB', 'mar', 'Apr', 'MAY', 'jun'],
    'category': ['a', 'b', 'a', 'b', 'a', 'b'],
    'value': [10, 15, 12, 8, 18, 20]
}
df = pd.DataFrame(data)

# cleaning the months by setting them to lower case
df['month'] = df['month'].str.lower()

# specifying the month order
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun']

chart = alt.Chart(df).mark_bar().encode(
    x=alt.X('month:O', sort=month_order),
    y='value:Q',
    color='category:N'
).properties(
    title='grouped bar chart with custom ordering'
)

chart.show()
```

first, we made sure all months are lowercase. then we defined a `month_order` list with our correct order of the months, and passed this list into the `sort` parameter within the `x` encoding, this ensures the graph is drawn in the correct order. this was a life saver when i was generating some time series data for a client and they asked for specific sorting in their reports, because i didn't have to manually sort my data before plotting it.

remember that the order you specify in the list is how altair will order the axis, so make sure is correct. always remember that the sorting is always applied to ordinal axis. quantitative or nominal axis will not respect this parameter.

some final touches, these are not mandatory but will make your plots so much more user-friendly and are worth considering: altair by default will scale the y-axis to fit the maximum data point value which can sometimes result in charts that are not ideal for comparison or might look squashed. it is recommended that you set your own y-axis limits using `scale` on your y axis encoding. adding tooltips using `tooltip` parameter can help improve the user experience by showing the exact values of each bar on hover.

for learning more about data visualization and altair i recommend "interactive data visualization for the web" by scott murray as a good starting point as well as "python data science handbook" by jake vanderplas. the altair official documentation is also very well written and detailed, i use it almost on a daily basis, and believe it is very well structured to answer all kind of questions related to altair.

i hope this helps and it clarifies the steps required to generate grouped bar charts in altair, if you have more questions please just ask, always happy to help. oh and remember, why did the data scientist break up with the bar chart? because they were growing apart! i know, lame joke. but i had to.
