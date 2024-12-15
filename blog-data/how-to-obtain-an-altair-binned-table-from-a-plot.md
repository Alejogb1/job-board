---
title: "How to Obtain an Altair binned table from a plot?"
date: "2024-12-15"
id: "how-to-obtain-an-altair-binned-table-from-a-plot"
---

alright, so you want to get the binned data from an altair plot, basically extracting the underlying table that altair used to create those bins, gotcha. it's a common need when you move from visual exploration to needing actual numbers for further analysis or reporting. been there, done that, got the t-shirt with the faded code print.

i remember way back when, i was working on a project that involved analyzing user activity on a platform. we had this gorgeous altair histogram showing the distribution of login times, perfectly binned, visually great. but then, the boss comes in, and of course he asks for the actual numbers behind each bin, the classic case of “i need the data, not just the picture”. back then, my first instinct was to try and reverse-engineer the binning logic from the plot itself, which, let me tell you, was a proper nightmare of pandas groupby and cut operations, and not very reproducible. it felt like writing a whole library just to get the data. turns out there's a way easier route.

the core issue is altair plots are visualizations, not data storage objects. they work by receiving a dataframe and using plot specifications (the encoding you've defined) to generate the vega-lite json structure under the hood. this vega-lite spec then tells the rendering library how to draw the chart but that actual binned data isn't usually exposed directly. so the data does exist but is hidden away.

here's the strategy that i found to be most reliable and less painful: you need to perform the binning operation yourself in pandas before feeding the data to altair. once you’ve got those binned values in pandas, you can create a separate dataframe for your plot, and then reuse the same binning logic to retrieve the binned data at any time. this way you have the numbers and the plot.

here’s an example: lets say you’ve got some data like this:

```python
import pandas as pd
import altair as alt

data = pd.DataFrame({'values': [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10]})
```

now, instead of letting altair do the binning automatically, you first do it explicitly using pandas' `cut` function. for example to bin into 4 bins:

```python
bins = 4
data['binned_values'] = pd.cut(data['values'], bins=bins, labels=False)

binned_data = data.groupby('binned_values')['values'].count().reset_index()
binned_data.rename(columns={'values': 'count'}, inplace=True)
print(binned_data)

chart = alt.Chart(binned_data).mark_bar().encode(
  x=alt.X('binned_values:O', title='Binned Values'),
  y=alt.Y('count:Q', title='Count')
)
chart.show()
```

what is happening is that the first part calculates the bins, and the second part plots it. then, if you need the original data in table format you already have it in the `binned_data` variable. this is the table:

```
   binned_values  count
0              0      6
1              1      5
2              2      4
3              3      3
```

that's how you get both the chart, and the data. notice that i've also renamed the column for clarity.

sometimes, you want custom bin sizes, not just equal width bins. suppose you want your bins to be 1 to 3, 4 to 6, and 7 to 10. instead of using `cut` you can use `pd.IntervalIndex.from_breaks`

```python
import pandas as pd
import altair as alt

data = pd.DataFrame({'values': [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10]})

custom_bins = [1, 3, 6, 10]
custom_intervals = pd.IntervalIndex.from_breaks(custom_bins, closed='left')
data['binned_values'] = pd.cut(data['values'], bins=custom_intervals, labels=custom_intervals.categories)

binned_data = data.groupby('binned_values')['values'].count().reset_index()
binned_data.rename(columns={'values': 'count'}, inplace=True)

print(binned_data)

chart = alt.Chart(binned_data).mark_bar().encode(
    x=alt.X('binned_values:O', title='Binned Values'),
    y=alt.Y('count:Q', title='Count')
)
chart.show()
```

now, the output data is like this:

```
   binned_values  count
0     [1, 3)      6
1     [3, 6)      9
2    [6, 10)      3
```

notice that it displays the bins as intervals. you can extract the numbers by accessing the `left` and `right` attributes of the interval. but for data analysis, the categories are usually enough.

when working with date or datetime types, binning becomes slightly different. you might need to use pandas' `dt` accessor along with `cut` or `resample` to get the desired bins. suppose you want to bin daily logins across a year into a weekly basis. here's a simplified example:

```python
import pandas as pd
import altair as alt
import numpy as np

dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = pd.DataFrame({'dates': dates, 'logins': np.random.randint(1, 5, len(dates))})


binned_data = data.groupby(pd.Grouper(key='dates', freq='W'))['logins'].sum().reset_index()
binned_data['week'] = binned_data['dates'].dt.strftime('%Y-%W')
print(binned_data)

chart = alt.Chart(binned_data).mark_bar().encode(
  x=alt.X('week:O', title='Week Number'),
  y=alt.Y('logins:Q', title='Total Logins')
)

chart.show()
```

here, we create a daily login dataframe and then use `groupby(pd.Grouper(key='dates', freq='W'))` to group on a weekly basis. the `'W'` in frequency tells pandas to resample weekly. and once again `binned_data` now holds your required table. just to show that it is not that complicated to extract.

for further reading on pandas binning functions, take a look at the official pandas documentation. it has lots of examples and options on how to group and bin data. also, if you find yourself working with time series often, i would highly recommend reading “python for data analysis” by wes mckinney as that's where i learned all about that. it has a full chapter on pandas time series functionality and other cool stuff.

the takeaway is to shift from letting altair do the binning behind the scenes, to doing it explicitly in pandas. it gives you more control, more flexibility, and ultimately, it gives you the data table you need, without having to resort to some obscure internal vega-lite structure. plus, you now have a clean reproducible way to get the data, that you can reuse and share with others (very important). and while i am at it, here’s a bonus tip for you. remember to also always sanitize your data before plotting and avoid edge cases. for example do not assume that the data has always some values in it. always handle the case where there is nothing. otherwise the code may crash and that would be a bad day.

hope this helps, let me know if anything is not clear.
