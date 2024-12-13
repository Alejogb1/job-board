---
title: "pandas case when default in pandas?"
date: "2024-12-13"
id: "pandas-case-when-default-in-pandas"
---

Okay so you're wrestling with pandas and that whole "case when default" situation it's a common headache I've been there countless times let me tell you

First off lets break down what we mean by "case when default" in pandas because it's not a single function thing like in SQL we're talking about conditional column creation and transformations with a default value if none of the conditions match yeah? Think of it like a giant if else if else chain but in pandas dataframes

I remember back in like 2016 I was working on this massive dataset for a startup I was at it had all sorts of customer data and we needed to categorize users based on their purchase history it was a mess of different columns and I spent a whole weekend figuring this out I was trying to avoid those clunky for loops and you know get something vectorized

The initial approach was kind of ugly like trying to use `apply` with a bunch of if statements it worked but slow as molasses not to mention debugging a nested lambda function is a special kind of torture so I ditched that real quick there had to be a better way and there is thankfully

So here's the core concept you want to use `np.select` it's numpy's vectorized conditional logic tool that's what you need for speed and clean code trust me It takes a list of conditions and a list of corresponding values and yeah the key part here a default value if none of the conditions are true it's exactly what we need for our "case when default" scenario

Let's dive into some example code because code explains things way better

```python
import pandas as pd
import numpy as np

# Sample DataFrame
data = {'product_id': [101, 102, 103, 104, 105],
        'price': [10.0, 20.0, 30.0, 15.0, 25.0],
        'discount': [0.1, 0.2, 0.0, 0.15, 0.25]}
df = pd.DataFrame(data)

# Conditions
conditions = [
    df['price'] > 25,
    (df['price'] > 15) & (df['discount'] > 0.1),
    df['price'] < 15
]

# Corresponding values
choices = [
    'Premium',
    'Special Discount',
    'Value'
]

# Default Value
default = 'Standard'

# Using np.select with default
df['category'] = np.select(conditions, choices, default=default)
print(df)
```

Okay so in this snippet we create a simple dataframe and define three conditions based on price and discount each condition has a corresponding value if the condition is met and the default value Standard is applied if no condition is true so this is our case when default equivalent using np.select its like a clean if elif else but way more pandas-friendly

Now you might be thinking "what if my conditions are more complex or need multiple columns?" good question its not always simple but the concept remains the same you build your conditions using logical operators and you can even use functions or methods that work on series its very versatile but you need to keep it vectorized as much as possible or you lose most of the speed advantages of pandas here is a more complex example

```python
import pandas as pd
import numpy as np

# Sample DataFrame with multiple columns
data2 = {'user_id': [1, 2, 3, 4, 5],
         'purchase_amount': [100, 250, 75, 150, 300],
         'days_since_last_purchase': [30, 10, 60, 20, 5],
         'has_coupon': [True, False, True, False, True]}
df2 = pd.DataFrame(data2)

# Complex Conditions
conditions2 = [
    (df2['purchase_amount'] > 200) & (df2['days_since_last_purchase'] < 15) & (df2['has_coupon']),
    (df2['purchase_amount'] > 150) | (df2['days_since_last_purchase'] < 30),
    df2['purchase_amount'] < 100
]

# Corresponding values
choices2 = [
    'High Value User',
    'Active User',
    'Low Value User'
]

# Default Value
default2 = 'Regular User'

# Apply np.select
df2['user_segment'] = np.select(conditions2, choices2, default=default2)

print(df2)
```

Here you can see how we use multiple columns and more complex logical operations and yeah the performance is still pretty good because of numpy's vectorization its the key to avoiding those for loops which are incredibly slow

Now you also might be thinking "but what if i need different default values based on some other criteria or a more dynamic way of choosing the default value?" good question again pandas provides a few alternatives if your cases are too complicated for np select

A couple of options you can use are the .apply method in a dataframe with a lambda function to create dynamic default behavior and the `fillna()` method in combination with Boolean masking or the `where()` method

I will give you an example of `where()` method for that because its a cleaner approach than `fillna()` in this kind of problem.

```python
import pandas as pd
import numpy as np

# Sample DataFrame
data3 = {'value': [10, 20, -1, 30, -2],
         'status': ['ok', 'ok', 'error', 'ok', 'error']}
df3 = pd.DataFrame(data3)

# Conditions for replacing negative values with dynamic default based on status
df3['adjusted_value'] = df3['value'].where(df3['value'] >= 0, other =
    df3['status'].apply(lambda status: 0 if status == 'ok' else -100))
print(df3)
```

In this case we use the `where` method and we apply the condition that the value must be greater than 0 if not we use a lambda function that checks the status column to determine the default value. So we dont have only one default but a default derived from another column.

I once had a bug where I was accidentally setting a default value to zero when it should have been a very large number because of some silly typo man that took a while to debug but you learn from those mistakes you know it's like a right of passage to lose a few days on small bugs like that I would say but that's debugging for ya

I don't have links for you because I'm not some search engine I'm giving you personal experience not some Google result but for more detailed explanations and a deeper dive I would highly recommend two resources which I found very helpful in my past experiences "Python for Data Analysis" by Wes McKinney is essential for pandas understanding It's the creator of pandas and his book will give you much more than you need for this specific problem and if you really want to delve into the performance aspects of vectorized operations and numpy in general then "High Performance Python" by Micha Gorelick and Ian Ozsvald is a great choice but its a lot more deep and extensive so start with McKinney book first

So to summarize np.select is your best friend for simple case when default logic in pandas the other options are there but keep in mind vectorized operation performance and also keep an eye on your default values because sometimes they might not be what you intended them to be trust me on this one I've been there

Hopefully this was helpful and saved you some headache good luck with your pandas adventures
