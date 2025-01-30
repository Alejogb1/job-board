---
title: "How can substring filtering be implemented in Altair using parameters?"
date: "2025-01-30"
id: "how-can-substring-filtering-be-implemented-in-altair"
---
Leveraging Altair’s powerful expression language in conjunction with selection parameters provides a flexible approach to substring filtering within interactive visualizations. The core challenge resides in dynamically crafting a logical expression that operates on string fields based on user input. My experience building a data exploration tool for textual analysis required precisely this capability, revealing several key implementation details.

Essentially, we use a signal parameter to capture user input (a text string) and then utilize Altair’s `datum` object and string methods within a conditional selection. This allows data points containing substrings that match the input string to be dynamically highlighted or isolated. The `contains()` method, or variations thereof, becomes pivotal within the logical expression. It's important to understand that the filtering itself does not modify the underlying data, but rather adjusts the visual encoding of marks based on the selection’s conditional application. This also applies to filtering at the transform level via `filter` parameters, though for this exercise, we are primarily focused on visual conditional encoding.

Let’s illustrate this with several practical code examples. Consider a dataset of book titles and authors. We want to enable a user to filter the chart to only display books containing a specified string in the title.

```python
import altair as alt
import pandas as pd

data = pd.DataFrame({
    'title': ['The Lord of the Rings', 'Pride and Prejudice', 'To Kill a Mockingbird', 'The Hobbit', '1984', 'Moby Dick'],
    'author': ['J.R.R. Tolkien', 'Jane Austen', 'Harper Lee', 'J.R.R. Tolkien', 'George Orwell', 'Herman Melville']
})

input_text = alt.param(value='', name='substring')

selection = alt.selection_single(fields=['title'],
                                empty='none',
                                on='mouseover',
                                nearest=True,
                                name='select',
                                clear='mouseout')

chart = alt.Chart(data).mark_bar().encode(
    x='title',
    y='count()',
    color=alt.condition(selection, 'author', alt.value('lightgray')),
    tooltip=['title', 'author', 'count()']
).add_params(
    input_text,
    selection
).transform_filter(
    alt.expr(f'contains(datum.title, substring)')
)

chart.interactive()
```

In this initial example, I’ve defined an `input_text` parameter which is initialized to an empty string. The key element is within the `transform_filter`, where an Altair expression leveraging the `contains` function is used. The syntax `datum.title` references the 'title' field in each data point. The `substring` is a reference to the input parameter. Thus, the filter dynamically includes only data points where the 'title' field contains the user-provided substring. The `selection` parameter is also utilized to enhance visualization by allowing users to highlight or select individual bars. Note that this example uses the transform filter, to show how it works, even though it does not directly apply conditional encoding.

Now let’s look at an example of filtering that *does* use conditional encoding. In this example, we will again use substring matching but alter the color of any row that contains a user input substring within the title. This is particularly useful when you want to maintain the overall data context but highlight only specific entries of interest.

```python
import altair as alt
import pandas as pd

data = pd.DataFrame({
    'title': ['The Lord of the Rings', 'Pride and Prejudice', 'To Kill a Mockingbird', 'The Hobbit', '1984', 'Moby Dick'],
    'author': ['J.R.R. Tolkien', 'Jane Austen', 'Harper Lee', 'J.R.R. Tolkien', 'George Orwell', 'Herman Melville']
})

input_text = alt.param(value='', name='substring')

chart = alt.Chart(data).mark_bar().encode(
    x='title',
    y='count()',
    color=alt.condition(
        alt.expr(f'contains(datum.title, substring)'),
        alt.value('firebrick'),  # color to apply when matching
        alt.value('steelblue')   # default color
    ),
    tooltip=['title', 'author', 'count()']
).add_params(input_text)

chart.interactive()
```

In this refined example, we directly encode color using an Altair conditional expression. When the `contains(datum.title, substring)` expression evaluates to true (the input string is a substring of the current title), we apply a color of 'firebrick'; otherwise, we apply 'steelblue'. This showcases a different strategy of visual highlighting via conditional encoding rather than data filtering through transformations. This avoids filtering out any data so the overall context is always maintained, which is useful in some cases.

Lastly, I want to illustrate a case-insensitive version for the substring matching. This is often crucial, as users might not always remember the exact capitalization. While Altair’s expression language doesn’t offer an inherent case-insensitive `contains` method, we can simulate it through a combination of `lower()` and `contains()`.

```python
import altair as alt
import pandas as pd

data = pd.DataFrame({
    'title': ['The Lord of the Rings', 'Pride and Prejudice', 'To Kill a Mockingbird', 'The Hobbit', '1984', 'Moby Dick'],
    'author': ['J.R.R. Tolkien', 'Jane Austen', 'Harper Lee', 'J.R.R. Tolkien', 'George Orwell', 'Herman Melville']
})

input_text = alt.param(value='', name='substring')

chart = alt.Chart(data).mark_bar().encode(
    x='title',
    y='count()',
    color=alt.condition(
        alt.expr(f'contains(lower(datum.title), lower(substring))'),
        alt.value('forestgreen'),
        alt.value('cadetblue')
    ),
    tooltip=['title', 'author', 'count()']
).add_params(input_text)

chart.interactive()
```
Here, we apply `lower()` to both the `datum.title` and the user provided `substring` prior to the `contains()` function call. This ensures that string comparison is performed without regard to case sensitivity. This approach significantly improves usability, since user input will not be required to be a perfect case match, and it is generally considered best practice for many data visualization scenarios.

In summary, dynamic substring filtering in Altair is achieved by combining parameter inputs, Altair expression language, and conditional encoding. While the specific methods shown use `contains`, other variations for more complex string matching may be constructed through a combination of `regexp_match` and other string manipulation capabilities within the Altair expression system. It’s essential to remember that the filtering does not alter the underlying data but adjusts visual encoding based on conditional checks, or filters the data using transformations. This provides a more fluid and responsive user experience for exploring data containing text values.

For further learning on string manipulation functions within Altair’s expression language, reviewing the official Altair documentation regarding transformations and selections is strongly recommended. A thorough examination of the Vega-Lite expression language specification provides greater understanding of available functions and conditional operators. Furthermore, studying examples of parameter utilization in Altair galleries will offer inspiration and insight into sophisticated filtering and interactive visualization design. These resources, combined with experimentation, provide a solid foundation for mastering interactive text filtering with Altair.
