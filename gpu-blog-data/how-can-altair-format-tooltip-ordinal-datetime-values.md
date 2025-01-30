---
title: "How can Altair format tooltip ordinal datetime values?"
date: "2025-01-30"
id: "how-can-altair-format-tooltip-ordinal-datetime-values"
---
Altair's default tooltip behavior for ordinal datetimes often presents a challenge: numerical representation rather than human-readable dates and times. I encountered this directly while developing an interactive dashboard for tracking sensor readings, where timestamps, despite being used as an ordinal scale, needed to appear in a clear, formatted date-time style when users hovered over data points. The core issue is that Altair, when dealing with an ordinal scale, defaults to displaying the underlying numerical value, representing time since the Unix epoch. Therefore, proper tooltip formatting requires explicit transformation.

To effectively address this, I rely on Altair's `tooltip` encoding coupled with a specific set of options to perform the necessary conversion. The challenge stems from needing to interpret the ordinal scale value, typically a Unix timestamp representation, as a time-based object for formatting. Altair does not directly provide a built-in feature that automatically translates an ordinal scale into formatted datetime tooltips without explicit instructions. Instead, a combination of `timeUnit`, `format`, and potentially a custom JavaScript function is often required. I find that leveraging `timeUnit` along with the `format` property within the `tooltip` encoding is a versatile and manageable approach for most use cases.

Let me illustrate with some specific examples:

**Example 1: Basic Date Formatting**

This first example demonstrates how to format an ordinal datetime value as a simple date representation (YYYY-MM-DD).  I've assumed here that the data `df` contains a column `timestamp` representing the number of milliseconds since the Unix epoch (this is the format Altair expects for ordinal time values) and a column `value` representing another quantitative measure.

```python
import altair as alt
import pandas as pd

data = {
    'timestamp': [1678886400000, 1678890000000, 1678893600000, 1678897200000, 1678900800000],
    'value': [10, 15, 12, 18, 20]
}
df = pd.DataFrame(data)

chart = alt.Chart(df).mark_line().encode(
    x=alt.X('timestamp:O', title="Timestamp"), # Explicitly use O for ordinal
    y='value:Q',
    tooltip=[
        alt.Tooltip('timestamp:O',
                    title="Date",
                    timeUnit="yearmonthdate",
                    format="%Y-%m-%d"), # Format specified
        alt.Tooltip('value:Q', title="Value")
    ]
)
chart
```

Here, I've specified `x='timestamp:O'` to declare that the `timestamp` column should be treated as an ordinal value, which is crucial for proper time-based data visualization in Altair. This also means Altair uses underlying numeric values for positioning. Within the `tooltip` encoding, I use a `Tooltip` object to define what's displayed in the tooltip. Crucially, I added `timeUnit="yearmonthdate"` which instructs Altair to interpret the numerical `timestamp` as a datetime object using year, month, and day information, enabling the `format` argument to then be applied. `format="%Y-%m-%d"` sets the string representation of that datetime object, resulting in human-readable dates like "2023-03-15". This transformation is critical; without `timeUnit`, the numerical representation of the `timestamp` will be shown.

**Example 2: Date and Time with Specified Format**

This expands on the first example and demonstrates the formatting of both date and time information with a specific format (YYYY-MM-DD HH:MM:SS). My use case often requires precise timestamps to be displayed when reviewing events.

```python
import altair as alt
import pandas as pd

data = {
    'timestamp': [1678886400000, 1678890000000, 1678893600000, 1678897200000, 1678900800000],
    'value': [10, 15, 12, 18, 20]
}
df = pd.DataFrame(data)

chart = alt.Chart(df).mark_line().encode(
    x=alt.X('timestamp:O', title="Timestamp"),
    y='value:Q',
    tooltip=[
        alt.Tooltip('timestamp:O',
                    title="DateTime",
                    timeUnit="yearmonthdatehoursminutesseconds",
                    format="%Y-%m-%d %H:%M:%S"),
        alt.Tooltip('value:Q', title="Value")
    ]
)
chart
```
In this scenario, the core difference lies within the `tooltip` specifications. I've switched the `timeUnit` argument to `yearmonthdatehoursminutesseconds` to specify the precision required. Then, I specified the format with `format="%Y-%m-%d %H:%M:%S"`. This change in `format` string enables display of time components (hours, minutes, seconds). This provides a clearer view of when data was recorded which is often required for analyzing logs. This demonstrates the flexibility `timeUnit` provides and the importance of matching the format string with the selected time units. The principle remains that Altair uses numerical values as underlying ordinal values, but within the tooltip, this is transformed into human-readable date/time information.

**Example 3:  Local Timezone and Custom Formatting**

This example integrates a JavaScript-based timezone transformation, accommodating scenarios where timezone awareness is necessary. While Altair's built-in `timeUnit` functionality is suitable for many situations, sometimes a more custom approach is necessary when direct timezone conversion within the Vega-Lite spec is challenging or impossible. Here, I'm leveraging a custom JavaScript expression to perform the necessary conversion.

```python
import altair as alt
import pandas as pd

data = {
    'timestamp': [1678886400000, 1678890000000, 1678893600000, 1678897200000, 1678900800000],
    'value': [10, 15, 12, 18, 20]
}
df = pd.DataFrame(data)

timezone_js = """
        datum.timestamp  ?
        new Date(datum.timestamp).toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            timeZone: 'America/New_York'
            }): '';
        """

chart = alt.Chart(df).mark_line().encode(
    x=alt.X('timestamp:O', title="Timestamp"),
    y='value:Q',
    tooltip=[
        alt.Tooltip(field='timestamp', type='quantitative',
                   title="DateTime (NYC)",
                   format=timezone_js
                  ),
        alt.Tooltip('value:Q', title="Value")
    ]
)
chart
```

In this advanced example, the significant shift lies in the `tooltip` encoding for the `timestamp` field.  I've moved away from Altair's `timeUnit` and used `format` to embed a JavaScript string. I've also switched the `type` to 'quantitative' since `format` is going to process the underlying number. The `timezone_js` string contains a JavaScript expression executed within Vega-Lite. It takes the Unix timestamp (represented as `datum.timestamp`), converts it to a JavaScript Date object, and then uses `toLocaleString` with the `'America/New_York'` timezone to format it into a human-readable string. This showcases a scenario where the user requires specific timezone representation and demonstrates how a custom Javascript function provides the needed level of control. While this approach allows for sophisticated transformations, I ensure to thoroughly test these JavaScript expressions for performance and accuracy.

In summary, properly formatting ordinal datetime tooltips in Altair involves understanding that the underlying value is an integer. The `timeUnit` parameter in `alt.Tooltip` is used to transform that numerical representation into a date object, and the `format` argument determines the string representation. I often use custom JavaScript expressions when specific timezone or formatting requirements exceed Altair's built-in capabilities.

For continued learning, I recommend consulting Altair's documentation on encodings and tooltips which elaborates further on the `timeUnit` and format parameters. Additionally, referencing Vega-Lite’s specification on time unit definitions and formatting can provide a deeper understanding of the underlying principles. Finally, resources describing JavaScript's `Date` object and formatting capabilities can be incredibly beneficial when using custom JavaScript functions for tooltip formatting.
