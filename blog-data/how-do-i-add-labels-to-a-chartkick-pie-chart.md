---
title: "How do I add labels to a Chartkick pie chart?"
date: "2024-12-23"
id: "how-do-i-add-labels-to-a-chartkick-pie-chart"
---

Let’s unpack adding labels to Chartkick pie charts. Over my years of dealing with data visualization, I’ve come across this challenge countless times—often finding that the default presentation just doesn’t cut it. It's not about just making the pie chart ‘look pretty,’ it’s about effectively communicating the underlying information. Chartkick provides a robust framework, but sometimes we need to go beyond the initial setup to tailor the presentation to the specific needs of the project.

The out-of-the-box Chartkick pie chart typically displays slices with percentages when hovering over them. However, labels directly on or near each slice make the data much more immediately understandable. I've found that this is particularly helpful for dashboards, where users often don’t have time to hover over each slice and interpret percentages.

The key to label customization in Chartkick lies in leveraging the options available within its API, specifically the `library` option, which allows us to pass configuration details directly to the underlying charting library—which, in this case, is most often Chart.js or Google Charts. Since we're focusing on labels, we'll primarily be dealing with the Chart.js configuration options. Note that the specifics can differ slightly if you’re using Google Charts, but the general principles remain similar.

Let's dive into practical examples.

**Example 1: Basic Label Display with Chart.js**

Let's start with a simple use case: we want to display the name of each slice's category directly on the slice itself. Assume we are working with a Rails application and have data like the following in your view:

```ruby
data = { "Category A" => 30, "Category B" => 45, "Category C" => 25 }
```

Here's how you would configure the chart in your ERB or HAML template:

```erb
<%= pie_chart data, library: {
    plugins: {
        datalabels: {
            display: true,
            color: 'white',
            font: {
                weight: 'bold'
            },
            formatter: function(value, context) {
                return context.chart.data.labels[context.dataIndex];
            }
        }
    }
} %>
```

In this code snippet:

*   We’re setting the `library` option, as discussed.
*   The `plugins` key introduces custom plugins for Chart.js, and we're using `datalabels`. If you don't have the plugin, you will likely get errors in the JavaScript console so be sure to install it via a package manager.
*   `display: true` turns the labels on.
*   `color: 'white'` sets the text color to white to contrast against typical pie chart slice colors.
*   `font: { weight: 'bold' }` ensures the labels stand out further.
*   The crucial part is the `formatter` function. It receives the `value` and `context`. The `context` allows us to access the chart data structure and we are then able to return the actual category label corresponding to each slice using the `context.dataIndex` which maps the current data to the relevant label.

The `datalabels` plugin offers extensive customisation and you should reference it's documentation for the full set of options. I’ve found in practice that getting the labels positioned appropriately can require a bit of experimentation, but this basic setup is often a good starting point.

**Example 2: Displaying Values and Percentages**

Now, let’s say that showing both the raw value *and* the percentage is necessary. This is quite common when you want users to understand not just the relative proportions but also the actual scale of each category. Again, given the same data, we’d modify our configuration within the `library` option:

```erb
<%= pie_chart data, library: {
    plugins: {
      datalabels: {
        display: true,
        color: 'black',
        formatter: function (value, context) {
          var dataset = context.chart.data.datasets[context.datasetIndex];
          var total = dataset.data.reduce(function(previousValue, currentValue){
            return previousValue + currentValue;
          });
          var percentage = (value / total * 100).toFixed(1);
          return context.chart.data.labels[context.dataIndex] + ' (' + value + ', ' + percentage + '%)';
        }
      }
    }
  } %>
```

Here's a breakdown of the changes:

*   We keep the label display turned on.
*   We changed the color to 'black' to illustrate another option
*   The key change is within the `formatter` function. Now, we are accessing the `value` of the slice and using the `datasetIndex` to correctly find the dataset and `reduce` its values to get the total. After determining the total we calculate the percentage and format it to one decimal place and finally return a string including the category name, the raw value and the calculated percentage within parenthesis.

This gives you a more comprehensive view of the data directly on the chart. I've often relied on this approach, especially when dealing with financial or budgetary data where seeing both the numerical values and their proportions is crucial.

**Example 3: Customizing Label Positioning**

Sometimes, the labels might overlap, especially with smaller slices. To prevent that, we need to tweak their positioning. Chart.js provides several options to deal with this. Here is another example of doing this, which also includes using an arrow style for the labels:

```erb
<%= pie_chart data, library: {
    plugins: {
      datalabels: {
        display: true,
        color: 'black',
         anchor: 'end',
        align: 'start',
        offset: 8,
        borderWidth: 2,
        borderColor: 'grey',
        borderRadius: 4,
        textAlign: 'center',
        formatter: function (value, context) {
          var dataset = context.chart.data.datasets[context.datasetIndex];
          var total = dataset.data.reduce(function(previousValue, currentValue){
            return previousValue + currentValue;
          });
          var percentage = (value / total * 100).toFixed(1);
          return context.chart.data.labels[context.dataIndex] + ' (' + percentage + '%)';
        }
      }
    }
  } %>
```

Key changes:

*   `anchor: 'end'` anchors the label to the end of the slice.
*   `align: 'start'` aligns the label to the start of the slice's end point.
*   `offset: 8` pushes the label away from the pie slice by a defined number of pixels
*   `borderWidth: 2` sets a 2 pixel wide border
*   `borderColor: 'grey'` sets the border color to grey
*   `borderRadius: 4` sets a small border radius
*   `textAlign: 'center'` aligns the text within its container to the center
*   The formatter now shows only the percentage alongside the category label.

This configuration makes it more likely that the labels are clear and don't run over each other. Achieving perfect label placement often involves trial and error. I typically fine tune until I achieve a satisfactory and balanced aesthetic.

**Further Reading and Resources**

For a deeper dive into these topics, I’d highly recommend the official Chart.js documentation which is found at [www.chartjs.org](https://www.chartjs.org/). It contains extensive information on the various customization options, plugins, and how they work internally. Be sure to examine the specifics of their datalabels plugin at [https://chartjs-plugin-datalabels.netlify.app/](https://chartjs-plugin-datalabels.netlify.app/) which is what we used for these examples. Additionally, the book *Interactive Data Visualization for the Web* by Scott Murray is an excellent resource to understand the fundamentals and how to apply them in various scenarios. Finally, understanding the basics of JavaScript will enable you to debug these issues, so taking some time to familiarize yourself with it can help.

By understanding Chartkick's API and the underlying libraries, you can achieve a far more granular level of control and create visualizations that are not only informative but also highly polished and user-friendly. Remember, data visualization is about effective communication, and label customisation is crucial to achieve that goal.
