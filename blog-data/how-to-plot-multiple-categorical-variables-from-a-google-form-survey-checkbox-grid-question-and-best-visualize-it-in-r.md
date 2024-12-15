---
title: "How to Plot Multiple Categorical Variables From a Google Form Survey Checkbox Grid Question and Best Visualize It in R?"
date: "2024-12-15"
id: "how-to-plot-multiple-categorical-variables-from-a-google-form-survey-checkbox-grid-question-and-best-visualize-it-in-r"
---

alright, so you've got a google forms survey with a checkbox grid, and now you're staring at a pile of data wondering how to make sense of it all in r. been there, done that. i remember back in my early days of data exploration, i had a similar challenge with a feedback survey about some internal software we were developing. the responses were, let's say, enthusiastically multi-faceted. trying to extract insights from that matrix of checkboxes felt like trying to decode a secret language written in tic-tac-toe. anyway, it's doable, and here's how i usually approach this.

the first hurdle is getting that google sheets data properly into r. assuming you've already exported the results and have it as a csv, the typical approach will work:

```r
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)

# replace with your path
survey_data <- read_csv("path/to/your/survey_data.csv")

# let's take a quick look
head(survey_data)
```

this loads the necessary libraries and reads your data. important point here: make sure your paths are correctly specified. the `head()` function is a sanity check, and i'd say it's always a good starting point to visualize the data. you can also use `str()` for a more detailed look into the data types in each column. that will save you a lot of trouble down the line.

now, the real fun begins. your checkbox grid is likely represented with multiple columns, each column representing a choice within a row. for example, say your grid had questions about which features users utilized regularly, with "feature a," "feature b," and "feature c" options, with rows of "daily," "weekly," "monthly," and "rarely." your google sheet is probably formatted with columns like "daily - feature a", "daily - feature b", "weekly - feature a," and so on. each cell will either be a 0 or 1 (or blank if not checked in some cases). we need to reshape this from "wide" to "long" format, a task tidyverse excels at. i've spent considerable time crafting code to convert wide data tables into long tables, it has helped me a lot in data analysis.

```r
survey_long <- survey_data %>%
  pivot_longer(
    cols = starts_with(c("daily", "weekly", "monthly", "rarely")),
    names_to = c("frequency", "feature"),
    names_sep = " - ",
    values_to = "selected",
    values_drop_na = TRUE
  ) %>%
  filter(selected == 1)

# let's have a look to the transformed data
head(survey_long)
```

here, `pivot_longer` does all the magic. we specify the columns we want to melt, what to name the resulting columns, and how to separate the names. remember to adjust the `starts_with()` parameters and `names_sep` argument to match your specific column names. then we can use `filter` to keep only those rows where something was actually selected. it’s like saying "show me only the times somebody actually clicked something on the form." a quick peek with `head(survey_long)` helps to verify it did what we wanted. you might end up with an error if the names are not consistently separated and it can cause a lot of confusion so, make sure it is the expected format.

now, for the visualization. a good approach is a grouped bar chart, which allows us to see both the frequency of usage and the features.

```r
ggplot(survey_long, aes(x = feature, fill = frequency)) +
  geom_bar(position = "dodge") +
  labs(title = "Feature Usage Frequency",
       x = "Feature",
       y = "Number of Users",
       fill = "Frequency of Use") +
   theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

this creates a plot where each feature has bars grouped by the usage frequency (daily, weekly, monthly, and rarely). you can tweak this to your heart’s content: change the color schemes, add percentages, use different plots (e.g. stacked bar), etc. i tend to fiddle with `ggplot2` parameters until the plot tells me exactly what i need to know and the look is as simple as possible, never adding unnecessary complex graphs to the report.

if you're dealing with very many variables, heatmaps may be a better option.

```r
survey_matrix <- survey_long %>%
    group_by(feature, frequency) %>%
    summarise(n = n()) %>%
    pivot_wider(names_from = frequency, values_from = n, values_fill = 0) %>%
    column_to_rownames(var = "feature")

heatmap(as.matrix(survey_matrix),
        scale="none",
        Rowv = NA,
        Colv=NA,
        main = "Heatmap of Feature Usage Frequency",
        xlab="Frequency",
        ylab = "Features",
        margins=c(10,10)
)
```

the above code snippet groups and summarizes counts, pivots again to create a frequency matrix, then plots it as a simple heatmap. `scale="none"` makes the colors represent the counts directly, without any normalization. i prefer to use this approach when the data isn't too large, for the user to easily interpret the numbers with colours. if you are dealing with more complicated data, perhaps you could consider a more advanced approach, but if the number of features is less than 20, i think this would do the trick nicely.

the key here is the transformations to your data before plotting. and this is something i wish i had fully grasped earlier in my career. i used to spend hours trying to tweak graphs from messy data instead of investing in time on correct data transformation. so my recommendation would be to always clean your data first.

a few closing tips: check out books like “r for data science” by hadley wickham and garrett grolemund or “ggplot2: elegant graphics for data analysis” also by hadley wickham. these books were total game changers for me. they explain the underlying data principles and the `tidyverse` logic clearly. understanding the grammar of graphics and data manipulation pipelines are the best tools you can have in your arsenal. and also if you ever hear somebody saying that the x axis is horizontal you can say "yeah, it’s like a horizontal line" (it’s a joke, you get it).

remember, data analysis is an iterative process. experiment with different visualizations, and always strive for clarity in your presentation. sometimes, the simplest plot tells the most compelling story, and if you are stuck, just come back to stackoverflow. and, like always, make sure you understand what your code is doing. good luck and happy plotting.
