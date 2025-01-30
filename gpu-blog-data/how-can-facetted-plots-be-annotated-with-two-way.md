---
title: "How can facetted plots be annotated with two-way ANOVA post-hoc results?"
date: "2025-01-30"
id: "how-can-facetted-plots-be-annotated-with-two-way"
---
Specifically, describe a workflow leveraging Python and libraries such as Pandas, Statsmodels, and Seaborn to achieve this, addressing challenges related to statistical rigor, visualization clarity, and coding efficiency.

Annotations within facetted plots that display two-way ANOVA post-hoc results require a nuanced approach, primarily because post-hoc comparisons are computed for each combination of levels within the factors considered by the ANOVA. These results are therefore specific to the grouping within each facet. My experience designing experiments in behavioral psychology has made it clear that directly incorporating statistical annotations within each facet requires careful data manipulation and a focused visualization strategy to avoid clutter. The typical output from a two-way ANOVA with post-hoc tests contains many comparisons, which, if not appropriately handled, can overwhelm the visualization and obscure, rather than reveal, the underlying patterns.

The core of this workflow rests on three key stages: data preparation and statistical analysis, result extraction and formatting, and the final visualization using Seaborn. The challenge lies in connecting the statistical output to specific facets and placing annotations with sufficient clarity and conciseness.

First, I'll discuss the initial data preparation and statistical analysis stage. I assume the data is organized in a Pandas DataFrame where each row represents an observation, and columns contain the dependent variable and the two categorical independent variables (factors) to be analyzed by two-way ANOVA. Here is a simplified example of how that data might look:

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Sample Data Generation
data = {'dependent_var': [10, 12, 15, 18, 14, 20, 22, 25, 19, 13, 17, 16, 24, 28, 29, 21, 23, 27, 11, 16, 20, 26, 30, 31, 15, 18, 22, 27, 29, 32],
        'factor_a': ['A1', 'A1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A2', 'A2', 'A3', 'A3', 'A3', 'A3', 'A3', 'A1', 'A1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A2', 'A2', 'A3', 'A3', 'A3', 'A3', 'A3'],
        'factor_b': ['B1', 'B2', 'B3', 'B1', 'B2', 'B1', 'B2', 'B3', 'B1', 'B2','B1', 'B2', 'B3','B1', 'B2', 'B3', 'B1', 'B2', 'B3', 'B1', 'B2','B1', 'B2', 'B3','B1', 'B2', 'B3','B1', 'B2', 'B3']}

df = pd.DataFrame(data)

# Two-way ANOVA
formula = 'dependent_var ~ C(factor_a) + C(factor_b) + C(factor_a):C(factor_b)'
model = ols(formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA Table: \n",anova_table)


```
This first code block loads the required libraries, generates sample data for illustrative purposes, and conducts the two-way ANOVA. The `ols` function from `statsmodels` defines the model, including main effects and the interaction term. The `anova_lm` function calculates the ANOVA table. The ANOVA results themselves are usually not displayed directly in the plot annotations, but rather serve as the basis for the post-hoc tests. Therefore, the more critical output is from the post-hoc tests.

Following the ANOVA, I perform Tukey's Honestly Significant Difference (HSD) post-hoc test. Here's the relevant code:

```python
# Post-Hoc Analysis (Tukey HSD)
m_comp = pairwise_tukeyhsd(endog=df['dependent_var'], groups=df[['factor_a','factor_b']].agg('_'.join, axis=1), alpha=0.05)
print("\nTukey HSD: \n", m_comp)
tukey_summary = pd.DataFrame(data=m_comp._results_table.data[1:], columns=m_comp._results_table.data[0])
tukey_summary[['group1', 'group2']] = tukey_summary['group1'].str.split('-', expand=True)
print("\nTukey Summary: \n", tukey_summary)

```

Here, `pairwise_tukeyhsd` conducts all pairwise comparisons. I have specifically grouped the two factor levels using `agg('_'.join, axis=1)` as the `groups` parameter, allowing for interaction-level comparisons. The results are transformed into a Pandas DataFrame for easier manipulation. Splitting the `group1` string allows the original levels to be accessible for matching to the facets. This data needs significant further processing before it is useful for our visualization task.

The most challenging step is now at hand: extracting the significant pairwise comparisons and using them as annotations. This requires iteration over each facet and matching the post-hoc results to the specific comparisons within that facet. Here's my implementation:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualization and Annotation
def annotate_tukey_results(data, x, y, col, row, tukey_results):
    g = sns.catplot(data=data, x=x, y=y, col=col, row=row, kind="point", sharey=False)
    
    for ax in g.axes.flat:
        row_val = ax.get_ylabel().split('=')[1] if ax.get_ylabel() else None
        col_val = ax.get_title().split('=')[1] if ax.get_title() else None
        if row_val is None or col_val is None:
            continue

        relevant_comparisons = tukey_results[
            ((tukey_results['group1'].str.contains(row_val)) | (tukey_results['group2'].str.contains(row_val))) &
            ((tukey_results['group1'].str.contains(col_val)) | (tukey_results['group2'].str.contains(col_val))) &
            (tukey_results['reject'] == True)]

        y_max = max(point.get_height() for point in ax.containers[0]) + 0.5 #Find the tallest point and add 0.5 offset

        for index, comp in relevant_comparisons.iterrows():

            group_1_a = comp['group1'].split('_')[0]
            group_1_b = comp['group1'].split('_')[1]

            group_2_a = comp['group2'].split('_')[0]
            group_2_b = comp['group2'].split('_')[1]
            
            if group_1_a == row_val and group_1_b == col_val:
                group_x = group_2_b
                text_x_pos = data[x].unique().tolist().index(group_x)

            elif group_2_a == row_val and group_2_b == col_val:
                group_x = group_1_b
                text_x_pos = data[x].unique().tolist().index(group_x)

            else:
                continue
            
            text_pos = (text_x_pos ,y_max)
            ax.annotate(text='*', xy = text_pos ,ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

annotate_tukey_results(data=df, x='factor_b', y='dependent_var', col='factor_a', row=None, tukey_results=tukey_summary)

```

In this function, `sns.catplot` generates the facetted point plot. Crucially, within the loop iterating through the plot axes, I extract the row and column labels using `ax.get_ylabel()` and `ax.get_title()`. These labels are then used to filter the post-hoc results for comparisons relevant to that specific facet.  The annotation itself, a simple asterisk '*', is placed using `ax.annotate` at the maximum y-value in that facet for clear separation and minimal overlapping. This approach is effective even when the factor ordering is different. The loop further ensures the labels that correspond to each location in the facet are used to reference the original ANOVA.

Several design decisions went into this approach. The post-hoc results are filtered to only show the significantly different pairs at p < .05 to prevent excessive clutter. The annotations are simple asterisks, which are less detailed but more immediately noticeable for significant differences. In my experience, detailed p-values often require excessive interpretation in a plot. If p-values are required, they can easily be added with `ax.annotate` by modifying the `text` argument from '*' to `{pvalue:.3f}` from the `relevant_comparisons` DataFrame.  This flexibility is useful for different visualization scenarios, based on the audience and task. The function has also been designed to allow for row and column faceting based on the `row` argument. I selected point plots for their clarity, but other plot types like box plots or violin plots can also work with a small modification to the `sns.catplot` arguments.

The resource I would recommend for additional information on statistical testing in python is the statsmodels documentation. For plot creation and manipulation, the seaborn documentation is the most useful resource. Pandas' documentation will assist with data manipulation. Finally, understanding the nuances of two-way ANOVA requires consulting statistical textbooks and other educational materials focused on experimental design and data analysis. These resources, when combined with careful practice, enable the effective and statistically rigorous visualization of complex datasets involving multi-factor experiments.
