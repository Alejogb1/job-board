---
title: "How do I display optimization results in AnyLogic?"
date: "2025-01-30"
id: "how-do-i-display-optimization-results-in-anylogic"
---
Displaying optimization results effectively in AnyLogic requires careful consideration of both data presentation and user interaction. Over the years, I've encountered scenarios where haphazard output made analysis nearly impossible; a well-structured approach, however, significantly clarifies the optimizer's findings. The core challenge lies in transforming raw numeric output, often a series of objective function values and parameter sets, into a digestible and actionable format.

My approach involves a combination of tabular displays, charting, and interactive elements. The specific techniques I utilize depend on the nature of the optimization problem and the stakeholders’ needs. For instance, when dealing with a process optimization for a manufacturing line, clear visualization of production throughput against machine configurations is paramount. Conversely, a supply chain network optimization might require a map-based representation.

**Explanation**

The process begins with capturing the desired outputs from the optimization experiment. AnyLogic's `OptimizationResult` object provides all the necessary data, including the objective function values for each iteration, the corresponding parameter sets, and execution time. The key is to extract this information programmatically during or after the experiment, not attempting manual data manipulation. I generally prefer to gather and process this data post-experiment, giving flexibility to generate different types of visualizations without rerunning the simulation.

Once captured, the data transformation phase is critical. The `OptimizationResult` object stores parameter sets as an array of doubles; this needs to be mapped back to the meaningful parameters in your model. For example, an optimizer might adjust `machineSpeed` which, while internally a double, represents an actual property in my model. I usually implement custom methods to handle this mapping and data preparation. Furthermore, for better analysis, I calculate derived metrics, such as the percentage change in objective function value relative to the initial configuration, or the total cost associated with each parameter setting.

The final stage is the actual visualization. AnyLogic’s built-in presentation capabilities are adequate for most use cases, but often require custom code for enhanced clarity. Tables are suitable for displaying detailed parameter settings and corresponding objective function values in a structured way. I often leverage the `Table` element, populating it with data from the processed output arrays. Charts, particularly line and scatter charts, are excellent for visualizing the progress of the optimization over time or for examining the correlation between different parameters and the objective function. Interactive elements, such as sliders or drop-down menus linked to chart data points, enable users to explore the results in detail and compare different scenarios.

**Code Examples**

These examples will illustrate how I structure code in my own projects. The examples assume the existence of an experiment called "myOptimizationExperiment" and that its optimization result is accessible through `myOptimizationExperiment.getOptimizationResult()`.

**Example 1: Populating a Table**

This snippet illustrates extracting parameter sets and objective function values and displaying them in a table called `resultsTable`.

```java
// In the experiment's 'On completion' action
OptimizationResult optResult = myOptimizationExperiment.getOptimizationResult();
if (optResult == null) return;  // Handle case where no optimization was performed

int iterations = optResult.getNumberOfIterations();
double[][] parameterSets = optResult.getParameterSets();
double[] objectiveValues = optResult.getObjectiveValues();

resultsTable.clear();
resultsTable.addColumn("Iteration", Integer.class);
resultsTable.addColumn("Parameter 1", Double.class);  // Map parameter names to columns
resultsTable.addColumn("Parameter 2", Double.class);
resultsTable.addColumn("Objective Value", Double.class);

for (int i = 0; i < iterations; i++) {
    double[] parameters = parameterSets[i];
    resultsTable.addRow(i, parameters[0], parameters[1], objectiveValues[i]); //Assuming 2 parameters
}

```

*   **Commentary:** This code first retrieves the optimization results and ensures they are not null. It then obtains the number of iterations, parameter sets, and corresponding objective values. It clears the existing table and adds columns based on the model’s parameters. Finally, it iterates through the results, adding each iteration's data to a new row in the `resultsTable` in a structured manner. The assumption here is that we have exactly two parameters which are mapped to columns "Parameter 1" and "Parameter 2". In practice, these column headers would be set based on what the model is optimizing.

**Example 2: Creating a Line Chart**

This example generates a line chart that shows the objective function value across optimization iterations.

```java
// In the experiment's 'On completion' action, assuming 'objectiveChart' is a chart element
OptimizationResult optResult = myOptimizationExperiment.getOptimizationResult();
if (optResult == null) return;

int iterations = optResult.getNumberOfIterations();
double[] objectiveValues = optResult.getObjectiveValues();

objectiveChart.clearSeries();
Series series = objectiveChart.addSeries("Objective Function");
for (int i = 0; i < iterations; i++) {
    series.add(i, objectiveValues[i]);
}
objectiveChart.updateData(); // Force chart to update
```

*   **Commentary:** Here, we fetch the optimization results as before. We clear any previous data on the existing chart element called `objectiveChart`.  A new data series is created, to which data is iteratively added by using the iteration number as the X value and objective function value as the Y value. Finally the `updateData()` method is used to force chart update. This provides a visual representation of the optimization's progress toward a solution, highlighting trends in the objective function.

**Example 3: Parameter Mapping and Dynamic Updates**

This is an advanced example involving dynamic parameter interpretation and dynamic visualization. It uses parameter values to update a text element on the screen called `parameterDisplay` which updates dynamically when a user selects different rows in the resultsTable from example 1. This would be useful when trying to compare configurations by hand after an optimization run.

```java
// This code goes in the experiment's 'On selection' action for 'resultsTable'.
TableRow row = resultsTable.getSelectedRow();
if (row == null) return;

double param1 = (Double) row.get(1); // Get values from table, assuming parameter columns are in indices 1 and 2
double param2 = (Double) row.get(2);

String textToDisplay = "Current Parameter Settings:\n" +
                       "Parameter 1: " + param1 + "\n" +
                       "Parameter 2: " + param2;

parameterDisplay.setText(textToDisplay);
```

*   **Commentary:** This code extracts parameter values from a selected row in the table created in Example 1. It then maps these values to actual parameter labels and dynamically sets a text element called `parameterDisplay` in the model. It provides a way to correlate parameter settings with optimization results from the table in a more readable format. It showcases the ability to create user interaction for exploring the optimization results further. We assume the columns holding the parameter values in the resultsTable are in indices 1 and 2.

**Resource Recommendations**

For further exploration of these techniques, consider consulting the following resources:

1.  **AnyLogic Help Documentation:** The built-in help system is an invaluable resource. It covers the specifics of data structures, APIs, and presentation elements.

2.  **AnyLogic Examples:** Examine the pre-built example models. They offer practical demonstrations of optimization, data handling, and user interface creation. Pay special attention to the 'Optimization Examples' and the models that employ charts and tables.

3.  **AnyLogic User Forums and Communities:** Engage with other modelers to gain diverse perspectives and learn from their experiences. Many users openly share their code and solutions.

In conclusion, displaying optimization results is not just about presenting numbers; it is about crafting a narrative from data. By carefully capturing, processing, and visualizing the outputs, I enable myself, and my stakeholders, to understand the insights generated through complex simulations and take informed action. My approach stresses the importance of code-driven data manipulation and leveraging AnyLogic's existing functionalities, along with a dose of custom programming.
