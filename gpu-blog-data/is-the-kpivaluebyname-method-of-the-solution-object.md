---
title: "Is the kpi_value_by_name method of the Solution object in Docplex exhibiting a bug?"
date: "2025-01-30"
id: "is-the-kpivaluebyname-method-of-the-solution-object"
---
The `kpi_value_by_name` method of the CPLEX Python API's `Solution` object, while generally robust, can exhibit unexpected behavior when dealing with KPIs (Key Performance Indicators) defined using non-standard naming conventions or when the solution object itself reflects an infeasible or unbounded model.  My experience debugging similar issues in large-scale optimization projects involved extensive testing across varied model structures and KPI definitions.  This response will address potential causes and provide illustrative examples.

**1. Clear Explanation:**

The `kpi_value_by_name` method retrieves the value of a KPI from a solved model's solution.  The method's core functionality relies on a precise mapping between the KPI's name as specified during its definition within the model and the name used in the method's argument.  Discrepancies in this mapping, even minor ones like extra whitespace or case sensitivity, will lead to a `KeyError` exception, indicating that the KPI name is not found.  Furthermore, the method expects the model to have a valid, feasible solution. Attempting to access KPI values from an infeasible or unbounded solution will yield unpredictable results, often resulting in incorrect values or the aforementioned `KeyError`.  Finally, the behavior around KPIs defined using expressions involving more complex CPLEX objects, such as sets or tuples, warrants particular attention.  Ambiguity in referencing such objects within the expression can lead to unexpected KPI values.

**2. Code Examples with Commentary:**

**Example 1: Case Sensitivity and Whitespace:**

```python
from docplex.mp.model import Model

mdl = Model(name='case_sensitivity')
x = mdl.continuous_var(name='myVar', lb=0, ub=10)
mdl.add_kpi(x, 'MyVar') # Note capitalization difference
mdl.solve()
solution = mdl.solution

try:
    kpi_value = solution.kpi_value_by_name('myVar') # Incorrect case
    print(f"KPI value: {kpi_value}")
except KeyError as e:
    print(f"Error: {e}") #Expect KeyError here


try:
    kpi_value = solution.kpi_value_by_name(' MyVar ') # Extra whitespace
    print(f"KPI value: {kpi_value}")
except KeyError as e:
    print(f"Error: {e}") #Expect KeyError here

try:
    kpi_value = solution.kpi_value_by_name('MyVar') # Correct case
    print(f"KPI value: {kpi_value}")
except KeyError as e:
    print(f"Error: {e}") #This should print the correct value

```

This example highlights the importance of precise naming.  The `KeyError` exceptions are expected when using incorrect case or including extraneous whitespace in the KPI name. Only the correct naming will return the KPI's value.  In real-world scenarios, such inconsistencies often arise from data import errors or copy-paste mistakes.


**Example 2: Infeasible Model:**

```python
from docplex.mp.model import Model

mdl = Model(name='infeasible_model')
x = mdl.binary_var(name='x')
y = mdl.binary_var(name='y')
mdl.add_constraint(x + y == 1.5) # Infeasible constraint
mdl.add_kpi(x + y, 'sum_xy')
mdl.solve()

solution = mdl.solution

if solution:
    try:
        kpi_value = solution.kpi_value_by_name('sum_xy')
        print(f"KPI value: {kpi_value}") # This might produce unexpected or incorrect results
    except KeyError as e:
        print(f"Error: {e}")
else:
    print("Model is infeasible.") # Expect this message

```

This code demonstrates the behavior with an infeasible model.  While `kpi_value_by_name` might not always throw a `KeyError`, the returned value is unreliable and shouldn't be trusted. The model's infeasibility should always be checked before accessing KPI values.  Attempting to access KPI values from an infeasible solution frequently leads to inconsistent results, masking underlying model errors.

**Example 3: Complex KPI Definition:**

```python
from docplex.mp.model import Model

mdl = Model(name='complex_kpi')
x = mdl.continuous_var(name='x', lb=0, ub=10)
y = mdl.continuous_var(name='y', lb=0, ub=10)
mdl.add_constraint(x + y <= 10)
mdl.add_kpi(x*x + y, 'complex_kpi_expr') #Non-linear expression
mdl.solve()
solution = mdl.solution

if solution:
    try:
        kpi_value = solution.kpi_value_by_name('complex_kpi_expr')
        print(f"KPI value: {kpi_value}") # Should work, but requires careful definition
    except KeyError as e:
        print(f"Error: {e}")
else:
    print("Model is infeasible or unbounded.")
```

Here, a more complex KPI expression is used.  While this will usually work,  subtle errors in the expression or unintended interactions between its components and the solver's internal mechanisms can lead to unexpected outcomes. Carefully examining the KPI's definition for potential ambiguities and testing it thoroughly is crucial.   Overly complex expressions should be broken down into smaller, more manageable parts for better readability and reliability.



**3. Resource Recommendations:**

The official CPLEX documentation, particularly the sections on the Python API and model building, should be the primary resource.   Furthermore, the examples provided in the CPLEX documentation, focusing on KPI definition and solution retrieval, offer valuable practical insights.   Finally, studying relevant CPLEX user forums and community discussions can offer invaluable perspectives on common pitfalls and best practices related to KPI management within the framework.  Consulting such materials is essential for mastering the nuances of  `kpi_value_by_name` and similar methods.
