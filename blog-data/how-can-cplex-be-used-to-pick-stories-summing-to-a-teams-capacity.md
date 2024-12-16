---
title: "How can Cplex be used to pick stories summing to a team's capacity?"
date: "2024-12-16"
id: "how-can-cplex-be-used-to-pick-stories-summing-to-a-teams-capacity"
---

Alright, let’s tackle this problem of allocating stories to a team's capacity using Cplex. I remember a particularly challenging project a few years back where we needed to optimize sprint planning across multiple teams, each with varying capacities and story sizes—it’s a problem that crops up more often than you'd think. The manual approach was a nightmare, prone to errors and inefficiencies. This led me to explore using optimization solvers, and Cplex, given its robust capabilities, became our tool of choice.

The core issue, as you've framed it, is selecting a subset of user stories (or tasks, if you prefer that term) such that their total estimated effort does not exceed the team’s available capacity. This falls squarely into the domain of combinatorial optimization, specifically a variation of the knapsack problem.

Here's how I approach it using Cplex, focusing on clarity and practical application:

**Formulating the Problem:**

We essentially model this problem as a binary integer program. Let’s break down the components:

*   **Sets:** We have a set *S* representing all the available stories. For each story *i* in *S*, we have a corresponding effort estimate *e<sub>i</sub>*.
*   **Parameters:** We have a single parameter *C*, which denotes the total capacity of the team.
*   **Decision Variables:** For each story *i*, we introduce a binary variable *x<sub>i</sub>*. If *x<sub>i</sub>* = 1, it indicates that the story is selected for the sprint; otherwise, it’s 0.
*   **Objective Function:** Our goal is to maximize the total effort of the selected stories, assuming we want to fully utilize the available capacity. The objective function becomes: maximize ∑<sub>i∈S</sub> *e<sub>i</sub>* *x<sub>i</sub>*. This will also effectively favor larger story picks when given an option.
*   **Constraints:** The critical constraint is ensuring that the sum of the effort of the selected stories does not exceed the team's capacity: ∑<sub>i∈S</sub> *e<sub>i</sub>* *x<sub>i</sub>* ≤ *C*.

**Implementation using Python and Cplex (using the `docplex` library):**

I'll walk through three slightly different examples using Python with `docplex`, which is Cplex's Python API:

**Example 1: Basic Capacity Maximization**

```python
from docplex.mp.model import Model

def solve_capacity_problem_basic(stories, capacity):
    mdl = Model(name='story_selection')

    # Decision variables
    x = mdl.binary_var_dict(stories, name="select_story")

    # Objective: maximize total story effort
    mdl.maximize(mdl.sum(stories[i] * x[i] for i in stories))

    # Constraint: total effort must be <= capacity
    mdl.add_constraint(mdl.sum(stories[i] * x[i] for i in stories) <= capacity)

    # Solve the model
    solution = mdl.solve()

    if solution:
        selected_stories = [story for story in stories if x[story].solution_value > 0.5]
        selected_effort = sum([stories[story] for story in selected_stories])
        return selected_stories, selected_effort
    else:
      return [], 0


# Example Usage
stories_effort = {
    "story1": 5,
    "story2": 8,
    "story3": 3,
    "story4": 6,
    "story5": 9
}
team_capacity = 20

selected, total_effort = solve_capacity_problem_basic(stories_effort, team_capacity)
print(f"Selected stories: {selected}")
print(f"Total effort: {total_effort}")
```

This code directly implements the formulation I described earlier. The `docplex` library abstracts away the complexity of building and solving the optimization model.

**Example 2: Adding Story Priorities**

Sometimes, stories have differing priorities. We can easily incorporate this by adding a priority weight to each story within the objective function.

```python
from docplex.mp.model import Model

def solve_capacity_problem_prioritized(stories, capacity, priorities):
    mdl = Model(name='story_selection_prioritized')

    # Decision variables
    x = mdl.binary_var_dict(stories, name="select_story")

    # Objective: maximize prioritized effort
    mdl.maximize(mdl.sum(stories[i] * priorities[i] * x[i] for i in stories))

    # Constraint: total effort must be <= capacity
    mdl.add_constraint(mdl.sum(stories[i] * x[i] for i in stories) <= capacity)

    # Solve the model
    solution = mdl.solve()

    if solution:
      selected_stories = [story for story in stories if x[story].solution_value > 0.5]
      selected_effort = sum([stories[story] for story in selected_stories])
      return selected_stories, selected_effort
    else:
      return [], 0

# Example Usage
stories_effort = {
    "story1": 5,
    "story2": 8,
    "story3": 3,
    "story4": 6,
    "story5": 9
}
story_priorities = {
    "story1": 1,
    "story2": 3,
    "story3": 2,
    "story4": 1,
    "story5": 2
}
team_capacity = 20

selected, total_effort = solve_capacity_problem_prioritized(stories_effort, team_capacity, story_priorities)
print(f"Selected stories: {selected}")
print(f"Total effort: {total_effort}")
```

Here, we’ve simply added a `priorities` dictionary and incorporated it into the objective function. Stories with higher priorities are now more likely to be chosen.

**Example 3: Handling Dependencies (Simple Example)**

Let's consider a basic dependency example—story 'B' cannot be selected without story 'A'. We need to incorporate another constraint to represent these dependencies.

```python
from docplex.mp.model import Model

def solve_capacity_problem_with_dependency(stories, capacity, dependency):
    mdl = Model(name='story_selection_with_dependency')

    # Decision variables
    x = mdl.binary_var_dict(stories, name="select_story")

    # Objective: maximize total effort
    mdl.maximize(mdl.sum(stories[i] * x[i] for i in stories))

    # Constraint: total effort must be <= capacity
    mdl.add_constraint(mdl.sum(stories[i] * x[i] for i in stories) <= capacity)

    # Constraint: dependency (if selecting "B", select "A")
    if "A" in dependency and "B" in dependency:
      mdl.add_constraint(x["B"] <= x["A"])

    # Solve the model
    solution = mdl.solve()

    if solution:
      selected_stories = [story for story in stories if x[story].solution_value > 0.5]
      selected_effort = sum([stories[story] for story in selected_stories])
      return selected_stories, selected_effort
    else:
        return [], 0


# Example Usage
stories_effort = {
    "storyA": 5,
    "storyB": 8,
    "storyC": 3,
    "storyD": 6
}
team_capacity = 20
story_dependency = {
    "A": "B"
}

selected, total_effort = solve_capacity_problem_with_dependency(stories_effort, team_capacity, story_dependency)
print(f"Selected stories: {selected}")
print(f"Total effort: {total_effort}")
```

This snippet introduces a conditional dependency. We've modeled a very simple 'if B is selected, then A must be selected' logic. More complex dependencies can be implemented by expanding this logic into additional constraints.

**Further Considerations and Resources:**

*   **Model Complexity:** As you've seen, adding constraints can increase the complexity, but it also models the real world more accurately.
*   **Solver Performance:** Cplex is very efficient, but depending on the scale of your problem, the time to find an optimal solution can increase. For large-scale problems, techniques like column generation and heuristics might become necessary.
*   **Data Handling:** In my previous work, we often integrated these optimizers with a data pipeline, where story data was pulled from our project management software. This meant regular data transformation was involved before being usable.
* **Constraint Programming:** It's worth exploring constraint programming techniques with tools like Cplex if you have complex logical constraints.
*   **Resources:**
    *   For a solid grounding in optimization, I recommend "Introduction to Operations Research" by Frederick S. Hillier and Gerald J. Lieberman. It's a classic for a reason.
    *   To go deeper into integer programming, consider “Integer Programming” by Laurence A. Wolsey. It is a more advanced reference but very worthwhile.
    *   For specifics on the Cplex API, the official Cplex documentation is the best source; however, also look into the `docplex` library documentation as that will focus on the Python library.

The beauty of using optimization solvers like Cplex lies in their adaptability. I've presented a fairly direct method of solving the problem, but it provides a powerful foundation. Remember, the key is to accurately formulate your problem, map real-world rules to model constraints, and utilize appropriate tool sets. From my experience, this will prove vastly more effective and reliable than manually trying to allocate stories.
