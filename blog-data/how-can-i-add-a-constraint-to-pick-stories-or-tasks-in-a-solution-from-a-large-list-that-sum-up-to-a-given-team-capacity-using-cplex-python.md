---
title: "How can I add a constraint to pick stories (or tasks) in a solution from a large list that sum up to a given team capacity using Cplex Python?"
date: "2024-12-23"
id: "how-can-i-add-a-constraint-to-pick-stories-or-tasks-in-a-solution-from-a-large-list-that-sum-up-to-a-given-team-capacity-using-cplex-python"
---

Okay, let's talk about optimizing task selection with Cplex when you're aiming for a specific team capacity. This is a problem I've faced numerous times, typically during sprint planning for large, complex software projects where resource allocation is critical. I remember one particularly gnarly scenario at a previous company where we had to juggle hundreds of stories, each with varying effort estimations, and our team capacity seemed to fluctuate weekly. It's a classic knapsack problem, but with a focus on integer programming, which is where CPLEX shines.

The core challenge here is formulating the problem so CPLEX can efficiently find the optimal combination of stories that fit within your capacity constraint. We’re looking to maximize the 'value' of selected tasks – often this correlates with completing the most important stories – while strictly adhering to the total effort limit (your team capacity).

Here’s how we can approach it using the CPLEX Python API:

Firstly, you need to frame the problem as a mathematical model. You'll have:

*   **Decision variables:** Boolean values representing whether a specific story is included in the solution (1) or not (0).
*   **Objective function:** To maximize the total 'value' of the included stories. This 'value' can be the story's priority or business value.
*   **Constraints:** The primary constraint is that the sum of the story's effort estimations must not exceed the team capacity.

Let's put this into code:

```python
from docplex.mp.model import Model

def solve_capacity_constraint(stories, team_capacity):
  """Solves the story selection problem using CPLEX.

  Args:
    stories: A list of tuples, where each tuple is (story_id, effort, value).
    team_capacity: The total capacity of the team.

  Returns:
    A tuple containing:
      - A list of story ids included in the optimal solution, or None if no solution
        was found.
      - The maximum total value of the selected stories.
  """

  mdl = Model(name="story_selection")

  # Decision variables: 1 if story is selected, 0 otherwise
  x = mdl.binary_var_list(len(stories), name="story_selected")

  # Objective function: Maximize total value
  mdl.maximize(mdl.sum(stories[i][2] * x[i] for i in range(len(stories))))

  # Constraint: Total effort should not exceed team_capacity
  mdl.add_constraint(mdl.sum(stories[i][1] * x[i] for i in range(len(stories))) <= team_capacity)

  solution = mdl.solve()

  if solution:
    selected_stories = [stories[i][0] for i in range(len(stories)) if solution.get_value(x[i]) > 0.5]
    total_value = solution.get_objective_value()
    return selected_stories, total_value
  else:
    return None, 0


# Example usage
stories_data = [
    ("story1", 5, 10),  # (story_id, effort, value)
    ("story2", 3, 7),
    ("story3", 8, 15),
    ("story4", 2, 5),
    ("story5", 6, 12),
    ("story6", 1, 3)
]
team_capacity = 12

selected, total_value = solve_capacity_constraint(stories_data, team_capacity)


if selected:
    print("Selected stories:", selected)
    print("Total value:", total_value)
else:
    print("No solution found.")
```

In this snippet, I've defined a function, `solve_capacity_constraint`, that takes a list of stories (represented as tuples containing their id, effort, and value) and the team's capacity. It builds the CPLEX model using `docplex`, adds the constraints, and optimizes it. The function then returns the IDs of the selected stories and the total value of those stories. If no feasible solution is found, it returns `None`.

Now, one of the practical challenges I’ve observed is the complexity of real-world constraints. Sometimes, we can have story dependencies or task dependencies. Let's say a specific story can't be included unless another story is also selected. This would require additional constraints in your CPLEX model:

```python
def solve_capacity_with_dependency(stories, team_capacity, dependencies):
  """Solves the story selection problem with dependency using CPLEX.

    Args:
    stories: A list of tuples, where each tuple is (story_id, effort, value).
    team_capacity: The total capacity of the team.
    dependencies: A dictionary where key is dependant story id and value is the depended story id.
  Returns:
    A tuple containing:
      - A list of story ids included in the optimal solution, or None if no solution
        was found.
      - The maximum total value of the selected stories.
  """
  mdl = Model(name="story_selection_with_dependency")

  x = mdl.binary_var_list(len(stories), name="story_selected")

  mdl.maximize(mdl.sum(stories[i][2] * x[i] for i in range(len(stories))))
  mdl.add_constraint(mdl.sum(stories[i][1] * x[i] for i in range(len(stories))) <= team_capacity)

  # Add dependency constraints
  story_id_to_index = {stories[i][0]: i for i in range(len(stories))}

  for dependent_story, depended_story in dependencies.items():
      dependent_index = story_id_to_index[dependent_story]
      depended_index = story_id_to_index[depended_story]
      mdl.add_constraint(x[dependent_index] <= x[depended_index])

  solution = mdl.solve()

  if solution:
    selected_stories = [stories[i][0] for i in range(len(stories)) if solution.get_value(x[i]) > 0.5]
    total_value = solution.get_objective_value()
    return selected_stories, total_value
  else:
    return None, 0

# Example usage with dependencies:
stories_data_with_dependencies = [
    ("story1", 5, 10),
    ("story2", 3, 7),
    ("story3", 8, 15),
    ("story4", 2, 5),
    ("story5", 6, 12),
    ("story6", 1, 3)
]
team_capacity_with_dependencies = 12
dependencies = {
    "story3": "story2", # story3 depends on story2
    "story5": "story1"  # story5 depends on story1
}


selected, total_value = solve_capacity_with_dependency(stories_data_with_dependencies, team_capacity_with_dependencies, dependencies)

if selected:
    print("Selected stories (with dependencies):", selected)
    print("Total value (with dependencies):", total_value)
else:
    print("No solution found.")
```

Here, I’ve added a `solve_capacity_with_dependency` function which accepts a dictionary `dependencies` with the form `{"story_dependant": "story_depended"}`. I’ve then added a constraint stating that if a dependent story is selected, then its depended story must also be selected. This ensures that inter-task dependencies are respected.

Another common requirement is to deal with varying task values and priorities. In this scenario, you can use a weighted objective function to reflect the prioritization of each story.

```python
def solve_capacity_with_weighted_priority(stories, team_capacity, priorities):
    """Solves the story selection problem with weighted priorities using CPLEX.

    Args:
      stories: A list of tuples, where each tuple is (story_id, effort).
      team_capacity: The total capacity of the team.
      priorities: A dictionary where key is story id and value is its priority (weight).
    Returns:
    A tuple containing:
      - A list of story ids included in the optimal solution, or None if no solution
        was found.
      - The maximum weighted priority of the selected stories.
    """
    mdl = Model(name="story_selection_with_priorities")

    x = mdl.binary_var_list(len(stories), name="story_selected")
    # Convert story list to dictionary for easy priority retrieval.
    story_dict = {story[0]: story for story in stories}

    # Use story_dict to compute objective value based on stored priority values in dictionary
    mdl.maximize(mdl.sum(priorities[story[0]] * x[i] for i, story in enumerate(stories) if story[0] in priorities))

    mdl.add_constraint(mdl.sum(story_dict[story[0]][1] * x[i] for i, story in enumerate(stories)) <= team_capacity)


    solution = mdl.solve()

    if solution:
        selected_stories = [stories[i][0] for i in range(len(stories)) if solution.get_value(x[i]) > 0.5]
        total_priority = solution.get_objective_value()
        return selected_stories, total_priority
    else:
        return None, 0


# Example usage with priorities:
stories_data_with_priorities = [
    ("story1", 5),
    ("story2", 3),
    ("story3", 8),
    ("story4", 2),
    ("story5", 6),
    ("story6", 1)
]
team_capacity_with_priorities = 12
priorities = {
    "story1": 5,
    "story2": 8,
    "story3": 2,
    "story4": 10,
    "story5": 6,
    "story6": 1
}

selected, total_priority = solve_capacity_with_weighted_priority(stories_data_with_priorities, team_capacity_with_priorities, priorities)

if selected:
    print("Selected stories (with priorities):", selected)
    print("Total weighted priority:", total_priority)
else:
    print("No solution found.")
```

In this final example, I’ve removed the value of the story directly from the `stories_data` and introduced a separate `priorities` dictionary to demonstrate handling story priorities. We now maximize based on the weighted priority assigned to each story.

For deeper dives into optimization techniques and the theory behind this, I strongly recommend looking into the book "Integer Programming" by Wolsey. Additionally, "Linear Programming: Foundations and Extensions" by Vanderbei is a helpful resource for the mathematical underpinnings of what CPLEX is doing. And, of course, the official CPLEX documentation itself is an essential resource for specific API details.

These three examples cover basic capacity constraints, dependencies, and prioritization; however, these are building blocks that you can combine and extend to meet the unique challenges of your situation. The flexibility of CPLEX allows you to build even more intricate and complex problem formulations. Remember, careful mathematical modeling and thoughtful application of the CPLEX API are crucial to getting optimal results. Good luck, and let me know if you hit any specific roadblocks.
