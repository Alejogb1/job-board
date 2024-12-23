---
title: "How to pick stories summing to a team's capacity using Cplex?"
date: "2024-12-23"
id: "how-to-pick-stories-summing-to-a-teams-capacity-using-cplex"
---

Alright,  The task of selecting a subset of user stories that fit within a team's capacity, particularly when that capacity is measured in story points or some comparable metric, is a common challenge. I’ve personally encountered this exact scenario multiple times across different project lifecycles. The core problem, when framed mathematically, actually boils down to a variation of the classic knapsack problem, a well-studied area in operations research. Now, while various heuristic approaches might give you acceptable results, using a proper optimization engine like cplex (or similar) provides you with guarantees of optimality, assuming you've properly modelled the constraints. I remember one particular project where we had a wildly fluctuating team velocity. Trying to plan manually was just becoming unsustainable. That’s when I really started leaning heavily into constraint programming solutions, and cplex was a real workhorse.

The initial step is to formulate the problem appropriately. We're not just throwing items into a bag, we're selecting stories based on their estimated points while adhering to a total team capacity. The model should aim to *maximize* some value, and in our case, I usually optimize for a combination of factors: the value assigned to the story by product owners, or a priority score. For simplicity, let's focus on optimizing total value. This helps ensures we are selecting the most impactful stories within the available capacity.

The core idea behind this using cplex is to build a mixed-integer programming (mip) model. Essentially, we're using integer variables to indicate if a story is *in* or *out* (1 or 0), and cplex then figures out the optimal combination that maximizes the defined objective function, while respecting capacity constraint.

Here's a python snippet using `docplex`, the python api for cplex, demonstrating how this is modeled:

```python
from docplex.mp.model import Model

def solve_story_selection(stories, capacity):
    """
    Solves the story selection problem using CPLEX.

    Args:
        stories (list of tuples): A list where each tuple is (story_id, points, value).
        capacity (int): Total capacity of the team.

    Returns:
        tuple: A tuple containing the list of selected story IDs and the total selected value, or None if infeasible.
    """

    model = Model("story_selection")

    # Decision variables (0 or 1): does a story get selected or not?
    x = model.binary_var_list(keys=[s[0] for s in stories], name="select_story")


    # Objective: maximize sum of the value of selected stories
    model.maximize(model.sum(x[s[0]] * s[2] for s in stories))

    # Constraint: total points used should be <= capacity
    model.add_constraint(model.sum(x[s[0]] * s[1] for s in stories) <= capacity)

    solution = model.solve()

    if solution is not None:
        selected_stories = [s[0] for s in stories if solution.get_value(x[s[0]]) > 0.5]
        total_value = solution.get_objective_value()
        return selected_stories, total_value
    else:
        return None  # No solution was found (infeasible problem).

# Example usage:
stories = [
    ("story_a", 5, 20),
    ("story_b", 3, 15),
    ("story_c", 7, 30),
    ("story_d", 2, 10),
    ("story_e", 4, 25),
]

team_capacity = 10

selected, total_value = solve_story_selection(stories, team_capacity)

if selected:
    print(f"selected stories: {selected}, total value: {total_value}")
else:
   print("No solution found (infeasible problem)")
```

In this snippet, I first define the model and the decision variables. Then the objective function (maximize total value) is established, followed by a single constraint—the capacity limit. The method `model.solve()` instructs cplex to determine the optimal combination. The result can be used to extract the solution, specifically the selected stories and their total value.

Sometimes, constraints other than capacity are necessary. Let's suppose we have dependencies between stories - say that story 'b' can only be considered if story 'a' is also selected. Here is a modified example:

```python
from docplex.mp.model import Model

def solve_story_selection_with_dependencies(stories, capacity, dependencies):
    """
    Solves the story selection problem with dependencies using CPLEX.

    Args:
        stories (list of tuples): A list where each tuple is (story_id, points, value).
        capacity (int): Total capacity of the team.
        dependencies (dict): A dictionary where keys are dependent stories and values are their pre-requisite stories.

    Returns:
        tuple: A tuple containing the list of selected story IDs and the total selected value, or None if infeasible.
    """

    model = Model("story_selection_with_dependencies")

     # Decision variables (0 or 1): does a story get selected or not?
    x = model.binary_var_list(keys=[s[0] for s in stories], name="select_story")

    # Objective: maximize sum of the value of selected stories
    model.maximize(model.sum(x[s[0]] * s[2] for s in stories))

    # Constraint: total points used should be <= capacity
    model.add_constraint(model.sum(x[s[0]] * s[1] for s in stories) <= capacity)


    # add constraint for dependencies.
    for dependent_story, prereq_story in dependencies.items():
        model.add_constraint(x[dependent_story] <= x[prereq_story])

    solution = model.solve()

    if solution is not None:
        selected_stories = [s[0] for s in stories if solution.get_value(x[s[0]]) > 0.5]
        total_value = solution.get_objective_value()
        return selected_stories, total_value
    else:
        return None  # No solution was found (infeasible problem).


# Example usage:
stories = [
    ("story_a", 5, 20),
    ("story_b", 3, 15),
    ("story_c", 7, 30),
    ("story_d", 2, 10),
    ("story_e", 4, 25),
]

team_capacity = 10

dependencies = { "story_b" : "story_a" }


selected, total_value = solve_story_selection_with_dependencies(stories, team_capacity, dependencies)

if selected:
    print(f"selected stories: {selected}, total value: {total_value}")
else:
   print("No solution found (infeasible problem)")
```

Here, a `dependencies` dictionary is introduced, and added as constraints in the model that ensures any dependent story `b` is only included if prerequisite story `a` is also selected. This kind of dependency handling can be crucial in a realistic sprint planning process.

Finally, sometimes you might not be working solely with points and value. You could also have constraints around the *type* of story (front-end, back-end, etc), or specific skills needed. Here is an example demonstrating this:

```python
from docplex.mp.model import Model

def solve_story_selection_with_skills(stories, capacity, skill_limits):
    """
    Solves the story selection problem with skill requirements using CPLEX.

    Args:
        stories (list of tuples): A list where each tuple is (story_id, points, value, skill type).
        capacity (int): Total capacity of the team.
        skill_limits (dict): A dictionary that contains the maximum number of points a team can spend on a given skill.

    Returns:
       tuple: A tuple containing the list of selected story IDs and the total selected value, or None if infeasible.
    """

    model = Model("story_selection_with_skills")

    # Decision variables (0 or 1): does a story get selected or not?
    x = model.binary_var_list(keys=[s[0] for s in stories], name="select_story")

    # Objective: maximize sum of the value of selected stories
    model.maximize(model.sum(x[s[0]] * s[2] for s in stories))

    # Constraint: total points used should be <= capacity
    model.add_constraint(model.sum(x[s[0]] * s[1] for s in stories) <= capacity)

    # Constraint on skills
    for skill, limit in skill_limits.items():
        model.add_constraint(model.sum(x[s[0]] * s[1] for s in stories if s[3] == skill) <= limit)


    solution = model.solve()

    if solution is not None:
        selected_stories = [s[0] for s in stories if solution.get_value(x[s[0]]) > 0.5]
        total_value = solution.get_objective_value()
        return selected_stories, total_value
    else:
        return None  # No solution was found (infeasible problem).

# Example Usage:
stories = [
    ("story_a", 5, 20, "frontend"),
    ("story_b", 3, 15, "backend"),
    ("story_c", 7, 30, "frontend"),
    ("story_d", 2, 10, "backend"),
    ("story_e", 4, 25, "qa"),
]

team_capacity = 12

skill_limits = {
    "frontend": 10,
    "backend": 8,
    "qa": 4
}

selected, total_value = solve_story_selection_with_skills(stories, team_capacity, skill_limits)

if selected:
   print(f"selected stories: {selected}, total value: {total_value}")
else:
    print("No solution found (infeasible problem)")
```
Here we’ve added a "skill" column to the story definition. The `skill_limits` dictionary enforces constraints on how many story points of each skill type can be chosen.

When building out your models, I'd recommend starting simple and then adding complexity only as necessary.  For a deeper dive, consider exploring "Integer Programming" by Wolsey for a comprehensive understanding of mixed-integer programming. "Optimization in Operations Research" by Ronald Rardin provides an excellent mathematical framework. I’ve used these resources often. The key is understanding that cplex isn't a magical box; it's a powerful tool, but it requires that we understand how to formulate the problem mathematically and encode that into constraints and objective functions. Doing so gives you the confidence of knowing you're making mathematically sound choices that will lead to more predictable sprint execution, which, at the end of the day, is what all of this is about.
