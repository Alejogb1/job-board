---
title: "How can Minizinc handle enumerated option sets based on array index, rather than a single set for all indices?"
date: "2025-01-30"
id: "how-can-minizinc-handle-enumerated-option-sets-based"
---
In MiniZinc, scenarios often arise where decision variables need to select from different option sets depending on their position within an array. Simply declaring a single enumerated set for all elements of the array becomes insufficient. I encountered this directly when modeling a complex scheduling problem involving diverse resource types, each with a unique availability profile. My solution required each task, represented as an array element, to choose a resource from a specific subset, rather than from a universal pool. This necessitates a strategy to define sets of options varying by index.

The core challenge is that MiniZinc does not directly support index-dependent enumeration sets within the core language syntax. The standard approach of declaring a single `set of int` or `enum` type for an array forces each array element to choose from the same set. To bypass this limitation, I utilize a combination of array indexing into a set of sets and boolean constraint programming. Specifically, I construct an array where each element holds the enumerated set pertinent to that index. Then I use boolean variables coupled with `forall` constraints to ensure selection of elements only from their respective indexed sets. This methodology enables a flexible representation of dynamically changing option pools.

Here’s how it breaks down: I first create an array of `set of int` where each position represents the enumerated set applicable to the corresponding index in my target decision variable array.  Let's say I have a 5-element array, `tasks`, each requiring an option from different option sets, named `optSets`. These option sets could represent resources, colors, or any other set of choices defined by integers for simplicity. I initialize `optSets` such that each element contains the specific integers.  Next, the decision variable array, `tasks`, will be defined using the standard `var` keyword, but without a static constraint. Finally, I employ a `forall` loop coupled with boolean implication constraints to ensure that each `tasks[i]` value belongs to `optSets[i]`.

```minizinc
int: nTasks = 5;

% Define the option sets (represented as integer sets)
array[1..nTasks] of set of int: optSets = [
  {1, 2, 3},  % Set for tasks[1]
  {4, 5},   % Set for tasks[2]
  {2, 6},     % Set for tasks[3]
  {7, 8, 9},  % Set for tasks[4]
  {1, 9}      % Set for tasks[5]
];

% Define decision variables without initial domain restriction
array[1..nTasks] of var int: tasks;

% Apply the constraints.  For each task 'i', make sure
% it takes a value from the set defined by optSets[i]
constraint forall(i in 1..nTasks) (
  exists (option in optSets[i]) (tasks[i] == option)
);

solve satisfy;

output [show(tasks)];
```

In this example, `optSets` is explicitly defined with integer sets for clarity. `tasks` is declared as a `var int` array, meaning its elements can initially take any integer value. The core logic lies within the `forall` constraint. For each task index `i`, `exists (option in optSets[i]) (tasks[i] == option)` ensures that the chosen integer for `tasks[i]` is an element within `optSets[i]`. The 'exists' construct is needed because minizinc doesn't allow direct membership check, only comparison. This code results in `tasks` being filled with values where each `tasks[i]` value is present within the corresponding `optSets[i]`. In essence, I'm enforcing the membership of each decision variable through a set of logical expressions.

Let’s consider a slightly more complex scenario, where my option sets are not defined by integer literals but calculated based on some input parameters. Let's assume that each `tasks` element needs to choose a machine number, where the valid machines depend on the task type, which I will assume is represented by an integer `taskType`. I will use this `taskType` to index another array of sets called `allowedMachines`.

```minizinc
int: nTasks = 5;

% Represent task types, for example
array[1..nTasks] of int: taskType = [1, 2, 1, 3, 2];

% Allowed machines for each task type.
array[1..3] of set of int: allowedMachines = [
  {1, 2, 3}, % Machines for task type 1
  {4, 5, 6},  % Machines for task type 2
  {7, 8}    % Machines for task type 3
];

% Precalculate the sets each task can choose from, based on their type.
array[1..nTasks] of set of int: optSets = [
  allowedMachines[taskType[i]] | i in 1..nTasks
];


array[1..nTasks] of var int: tasks;

% Apply the constraints.
constraint forall(i in 1..nTasks) (
    exists(option in optSets[i]) (tasks[i] == option)
);

solve satisfy;

output [show(tasks)];
```

Here, I've introduced `taskType` which dictates, through `allowedMachines`, the possible option sets for each task using list comprehension. `optSets` is built dynamically using list comprehension based on the `taskType` of each index which is then used in the same manner to constrain each `tasks` element.  This demonstrates how option sets can depend on complex logic involving other parameters without altering the general constraint programming pattern.

Finally, let's say instead of a static set of options for each index, the valid options dynamically depend on an external data source, such as a matrix read from a text file. In this example, the presence of '1' at position [i,k] means that option 'k' is available for index i. This provides a very flexible way of defining complex dependencies. The size of the option set can vary greatly for each index.

```minizinc
int: nTasks = 5;
int: nOptions = 10;

% Dynamically generate the option sets from an external data source
% Represented here as a hard coded 2D array
array[1..nTasks, 1..nOptions] of int: availOptions =
  array2d(1..nTasks,1..nOptions,
     [ 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 1, 0]);

% Create the sets dynamically based on available options
array[1..nTasks] of set of int: optSets =
  [ { k | k in 1..nOptions where availOptions[i,k] == 1 } | i in 1..nTasks ];

array[1..nTasks] of var int: tasks;

constraint forall(i in 1..nTasks) (
    exists(option in optSets[i]) (tasks[i] == option)
);

solve satisfy;

output [show(tasks)];
```

In this version, `availOptions` defines the availability of options. The option sets are calculated through list comprehension using a `where` clause filtering through the `availOptions` array, enabling dynamic generation. Again, the core constraint logic involving the `forall` loop remains unchanged. The key idea is that the logic used to generate the `optSets` can be arbitrary.

In summary, implementing enumerated option sets that vary by array index in MiniZinc requires a workaround using array of sets and boolean `exists` constraints within a `forall` loop. It does not require any special compiler options, but it is critical to understand that it works as set membership test rather than a direct constraint based upon a static domain declaration. It provides a flexible and robust solution for a wide range of modeling scenarios where static enumeration sets are not sufficient. This method can be generalized to far more complex cases where the option sets change dynamically based on many other modeling parameters. To further develop your understanding, I suggest exploring the MiniZinc Handbook, specifically the sections on set types, and constraints. Practice creating diverse model examples with varied set constructions. Furthermore, reviewing case studies focusing on complex scheduling or resource allocation can be beneficial. These resources should provide a solid theoretical understanding as well as examples of different modeling patterns which can be adapted to different problem domains.
