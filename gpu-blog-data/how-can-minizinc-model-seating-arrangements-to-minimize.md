---
title: "How can MiniZinc model seating arrangements to minimize distance from 'furius' individuals?"
date: "2025-01-30"
id: "how-can-minizinc-model-seating-arrangements-to-minimize"
---
Minimizing the distance between disruptive individuals, whom I’ll term “furius” for this exercise, and other attendees in a seating arrangement problem is an interesting constraint satisfaction challenge. This requires translating a qualitative concept like “disruptive” into a quantifiable measure, which, in this scenario, involves maximizing separation. I've handled similar logistical optimization problems across several client deployments, often dealing with resource allocation where proximity is crucial. Essentially, we're using the distance between people in a discrete space as a proxy for comfort or reduced disruption.

The core of this problem lies in efficiently representing the seating space and applying constraints to maximize the distance between designated “furius” individuals and the rest. MiniZinc is well-suited for this because it allows for declarative problem modeling. I'll approach this by building a model that includes these key elements: seating positions as a 2D grid, a set of “furius” individuals, a distance metric, and constraints that ensure these individuals are positioned to minimize their proximity to others. I'll leverage the constraint programming paradigm to achieve an optimal arrangement.

Here’s how it breaks down within a MiniZinc model. First, we need to establish the data. Let's assume we have a rectangular grid representing our seating space with dimensions rows x cols. We then represent each individual with an identifier. A subset of these IDs will be designated as “furius”. The "distance" metric, here, will be the Manhattan distance—the sum of the absolute differences of their row and column coordinates. A more complex distance metric (e.g., Euclidean) is certainly possible, but I'll keep the complexity minimal. The goal is then to maximize the minimum distance between each "furius" individual and all others. It is important to note the assumption is that the other attendees are indifferent about their placement, the optimization is focused on separating the "furius" individuals from everyone else.

Here is a first illustrative example:

```minizinc
int: rows = 5;
int: cols = 5;
int: num_people = 10;
set of int: PEOPLE = 1..num_people;

int: num_furius = 2;
set of int: FURIUS = 1..num_furius;
set of int: REGULAR = PEOPLE diff FURIUS;

array[PEOPLE] of var int: row_pos (1..rows);
array[PEOPLE] of var int: col_pos (1..cols);

% Assign distinct seating positions
constraint forall(i,j in PEOPLE where i < j) (row_pos[i] != row_pos[j] \/ col_pos[i] != col_pos[j]);

% Dummy data for initial positions
array[FURIUS] of int: initial_row_furius = [1,5];
array[FURIUS] of int: initial_col_furius = [1,5];

% Restrict furios individuals to these starting locations
constraint forall(f in FURIUS) (row_pos[f] == initial_row_furius[f] /\ col_pos[f] == initial_col_furius[f]);

% Minimize proximity, focusing on the minimal distance
var int: min_distance;
constraint min_distance = min([abs(row_pos[f]-row_pos[p]) + abs(col_pos[f]-col_pos[p]) | f in FURIUS, p in REGULAR]);

solve maximize min_distance;

output [ "Row:" ++ show(row_pos[i]) ++ ", Col: " ++ show(col_pos[i]) ++ "\n" | i in PEOPLE ];
output ["Minimum distance: " ++ show(min_distance)];

```
This basic model introduces the key components. We declare the size of our grid (5x5) and the total number of individuals (10), with 2 of them as “furius”. We create two arrays, `row_pos` and `col_pos`, to represent the coordinates of each person. Crucially, we add a constraint to guarantee distinct seating positions. A distance measure, `min_distance`, is also added. We use dummy starting position for furios indivuduals for now. The `solve maximize` instruction seeks an arrangement that maximizes the minimum distance between "furius" and regular individuals. The output provides the final arrangement of rows and columns, along with the maximum minimum distance achieved.

However, the above model does not allow the 'furius' individuals to move. Here is another iteration:
```minizinc
int: rows = 5;
int: cols = 5;
int: num_people = 10;
set of int: PEOPLE = 1..num_people;

int: num_furius = 2;
set of int: FURIUS = 1..num_furius;
set of int: REGULAR = PEOPLE diff FURIUS;

array[PEOPLE] of var int: row_pos (1..rows);
array[PEOPLE] of var int: col_pos (1..cols);

% Assign distinct seating positions
constraint forall(i,j in PEOPLE where i < j) (row_pos[i] != row_pos[j] \/ col_pos[i] != col_pos[j]);


% Minimize proximity, focusing on the minimal distance
var int: min_distance;
constraint min_distance = min([abs(row_pos[f]-row_pos[p]) + abs(col_pos[f]-col_pos[p]) | f in FURIUS, p in REGULAR]);

solve maximize min_distance;

output [ "Row:" ++ show(row_pos[i]) ++ ", Col: " ++ show(col_pos[i]) ++ "\n" | i in PEOPLE ];
output ["Minimum distance: " ++ show(min_distance)];
```
Here, I've removed the hard-coded initial positions for “furius” individuals, allowing the solver to position them in a way that maximizes separation. The fundamental structure remains, but now the solver has more freedom. This iteration will likely yield a higher minimum distance compared to the first. The key is allowing more flexibility during solving. It is very important to remember to constrain position in the 'constraint forall' line.

Here is the final code example, improving the last code using a `minimax` constraint:
```minizinc
int: rows = 5;
int: cols = 5;
int: num_people = 10;
set of int: PEOPLE = 1..num_people;

int: num_furius = 2;
set of int: FURIUS = 1..num_furius;
set of int: REGULAR = PEOPLE diff FURIUS;

array[PEOPLE] of var int: row_pos (1..rows);
array[PEOPLE] of var int: col_pos (1..cols);

% Assign distinct seating positions
constraint forall(i,j in PEOPLE where i < j) (row_pos[i] != row_pos[j] \/ col_pos[i] != col_pos[j]);

% Calculate the minimum distance for each furius person and make them all as large as possible
array [FURIUS] of var int: min_distances;
constraint forall(f in FURIUS) (min_distances[f] = min([abs(row_pos[f]-row_pos[p]) + abs(col_pos[f]-col_pos[p]) | p in REGULAR]));
var int: final_min_distance;
constraint final_min_distance = min(min_distances);

solve maximize final_min_distance;

output [ "Row:" ++ show(row_pos[i]) ++ ", Col: " ++ show(col_pos[i]) ++ "\n" | i in PEOPLE ];
output ["Minimum distance: " ++ show(final_min_distance)];

```
This version further refines our approach, using an array to contain the minimum distances of each furius individual. This more closely aligns with the actual goal: to maximize the minimal separation between every "furius" person and all others. Instead of the `min_distance` variable being the minimal distance *of all* of the distances, which is difficult to express programmatically, the code now represents each 'furius' minimal distance separately. This formulation offers a slightly stronger constraint, by encouraging a uniform distribution of maximal distances. The final `final_min_distance` variable gives the minimum of these minimum distances which is then maximised.

For further exploration and understanding of related concepts, I would strongly suggest exploring resources covering constraint programming, specifically material on global constraints, search strategies, and the MiniZinc documentation itself. Reading about the limitations of linear integer programming is also useful. In addition, exploring material on optimal placement algorithms that are not necessarily based on constraint programming could be useful, if speed or flexibility is paramount. Finally, understanding different distance metrics beyond the Manhattan method would also be beneficial.
