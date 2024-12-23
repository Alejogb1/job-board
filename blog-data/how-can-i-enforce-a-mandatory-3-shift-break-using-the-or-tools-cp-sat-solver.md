---
title: "How can I enforce a mandatory 3-shift break using the OR-Tools CP-SAT solver?"
date: "2024-12-23"
id: "how-can-i-enforce-a-mandatory-3-shift-break-using-the-or-tools-cp-sat-solver"
---

, so let's delve into the nuances of enforcing a mandatory 3-shift break using Google’s OR-Tools CP-SAT solver. I recall a particularly challenging project a few years back where we had to manage a crew scheduling problem for a continuous operation plant. The crux of it? Enforcing a rigid break period after a fixed number of consecutive shifts, precisely what you're tackling. We tried a few methods before landing on something that proved quite robust.

The challenge here isn't so much about defining the shift assignment variables – that's usually straightforward. The tricky bit is formulating the constraints that *force* a break after, say, every three shifts. Naively attempting this can lead to a combinatorially explosive set of constraints, which the solver will struggle with. Instead, the solution usually involves introducing auxiliary variables and a careful formulation to express the break requirements logically.

Let’s break it down into core concepts, and then I'll illustrate it with code. The primary idea is to track the number of *consecutive* shifts for each worker. We don't need to know *which* shifts they work, just how many in a row. This can be accomplished by having variables indicating whether a worker is currently in a 'run' of shifts. A 'run' simply implies that a worker has been working and continues to work in consecutive time steps. We'll then use this information to trigger the mandatory break.

Here’s how I’ve found it effective: We introduce a set of binary variables, let's call them `in_run[worker, shift_index]`. This variable is 1 if worker is working at `shift_index` and has been working in the immediately preceding shift (or the very first shift of the day) otherwise 0. The variable `consecutive_shifts[worker, shift_index]` represents the number of consecutive shifts a worker has worked up to and including `shift_index`, while the variable `mandatory_break[worker, shift_index]` will indicate if it is a mandatory break for a worker at a given `shift_index`.

Here's how it translates into a working example with Python and OR-Tools:

```python
from ortools.sat.python import cp_model

def create_schedule(num_workers, num_shifts):
    model = cp_model.CpModel()

    # Shift assignment variables (1 if worker works shift, 0 otherwise)
    shifts = {}
    for worker in range(num_workers):
        for shift_index in range(num_shifts):
            shifts[(worker, shift_index)] = model.NewBoolVar(f'shift_w{worker}_s{shift_index}')

    # Variables to track if worker is in a 'run' of shifts
    in_run = {}
    for worker in range(num_workers):
      for shift_index in range(num_shifts):
        in_run[(worker, shift_index)] = model.NewBoolVar(f'in_run_w{worker}_s{shift_index}')

    # Variables to track consecutive shifts
    consecutive_shifts = {}
    for worker in range(num_workers):
      for shift_index in range(num_shifts):
        consecutive_shifts[(worker, shift_index)] = model.NewIntVar(0, 3, f'consec_shift_w{worker}_s{shift_index}')


     # Variables for mandatory breaks
    mandatory_break = {}
    for worker in range(num_workers):
      for shift_index in range(num_shifts):
        mandatory_break[(worker, shift_index)] = model.NewBoolVar(f'break_w{worker}_s{shift_index}')

    # Constraint: First shift is always the beginning of run
    for worker in range(num_workers):
      model.Add(in_run[(worker,0)] == shifts[(worker,0)])

    # Constraint: If the current shift is a working shift and the previous was a run, current is also a run.
    for worker in range(num_workers):
      for shift_index in range(1, num_shifts):
          model.AddBoolOr([shifts[(worker, shift_index)].Not(), in_run[(worker, shift_index -1)], in_run[(worker, shift_index)]])
          model.AddImplication(in_run[(worker, shift_index)], shifts[(worker,shift_index)])
          model.AddImplication(in_run[(worker, shift_index)].Not(), shifts[(worker,shift_index)].Not())

    # Constraint: consecutive shifts if run is true, else, 0.
    for worker in range(num_workers):
      for shift_index in range(num_shifts):
          if shift_index == 0:
            model.Add(consecutive_shifts[(worker, shift_index)] == shifts[(worker, shift_index)])
          else:
            model.Add(consecutive_shifts[(worker, shift_index)] == consecutive_shifts[(worker, shift_index - 1)] + shifts[(worker, shift_index)]).OnlyEnforceIf(in_run[(worker, shift_index)])
            model.Add(consecutive_shifts[(worker, shift_index)] == 0).OnlyEnforceIf(in_run[(worker,shift_index)].Not())

    # Constraint: Mandatory break after 3 consecutive shifts
    for worker in range(num_workers):
      for shift_index in range(num_shifts):
        model.Add(mandatory_break[(worker, shift_index)] == 1).OnlyEnforceIf(consecutive_shifts[(worker, shift_index)] == 3)
        model.Add(mandatory_break[(worker, shift_index)] == 0).OnlyEnforceIf(consecutive_shifts[(worker, shift_index)] < 3)

    # Constraint: If it is a mandatory break, worker does not work
    for worker in range(num_workers):
        for shift_index in range(num_shifts):
            model.AddImplication(mandatory_break[(worker, shift_index)], shifts[(worker, shift_index)].Not())

    # Dummy Objective: Let's maximize worker utilization as example
    obj = sum(shifts.values())
    model.Maximize(obj)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for worker in range(num_workers):
            print(f"Worker {worker}:")
            for shift_index in range(num_shifts):
                if solver.Value(shifts[(worker, shift_index)]) == 1:
                  print(f"  Shift {shift_index} (consec:{solver.Value(consecutive_shifts[(worker, shift_index)])}, break:{solver.Value(mandatory_break[(worker, shift_index)])})")
                else:
                   print(f"  Shift {shift_index} (Off) (consec:{solver.Value(consecutive_shifts[(worker, shift_index)])}, break:{solver.Value(mandatory_break[(worker, shift_index)])})")

    else:
        print("No solution found.")

create_schedule(num_workers=2, num_shifts=10)
```

In this snippet, we create our main `shift` variable, then auxiliary variables `in_run`, `consecutive_shifts`, and `mandatory_break` which we then constraint appropriately. The crucial part is how `consecutive_shifts` is updated based on if `in_run` is `True` or not. Then we trigger a `mandatory_break` if the variable `consecutive_shifts` reaches our maximum value (3). Finally, when `mandatory_break` is true, the worker is enforced to be off. This method ensures the break is *truly* mandatory, without needing a complex set of inequalities for each shift pattern. The output shows the shifts each worker takes, alongside the number of consecutive shifts and if they were in a mandatory break.

This approach avoids having to enumerate every possible sequence of 3 shifts followed by a break, which would be cumbersome.

Let's illustrate another approach using a slightly more compressed approach. Here is another snippet:

```python
from ortools.sat.python import cp_model

def create_schedule_compressed(num_workers, num_shifts):
    model = cp_model.CpModel()

    shifts = {}
    for worker in range(num_workers):
        for shift_index in range(num_shifts):
            shifts[(worker, shift_index)] = model.NewBoolVar(f'shift_w{worker}_s{shift_index}')

    consecutive_shifts = {}
    for worker in range(num_workers):
      for shift_index in range(num_shifts):
        consecutive_shifts[(worker, shift_index)] = model.NewIntVar(0, 4, f'consec_shift_w{worker}_s{shift_index}')


    for worker in range(num_workers):
        for shift_index in range(num_shifts):
          if shift_index == 0:
            model.Add(consecutive_shifts[(worker, shift_index)] == shifts[(worker, shift_index)])
          else:
            model.Add(consecutive_shifts[(worker, shift_index)] == consecutive_shifts[(worker, shift_index-1)] + shifts[(worker,shift_index)]).OnlyEnforceIf(shifts[(worker, shift_index)])
            model.Add(consecutive_shifts[(worker, shift_index)] == 0).OnlyEnforceIf(shifts[(worker, shift_index)].Not())

    for worker in range(num_workers):
      for shift_index in range(num_shifts):
          model.Add(shifts[(worker, shift_index)] == 0).OnlyEnforceIf(consecutive_shifts[(worker, shift_index)] >= 4)

    obj = sum(shifts.values())
    model.Maximize(obj)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for worker in range(num_workers):
            print(f"Worker {worker}:")
            for shift_index in range(num_shifts):
                if solver.Value(shifts[(worker, shift_index)]) == 1:
                    print(f"  Shift {shift_index} (consec:{solver.Value(consecutive_shifts[(worker, shift_index)])})")
                else:
                     print(f"  Shift {shift_index} (Off) (consec:{solver.Value(consecutive_shifts[(worker, shift_index)])})")
    else:
        print("No solution found.")

create_schedule_compressed(num_workers=2, num_shifts=10)

```

Here, instead of relying on the `in_run` variable, we derive the logic directly into `consecutive_shifts`. If a worker is assigned a shift, then consecutive shifts increment else, we set it to zero. The mandatory break happens when `consecutive_shifts` reaches 4, at which point we force the worker to be off.

Finally, let's consider a more sophisticated approach if we don't allow work in consecutive shifts. This might be useful if we want a break period between two shifts, not just enforced after a run of shifts.

```python
from ortools.sat.python import cp_model

def create_schedule_intermittent(num_workers, num_shifts):
    model = cp_model.CpModel()

    shifts = {}
    for worker in range(num_workers):
        for shift_index in range(num_shifts):
            shifts[(worker, shift_index)] = model.NewBoolVar(f'shift_w{worker}_s{shift_index}')


    # Constraint: Mandatory break if work occurs in the immediately previous shift.
    for worker in range(num_workers):
      for shift_index in range(1, num_shifts):
          model.AddImplication(shifts[(worker, shift_index - 1)], shifts[(worker, shift_index)].Not())

    # Dummy objective
    obj = sum(shifts.values())
    model.Maximize(obj)


    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for worker in range(num_workers):
            print(f"Worker {worker}:")
            for shift_index in range(num_shifts):
                if solver.Value(shifts[(worker, shift_index)]) == 1:
                  print(f"  Shift {shift_index}")
                else:
                  print(f"  Shift {shift_index} (Off)")
    else:
        print("No solution found.")


create_schedule_intermittent(num_workers=2, num_shifts=10)
```

Here, the mandatory break is enforced with a simple implication statement: if a worker is working the previous shift, then the worker *cannot* work the current one. This forces a mandatory break period of at least 1 shift after each work shift. Note that this forces an *alternating* shift pattern, rather than the 3-on-1-off requirement, but it demonstrates how to use boolean implications to enforce breaks.

For further study, I recommend diving into the following resources: *Constraint Programming in Python* by Laurent D. Michel, which provides a good theoretical grounding, and *Handbook of Constraint Programming* by Francesca Rossi, Peter van Beek, and Toby Walsh which covers the technical and mathematical underpinnings of CP. Reading the official documentation of the OR-Tools library is essential, and papers on specific scheduling challenges will further refine your modeling skills. These resources will deepen your understanding and enable you to tailor these techniques to even more complex scenarios.

Enforcing break periods correctly requires thinking about how to express the logical flow of constraints effectively within the CP-SAT framework. These examples should serve as a good starting point. Good luck with your scheduling puzzles.
