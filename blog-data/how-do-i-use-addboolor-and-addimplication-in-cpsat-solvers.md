---
title: "How do I use AddBoolOr and AddImplication in cp_sat solvers?"
date: "2024-12-16"
id: "how-do-i-use-addboolor-and-addimplication-in-cpsat-solvers"
---

Alright, let’s tackle this. It's been a while since I’ve directly implemented constraint programming at this level, but the intricacies of `AddBoolOr` and `AddImplication` in cp-sat solvers are definitely something I’ve spent considerable time with in past projects—specifically, a scheduling system for a high-throughput manufacturing line where optimizing for both resource allocation and task dependencies was crucial. It wasn't pretty at times, and I certainly learned a few things the hard way, so hopefully this helps.

Essentially, these two methods are your bread and butter when expressing logical relationships between boolean variables within the constraint programming domain, allowing you to construct fairly complex constraints. Let's break them down individually before looking at code.

`AddBoolOr` constructs a constraint that ensures at least one boolean variable within a given list is true. Think of it as a logical "or" operation, extended to handle multiple variables. If you have a set of conditions, any of which can trigger an event, you'd use `AddBoolOr` to model this relationship. For example, if either machine *a*, machine *b*, or machine *c* being available implies that production can proceed, this would be a prime candidate for a `AddBoolOr` constraint.

Now, `AddImplication` is slightly different. It models a conditional statement: "if *variable x* is true, then *variable y* must be true." This is the classic logical implication. Crucially, the solver interprets this relationship correctly, including what happens when the *if* condition (variable *x*) is false; in that case, the *then* condition (variable *y*) is free to be either true or false. It’s important to remember it’s not the same as a bi-implication (an if and only if relationship). In practice, I often used `AddImplication` to model dependencies between activities, such as "if task A has started, then task B can start," or as part of more complex conditional constraint constructions where certain constraints were only active if a certain condition held. This often showed up in resource allocation logic; for example, "if the maintenance window is scheduled, then the specific resource used must be unavailable".

The real power of these constructs comes from combining them within a larger model. It allows you to express fairly intricate logical dependencies that go far beyond basic linear constraints, something that the manufacturing scheduling project relied on heavily.

, let's dive into some code examples, each using a different context to illustrate a practical use case. These examples assume you're working with a constraint solver library like Google OR-Tools, which is pretty common, but the principles will apply across most similar libraries.

**Example 1: Machine Selection**

Consider a scenario where you have three machines (A, B, and C) that can perform a job. We need at least one of these machines to be active (i.e., assigned to the job). If machine A is selected, an additional setup step is required.

```python
from ortools.sat.python import cp_model

class MachineSelectionModel(cp_model.CpModel):
    def __init__(self):
        super().__init__()
        self.machine_a_active = self.NewBoolVar('machine_a_active')
        self.machine_b_active = self.NewBoolVar('machine_b_active')
        self.machine_c_active = self.NewBoolVar('machine_c_active')
        self.setup_required = self.NewBoolVar('setup_required')

        # Constraint: At least one machine must be active
        self.AddBoolOr([self.machine_a_active, self.machine_b_active, self.machine_c_active])

        # Constraint: If machine A is active, then setup is required
        self.AddImplication(self.machine_a_active, self.setup_required)

if __name__ == '__main__':
    model = MachineSelectionModel()
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        print(f"Machine A Active: {solver.Value(model.machine_a_active)}")
        print(f"Machine B Active: {solver.Value(model.machine_b_active)}")
        print(f"Machine C Active: {solver.Value(model.machine_c_active)}")
        print(f"Setup Required: {solver.Value(model.setup_required)}")
    else:
        print("No solution found.")

```

This snippet models the scenario described. The `AddBoolOr` ensures that at least one of the machine booleans is true. The `AddImplication` then dictates that if machine A is selected, the `setup_required` boolean will also be true. This demonstrates a direct application of both functionalities, creating a simple, yet effective model.

**Example 2: Conditional Resource Availability**

Now, let’s move to a situation with some conditional resource availability. Suppose we have a resource that’s only available if a specific maintenance window is not scheduled.

```python
from ortools.sat.python import cp_model

class ResourceAvailabilityModel(cp_model.CpModel):
    def __init__(self):
        super().__init__()
        self.maintenance_scheduled = self.NewBoolVar('maintenance_scheduled')
        self.resource_available = self.NewBoolVar('resource_available')
        self.use_resource = self.NewBoolVar("use_resource")

        # Constraint: If maintenance is not scheduled, then resource is available
        self.AddImplication(self.maintenance_scheduled.Not(), self.resource_available)

        # Constraint: if we are using the resource it has to be available.
        self.AddImplication(self.use_resource, self.resource_available)


if __name__ == '__main__':
    model = ResourceAvailabilityModel()
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        print(f"Maintenance Scheduled: {solver.Value(model.maintenance_scheduled)}")
        print(f"Resource Available: {solver.Value(model.resource_available)}")
        print(f"Using Resource: {solver.Value(model.use_resource)}")
    else:
        print("No solution found.")
```
Here, the `AddImplication` is used with the negation of the `maintenance_scheduled` boolean variable, achieving a scenario where we need to assert conditional availability based on an inverted state. This is a common scenario when working with schedule exclusions. Note also how we are using another implication, this time to say if we use the resource, it needs to be available. This illustrates how these can be combined to model more nuanced conditions

**Example 3: Job Dependencies**

Let’s move to task dependencies. If job A is executed, then job B must also be executed. Crucially, if job A isn’t executed, job B can be executed or not.

```python
from ortools.sat.python import cp_model

class JobDependencyModel(cp_model.CpModel):
    def __init__(self):
        super().__init__()
        self.job_a_executed = self.NewBoolVar('job_a_executed')
        self.job_b_executed = self.NewBoolVar('job_b_executed')

        # Constraint: If job A is executed, then job B must also be executed.
        self.AddImplication(self.job_a_executed, self.job_b_executed)

if __name__ == '__main__':
    model = JobDependencyModel()
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        print(f"Job A Executed: {solver.Value(model.job_a_executed)}")
        print(f"Job B Executed: {solver.Value(model.job_b_executed)}")
    else:
        print("No solution found.")
```

This shows a straightforward use of `AddImplication` to represent dependencies between tasks, something that appears often in scheduling problems. It showcases the exact scenario for an `if a then b` condition.

Regarding further study, I highly recommend the book "Handbook of Constraint Programming" edited by Francesca Rossi, Peter van Beek, and Toby Walsh, as a good starting point for deep theory. For more practical applications using OR-Tools, the official documentation is essential. Additionally, look at specific academic papers on constraint-based scheduling and planning for more advanced uses. Finally, I've found that examining existing large-scale optimization models, often found in academic or open source projects, is a very valuable learning tool to understand best practices.

In summary, both `AddBoolOr` and `AddImplication` are foundational tools for expressing complex logical relationships in constraint programming, and they are quite versatile when combined. These snippets should give a solid start on implementing these constraints in practice. Remember, the key is to understand the underlying logical relationships you want to model, and then translate these directly to your constraint system. Let me know if you have further questions.
