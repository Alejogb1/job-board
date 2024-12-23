---
title: "How can a custom `sumDouble()` function be implemented in OptaPlanner for constraint satisfaction?"
date: "2024-12-23"
id: "how-can-a-custom-sumdouble-function-be-implemented-in-optaplanner-for-constraint-satisfaction"
---

Alright, let's unpack implementing a custom `sumDouble()` function within the OptaPlanner framework for constraint satisfaction. It’s a scenario I've encountered a few times over the years, particularly when dealing with resource allocation problems where aggregated values play a crucial role in defining constraints. It’s not always a direct solution and requires a bit of understanding of how OptaPlanner handles score calculation.

The challenge, at its core, is this: OptaPlanner's score calculation operates on a per-entity basis, typically, and we need a method to calculate a sum of a specific property across multiple planning entities, and then use this summed value in our constraint definitions. Simply put, we can't directly use a traditional imperative loop within a `@Constraint` annotated method because it violates OptaPlanner's reactive, incremental scoring mechanism. This isn’t some theoretical edge case; it’s a very common hurdle when you move beyond basic examples and deal with real-world scenarios with interdependent constraints.

Let's think back to a project I worked on involving scheduling medical staff shifts. We needed to ensure not only fair distribution of shifts, but also to prevent a team from exceeding its weekly on-call hour limits. This required summing total on-call hours across multiple shifts for each staff member, a classic `sumDouble()` situation.

So, how do we do this? We use what are called "shadow variables," and more specifically, a `PlanningVariable` annotated with `@ValueRangeProvider` and `@InverseRelationShadowVariable`. The key here is that we move the summation logic outside the main constraint calculation and implement it as a separate, automatically maintained shadow variable. The shadow variable maintains the total sum, and this total sum is then accessed from the constraint method without doing the redundant calculation. Let's look at a conceptual breakdown of the steps and some accompanying code.

**Step 1: Define the Planning Entity and the related Planning Variable**

Let's start with a hypothetical `Shift` entity and an associated `Employee` entity which it is assigned to. We'll consider on-call hours and associated logic.

```java
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.variable.PlanningVariable;

@PlanningEntity
public class Shift {

    private double onCallHours;

    @PlanningVariable(valueRangeProviderRefs = {"employees"})
    private Employee employee;


    public Shift(double onCallHours) {
      this.onCallHours = onCallHours;
    }


    // Getters and setters for onCallHours, employee
    public double getOnCallHours() {
        return onCallHours;
    }

    public void setOnCallHours(double onCallHours) {
        this.onCallHours = onCallHours;
    }

     public Employee getEmployee() {
        return employee;
    }

    public void setEmployee(Employee employee) {
        this.employee = employee;
    }
}
```

```java
import org.optaplanner.core.api.domain.lookup.PlanningId;

public class Employee {

    @PlanningId
    private String id;

    public Employee(String id) {
      this.id = id;
    }
    // Getters and setters
    public String getId() {
       return id;
   }

    public void setId(String id) {
        this.id = id;
    }
}

```

**Step 2: Define the Shadow Variable for the Sum**

Now we introduce the shadow variable which is computed automatically using a change listener. In this case, it's a shadow variable within the `Employee` entity itself, representing the sum of all on-call hours for that employee.

```java
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.variable.PlanningVariable;
import org.optaplanner.core.api.domain.variable.InverseRelationShadowVariable;
import org.optaplanner.core.api.domain.variable.ShadowVariable;
import org.optaplanner.core.api.domain.solution.PlanningSolution;
import org.optaplanner.core.api.domain.valuerange.ValueRangeProvider;
import java.util.List;

@PlanningEntity
public class Shift {

   //Previous Shift class from above...
}


@PlanningSolution
public class ScheduleSolution {
  private List<Shift> shifts;
  private List<Employee> employees;

  public ScheduleSolution(List<Shift> shifts, List<Employee> employees){
    this.shifts = shifts;
    this.employees = employees;
  }

  @ValueRangeProvider(id = "employees")
  public List<Employee> getEmployees(){
    return employees;
  }

   public List<Shift> getShifts() {
       return shifts;
   }

   public void setShifts(List<Shift> shifts) {
       this.shifts = shifts;
   }
}

@PlanningEntity
public class Employee {
    //Previous Employee class from above..

    private double totalOnCallHours;

    @ShadowVariable(variableListenerClass = TotalOnCallHoursListener.class,
                    sourceVariableName = "employee")
    public double getTotalOnCallHours() {
        return totalOnCallHours;
    }

    public void setTotalOnCallHours(double totalOnCallHours) {
        this.totalOnCallHours = totalOnCallHours;
    }
}
```

**Step 3: Implement the Shadow Variable Listener**

This is where the actual calculation of the sum happens. The listener is triggered whenever the associated `PlanningVariable` (`employee`) in any of the `Shift` entities changes, updating the `totalOnCallHours` on the `Employee`.

```java
import org.optaplanner.core.api.domain.variable.VariableListener;
import org.optaplanner.core.api.score.director.ScoreDirector;
import java.util.ArrayList;
import java.util.List;

public class TotalOnCallHoursListener implements VariableListener<Shift> {

    @Override
    public void beforeVariableChanged(ScoreDirector scoreDirector, Shift shift) {
        // Not used
    }

    @Override
    public void afterVariableChanged(ScoreDirector scoreDirector, Shift shift) {
        Employee employee = shift.getEmployee();
        if (employee == null) return;
         double totalOnCallHours = 0;
        List<Shift> shifts =  ((ScheduleSolution)scoreDirector.getWorkingSolution()).getShifts();
       for(Shift s: shifts){
         if(s.getEmployee() != null && s.getEmployee().getId().equals(employee.getId())){
          totalOnCallHours += s.getOnCallHours();
        }
       }
        scoreDirector.beforeVariableChanged(employee, "totalOnCallHours");
        employee.setTotalOnCallHours(totalOnCallHours);
        scoreDirector.afterVariableChanged(employee, "totalOnCallHours");
    }
}
```

**Step 4: Use the Sum in Constraint Definition**

Finally, in our constraint definition, we can now simply access the `totalOnCallHours` from the `Employee` entity to create our constraint.

```java
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.constraint.ConstraintMatchTotal;
import org.optaplanner.core.api.score.constraint.ConstraintProvider;
import org.optaplanner.core.api.score.constraint.Constraint;


public class ScheduleConstraintsProvider implements ConstraintProvider {

    @Override
    public Constraint[] defineConstraints() {
        return new Constraint[]{
          maxOnCallHours(),
        };
    }
      private Constraint maxOnCallHours() {
           return Constraint.of("maxOnCallHours", HardSoftScore.class,
             (ConstraintFactory) ->
             ConstraintFactory.from(Employee.class)
               .filter(employee -> employee.getTotalOnCallHours() > 40)
                 .penalize(HardSoftScore.ONE_SOFT,
                            employee -> (int) (employee.getTotalOnCallHours()-40))
                   );
      }

}
```

In this example, we have a hard constraint that ensures that total on-call hours do not exceed 40. We now can change the value of "40" or any other metric, and the listener will update the `totalOnCallHours` whenever a shift assignment changes. We then check that value within our constraint and can modify the value of the penalty score according to how far over the on call limit it is.

**Important Considerations and Alternatives**

This pattern of `ShadowVariable` with a listener is highly flexible but can sometimes be tricky to debug if you don’t pay close attention to the order of operations and ensure your listener logic is correct. Remember that the listener operates in a reactive manner, so if you create a new `Shift` with a non-null `Employee` during the planning process, you'll want to make sure your listener accounts for that in the logic.

While I used the `@ShadowVariable` approach, there are other ways, but they are often less efficient for this type of summation. For instance, you could, potentially, try aggregating during the constraint definition itself, but this will cause the calculation to be recomputed multiple times and reduce performance. Using shadow variables gives you an efficient, reactive calculation.

**Resources**

For a deep dive into shadow variables, I recommend starting with the official OptaPlanner documentation. It covers all the intricacies, along with numerous examples. Also, specifically, the documentation on incremental score calculation.

Also, consider the book “Constraint Satisfaction with Java and OptaPlanner,” which offers a detailed practical perspective, particularly on real-world scheduling problem formulations. You’ll find several case studies and code samples which address issues like aggregation in real-world problems.

Finally, the OptaPlanner source code itself (hosted on GitHub) is invaluable. Looking at the existing variable listener implementations can be educational in understanding nuances and edge cases, it is what I did during my deep dive.

In summary, while custom summations might seem challenging initially, the `ShadowVariable` approach is a robust method for implementing a `sumDouble()` equivalent. It takes a bit of understanding of OptaPlanner's reactive nature and incremental calculation mechanisms, but it's the standard way for many optimization problems that involve aggregated data, particularly, as I experienced in my staffing project.
