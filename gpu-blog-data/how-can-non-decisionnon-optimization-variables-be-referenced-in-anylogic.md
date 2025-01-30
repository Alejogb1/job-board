---
title: "How can non-decision/non-optimization variables be referenced in AnyLogic multi-objective optimization?"
date: "2025-01-30"
id: "how-can-non-decisionnon-optimization-variables-be-referenced-in-anylogic"
---
Non-decision variables, crucial for model fidelity yet excluded from the optimization process, require careful handling in AnyLogic's multi-objective optimization.  My experience optimizing complex supply chain models highlighted a critical oversight: neglecting their influence on objective function evaluations.  Ignoring these variables leads to unrealistic or incomplete solutions, compromising the validity of the optimization results.  The key lies in understanding that while not directly optimized, their values, calculated within the model's logic, fundamentally impact the objective functions and therefore need proper referencing.

The core challenge stems from AnyLogic's optimization framework expecting variables explicitly defined within the optimization experiment setup. Non-decision variables, by definition, fall outside this scope.  However, their values are intrinsically linked to the decision variables, and their impact on the objectives needs accurate representation. This is achieved through careful model structuring and leveraging AnyLogic's data handling capabilities.  Specifically, we must ensure the non-decision variables are computed *before* the objective functions are evaluated within each simulation run initiated by the optimizer.


**1. Clear Explanation**

The methodology involves a two-step process:

a) **Model Structure:**  Organize the model's elements to ensure the non-decision variables are calculated based on the current values of the decision variables *before* the objective functions are accessed.  This often necessitates a clear separation of model components. Create a dedicated agent or block to handle computations related to non-decision variables, ensuring its execution precedes the objective function calculations.  The sequence of events is paramount.  If the objectives are calculated first, the non-decision variables won't reflect the impact of the current decision variables, leading to incorrect optimization.

b) **Data Access:**  Use AnyLogic's built-in mechanisms (e.g., agent variables, experiment parameters, data collection blocks) to appropriately store and access the calculated values of the non-decision variables.  These values are then directly used within the objective function definitions.  Directly referencing the non-decision variable within the objective function expression ensures the optimizer considers their implicit influence on the solution.

**2. Code Examples with Commentary**


**Example 1: Simple Inventory Model**

This example demonstrates a simplified inventory model where production quantity (`productionQuantity`, a decision variable) influences inventory levels (`inventoryLevel`, a non-decision variable) which in turn impacts costs.


```java
//Decision variable defined in the AnyLogic optimization experiment
double productionQuantity;

//Non-decision variable calculated within the model
double inventoryLevel = initialInventory; //Initial inventory

//Model logic
void onTick(){
  inventoryLevel += productionQuantity - demand; //demand is predefined
  if(inventoryLevel < 0) inventoryLevel = 0; //Cannot have negative inventory
}

//Objective function (to be minimized)
double totalCost(){
  return productionCost * productionQuantity + holdingCost * inventoryLevel; //Cost functions are predefined
}
```

Here, `inventoryLevel` is not a decision variable, but its value directly affects the `totalCost`, which is the objective function minimized by the optimizer. The optimizer manipulates `productionQuantity`, consequently impacting `inventoryLevel` and the objective function.


**Example 2:  Agent-Based Model with Network Effects**

In this more complex scenario, we have agents with different characteristics. A decision variable is the resource allocation, which affects an agent's production and thereby its influence on a network-wide metric (a non-decision variable).


```java
//Decision variable (resource allocation to agent i)
double resourceAllocation[i]; //Array of decision variables

//Agent class definition
class Agent{
  double production;
  double influence; //Non-decision variable (network influence)

  void calculateInfluence(){
    influence = production * networkConnectivity; //networkConnectivity is a predefined network metric
  }
}

//Objective Function (to be maximized)
double overallNetworkInfluence(){
  double totalInfluence = 0;
  for(Agent a: agents){
    a.calculateInfluence();
    totalInfluence += a.influence;
  }
  return totalInfluence;
}

//Main model logic
void onTick(){
    for(Agent a : agents){
        a.production = resourceAllocation[a.id] * agentProductivity; //agentProductivity is predefined
    }
}
```

`influence` is a non-decision variable calculated based on `production` which in turn depends on the optimizer-controlled `resourceAllocation`.  The `overallNetworkInfluence` objective function uses the computed `influence` values, ensuring the network effects are considered during optimization.


**Example 3:  Queueing System with Dynamic Server Capacity**

In this example, server capacity is a non-decision variable dependent on the decision variable representing maintenance schedules.


```java
//Decision variable (maintenance schedule)
int maintenanceSchedule;

//Non-decision variable (server capacity)
int serverCapacity = initialCapacity;


//Model logic that updates server capacity
void onMaintenance(){
    serverCapacity = initialCapacity - maintenanceReduction[maintenanceSchedule]; //maintenanceReduction is predefined
}


//Objective function (to be minimized) - average waiting time
double avgWaitingTime(){
    return totalWaitingTime / totalCustomers;
}
```

Here, the `maintenanceSchedule` directly influences `serverCapacity` which is then implicitly factored into the calculation of `avgWaitingTime`, thereby ensuring the optimizer accounts for the impact of maintenance on queuing performance.


**3. Resource Recommendations**

I strongly recommend consulting the AnyLogic documentation on multi-objective optimization and model development.  Pay close attention to sections detailing the creation and configuration of experiments and the proper use of variables within the model's code.  A deep understanding of Java programming principles within AnyLogic's context is also invaluable for intricate model designs.  Exploring example models provided within AnyLogic's help system can provide further insights into model structuring and best practices.  Finally, a firm grasp of optimization theory will enhance your understanding of how to formulate your objective functions and interpret the results effectively.  Remember to systematically validate your model and optimize it iteratively to refine its accuracy and the efficacy of the optimization process.
