---
title: "How can a custom collection improve encapsulation?"
date: "2024-12-23"
id: "how-can-a-custom-collection-improve-encapsulation"
---

Okay, let's tackle this. Encapsulation, as we all appreciate, is pivotal in maintaining robust and maintainable code. It's about bundling data and the methods that operate on that data, and restricting direct access to some of the object's components. While readily available collections like `List`, `Set`, and `Map` often serve us well, sometimes they fall short, especially when fine-grained control over how data is accessed and modified is necessary. That's where custom collections shine—they offer a powerful mechanism for enhancing encapsulation.

My experience has shown me that using custom collections properly isn't just about adding complexity; it's about strategically shaping the interface to your data structure. I remember, a while back, during a large-scale sensor network project, we were tracking readings from various nodes. Initially, we dumped all the sensor data into a simple list of data points. It worked… initially. But soon, we had a massive problem with data corruption. Different parts of the application were inadvertently modifying parts of the data they shouldn't have, directly accessing list elements and making assumptions about its internal structure. This led to all sorts of frustrating bugs that were difficult to track down. It was clear we needed a more controlled way to handle this data.

This specific scenario highlighted that even with the best intentions, default collections can sometimes allow too much exposure. The core issue was that the standard `List` exposed its internals. Any code holding a reference to the list had full access to insert, remove, or change any element, regardless of whether it logically had that permission. We needed something that would enforce data access constraints.

Creating a custom collection became our next step, a move that significantly improved our encapsulation. The essence of enhancing encapsulation with a custom collection lies in defining a very clear interface and then carefully controlling which parts of that internal data structure are exposed. Instead of giving direct access to underlying data, methods in a custom collection offer specific behaviors. Think of it as handing out tools instead of keys to your house.

For example, let's consider a very basic custom collection designed to store unique IDs. In many applications, ensuring the uniqueness of identifiers is crucial. Simply using a `List` would necessitate adding checks throughout the application. Instead, we can bake that uniqueness into the collection itself:

```java
import java.util.HashSet;
import java.util.Set;

public class UniqueIdCollection {
    private final Set<String> ids = new HashSet<>();

    public boolean addId(String id) {
        if (id == null || id.trim().isEmpty()) {
            throw new IllegalArgumentException("ID cannot be null or empty.");
        }
        return this.ids.add(id);
    }

    public boolean containsId(String id) {
        return this.ids.contains(id);
    }


    public int size() {
      return this.ids.size();
    }


    // Prevent accidental modifications by not exposing the internal set directly.
    //No way to get the underlying data structure.
    // public Set<String> getIds() {
      //  return ids; // Wrong - breaks encapsulation
    //}

}

```

In this `UniqueIdCollection`, the underlying storage is a `HashSet`, which inherently enforces uniqueness. However, notice that we don't expose that `HashSet` directly. We provide well-defined methods (`addId`, `containsId`, `size`) that operate on the data in a controlled way. The user of this class cannot modify the underlying set structure directly. They can only add, check, or get the number of Ids based on the provided interface, enforcing the "add only unique id" constraint and keeping internal structure hidden. This approach hides the underlying data structure and the enforcement of uniqueness behind an interface. This example, although seemingly simple, does more than using `Set<String>` directly in other classes because of the encapsulation provided.

Let's take another example from a scenario where I had to manage a backlog of tasks in a project. While a `List` would be enough for storing the tasks, it would be very easy for some module to accidentally add duplicate tasks, or even modify the status of tasks without the task management module knowing about it. This led to some rather strange edge cases we spent hours debugging. We addressed that by developing a custom `TaskQueue` collection:

```java
import java.util.ArrayList;
import java.util.List;

enum TaskStatus {
  PENDING,
  PROCESSING,
  COMPLETED,
  FAILED
}

class Task {
  private String id;
  private String description;
  private TaskStatus status;

  public Task(String id, String description){
    this.id = id;
    this.description = description;
    this.status = TaskStatus.PENDING;
  }

    public String getId() {
        return id;
    }

    public String getDescription() {
        return description;
    }

    public TaskStatus getStatus() {
      return status;
    }
    public void setStatus(TaskStatus status){
      this.status = status;
    }
}


public class TaskQueue {
    private List<Task> tasks = new ArrayList<>();

    public void addTask(Task task) {
       if(task == null){
           throw new IllegalArgumentException("Task cannot be null.");
       }
        this.tasks.add(task);
    }

    public Task getNextTask() {
        if(this.tasks.isEmpty()){
           return null;
        }
        return this.tasks.remove(0);
    }
    public void completeTask(String taskId){
        for (Task task : tasks) {
          if(task.getId().equals(taskId)){
            task.setStatus(TaskStatus.COMPLETED);
             return;
          }
        }

    }

    public List<Task> getPendingTasks(){
     List<Task> pendingTasks = new ArrayList<>();
     for(Task task : tasks){
        if(task.getStatus() == TaskStatus.PENDING){
          pendingTasks.add(task);
        }
      }
     return pendingTasks;
    }
    public int size(){
      return this.tasks.size();
    }
}

```

Here, the `TaskQueue` manages `Task` objects in the order they were added. We've encapsulated not just the data storage but the logic of task handling as well. Instead of directly manipulating a `List`, external modules interact with the `TaskQueue` via its methods, preventing modules from changing the status of a task directly, and making sure all status changes happen via the `completeTask` method. This greatly reduces the scope of accidental modifications and makes the task management more robust.

One final example, perhaps more applicable to complex scenarios such as data validation, is where custom collections can shine. In a past project dealing with financial data, maintaining data integrity was of utmost importance. We couldn't rely on a standard collection because of the number of specific constraints imposed on the dataset. This inspired us to create a `ValidatedDataSet`:

```java

import java.util.ArrayList;
import java.util.List;

class DataPoint{
    private double value;

    public DataPoint(double value) {
        if(value < 0){
            throw new IllegalArgumentException("Data value must be non-negative.");
        }
        this.value = value;
    }
    public double getValue(){
        return this.value;
    }
}

public class ValidatedDataSet {
    private final List<DataPoint> dataPoints = new ArrayList<>();
    private final double maxValue;

    public ValidatedDataSet(double maxValue) {
        if(maxValue <= 0){
            throw new IllegalArgumentException("Max value must be positive.");
        }
        this.maxValue = maxValue;
    }

    public void addDataPoint(double value) {
        if(value > this.maxValue){
            throw new IllegalArgumentException("Data value exceeds max allowed value.");
        }
        this.dataPoints.add(new DataPoint(value));
    }
    public List<Double> getValues(){
      List<Double> values = new ArrayList<>();
      for(DataPoint dp : dataPoints){
        values.add(dp.getValue());
      }
      return values;
    }

     public int size(){
      return this.dataPoints.size();
    }

    //Not exposing the internals
     //public List<DataPoint> getRawDataPoints(){
      //  return this.dataPoints
     //}
}


```

The `ValidatedDataSet` collection ensures that all the data points are within a defined range. It does not just store data points; it validates and protects the state of the data at insert time, preventing any invalid entry into the dataset. It also hides the internal details about how data is stored. This custom collection allows other parts of the application to interact with the dataset with confidence that all the data within is consistent with the set validation constraints, ensuring data integrity.

In each of these examples, the crux of the matter is that we are *controlling* access to the underlying data, which is the essence of good encapsulation. Instead of direct access to a raw list or set, we are offering a very explicit, controlled interface. This approach not only hides the internal implementation details, but it prevents data integrity issues by limiting access.

For further reading, I suggest diving into "Effective Java" by Joshua Bloch, particularly the sections on encapsulation and object-oriented design. Another valuable resource is "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans, which discusses how to model business logic effectively through well-encapsulated objects and entities. These resources should provide further, in-depth understanding and help you develop your approach to crafting custom collections. In summary, custom collections are not just about re-inventing the wheel, they’re about crafting highly specialized tools that enhance the clarity, robustness, and maintainability of your code through improved encapsulation.
