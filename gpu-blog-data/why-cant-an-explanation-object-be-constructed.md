---
title: "Why can't an Explanation object be constructed?"
date: "2025-01-30"
id: "why-cant-an-explanation-object-be-constructed"
---
The inability to directly construct an `Explanation` object typically stems from its design as an abstract class or interface, requiring concrete implementations for instantiation. This design pattern promotes extensibility and polymorphism, allowing diverse explanation strategies to be treated uniformly while preventing the creation of bare, undefined explanations. I've encountered this pattern frequently in systems needing nuanced and context-aware debugging output.

An `Explanation` object, within a well-architected system, usually represents a structured collection of details intended to clarify the behavior or outcome of a specific process. It might include error messages, intermediate values, or traces of execution, forming a cohesive rationale. Directly creating an `Explanation` without specifying the type of information it should encapsulate defeats this purpose, hence the intentional lack of a public constructor. The framework expects subclasses to define the particulars of the information captured.

To illustrate, consider a scenario involving a hypothetical machine learning pipeline. In this context, an `Explanation` object might describe the reasons behind a specific prediction, a common requirement for model interpretability. A base `Explanation` class could declare methods for retrieving the underlying reasoning but not provide any concrete implementation. We then create subclasses, each implementing specific explanation generation strategies.

Here's how such an abstract class might appear in Java:

```java
public abstract class Explanation {

    public abstract String getSummary();
    public abstract String getDetails();

    // The lack of a public constructor is deliberate
    // Concrete subclasses will handle instantiation
}

```
This basic class defines the general contract for any `Explanation` object. Note the absence of a `public` constructor, enforcing the need to derive concrete explanation types. It mandates subclasses to implement `getSummary` and `getDetails`, ensuring that all explanation objects can provide a basic level of information. Attempting to instantiate `new Explanation()` will result in a compile error.

Now, let's examine a specific concrete implementation for feature importance, which I've frequently utilized when inspecting the output of tree-based models.
```java
import java.util.Map;
import java.util.TreeMap;

public class FeatureImportanceExplanation extends Explanation {

    private final TreeMap<String, Double> featureWeights;
    private final String decision;

    public FeatureImportanceExplanation(String decision, TreeMap<String, Double> featureWeights) {
        this.decision = decision;
        this.featureWeights = featureWeights;
    }


    @Override
    public String getSummary() {
        return "Decision: " + decision + ". Feature importance provided below.";
    }

    @Override
    public String getDetails() {
        StringBuilder details = new StringBuilder();
        details.append("Feature Importances:\n");
        for (Map.Entry<String, Double> entry : featureWeights.entrySet()) {
            details.append(String.format("%-20s : %.4f\n", entry.getKey(), entry.getValue()));
        }
        return details.toString();
    }
}
```

This `FeatureImportanceExplanation` class extends the abstract `Explanation` class and provides concrete implementations for `getSummary` and `getDetails`.  The constructor is `public`, allowing objects of this type to be created, passing in the feature weights as a `TreeMap` along with a decision string. This demonstrates that the base `Explanation` class doesn't dictate how to create the information, just that the subclasses provide an implementation that can respond to certain methods for retrieving that information. Crucially, there are no empty constructors or ways to instantiate it directly.

Consider another scenario within the system: A rule-based decision-making component. Let’s say that the component identifies which rules were triggered and which rule led to the eventual outcome. This rule-based explanation is another subclass with its own set of fields.

```java
import java.util.List;

public class RuleBasedExplanation extends Explanation {

    private final String triggeredRule;
    private final List<String> allTriggeredRules;

    public RuleBasedExplanation(String triggeredRule, List<String> allTriggeredRules) {
        this.triggeredRule = triggeredRule;
        this.allTriggeredRules = allTriggeredRules;
    }


    @Override
    public String getSummary() {
        return "Decision was driven by rule: " + triggeredRule + ".";
    }

    @Override
    public String getDetails() {
        StringBuilder details = new StringBuilder();
        details.append("List of triggered rules:\n");
        for (String rule : allTriggeredRules) {
          details.append(rule).append("\n");
        }
        return details.toString();
    }

}
```
Here, `RuleBasedExplanation` focuses on capturing the reasoning behind decisions made by a rule engine. The constructor requires a single key rule, and all rules that were engaged. As before the constructor is `public`, and there are no means of directly instantiating the abstract class.  Again, this class implements `getSummary` and `getDetails` as required by the parent. The variation in the field types and structure clearly showcases the value of the abstract class.

These examples highlight the purpose of an abstract class like `Explanation`. It doesn't describe how an explanation is created or implemented but provides the skeleton for concrete types. It's a contract enforced by the language that ensures type safety and guarantees that all derived explanation classes have methods to access summaries and details. It also helps to ensure that when dealing with an explanation, you can rely on the contract without worrying about the underlying implementation.

The inability to construct a base `Explanation` object directly is not a limitation, but rather a core design element, facilitating the creation of a well-structured and extensible system for capturing and managing diverse explanation types. It's an application of a very common and useful design pattern that prevents misuse of an otherwise ambiguous interface.

For a deeper understanding, I recommend exploring resources discussing abstract classes, interfaces, and the concept of polymorphism in object-oriented programming. Further investigation into the design patterns surrounding the Template method and the Strategy pattern might provide valuable insights into similar approaches to solving related problems. Look for discussions on how these patterns enhance code maintainability and flexibility. It’s also useful to learn about the SOLID principles of software design. These principles, especially the Liskov Substitution Principle, will help frame the context and benefits of designing such structures within an application or framework. Reading documentation related to the Strategy pattern would be beneficial since these explanation objects could easily be seen as an implementation of such a pattern.
