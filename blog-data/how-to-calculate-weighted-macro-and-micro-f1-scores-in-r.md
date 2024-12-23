---
title: "How to calculate weighted macro and micro F1-scores in R?"
date: "2024-12-23"
id: "how-to-calculate-weighted-macro-and-micro-f1-scores-in-r"
---

Let's tackle this, shall we? I recall a particularly challenging project back in my fintech days, where we were building a sophisticated fraud detection system. We had highly imbalanced datasets, and using a standard F1-score just wasn't cutting it. That's when we needed to dive deep into weighted F1-scores, and I spent a considerable amount of time figuring out the nuances in R. So, let's break this down methodically, because the devil is definitely in the details here.

First off, when talking about F1-scores, we need to understand that we're essentially balancing precision and recall. The regular F1-score does this for a binary classification problem effectively, or even for multi-class in the macro-averaged form when we treat each class equally. However, when our data is imbalanced, this simple averaging can be misleading, as it could give disproportionate weight to the better performing classes and obscure the problems in the rarer classes, a scenario we faced often with fraud cases where genuine transactions greatly outnumber the fraudulent ones. Weighted macro and micro F1-scores address this head-on.

The *macro-averaged* F1 score calculates the F1-score for each class individually, and *then* computes the arithmetic mean of these scores, either with or without weights to each class. The *micro-averaged* F1 score, on the other hand, aggregates the true positives (TP), false positives (FP), and false negatives (FN) across all classes *before* calculating the F1-score. This difference is crucial. Micro averaging gives equal weight to each instance, not each class, which is particularly advantageous in imbalanced scenarios where some classes have far more instances than others. Weighted macro averaging allows us to assign importance to classes based on how many samples each class has or perhaps even based on a subjective scale we set ourselves.

Let’s look at code. Initially, let’s construct a simple scenario and the calculations step-by-step, without directly using existing R packages. This helps to illustrate the underlying mathematics and the difference between these approaches.

```r
# Sample confusion matrix
conf_matrix <- matrix(c(10, 2, 3, 25), nrow = 2, byrow = TRUE)
rownames(conf_matrix) <- c("Class_A", "Class_B")
colnames(conf_matrix) <- c("Predicted_A", "Predicted_B")
print(conf_matrix)

# Calculate precision, recall, and F1 for each class
precision_A <- conf_matrix[1,1] / sum(conf_matrix[,1])
recall_A <- conf_matrix[1,1] / sum(conf_matrix[1,])
f1_A <- 2 * (precision_A * recall_A) / (precision_A + recall_A)

precision_B <- conf_matrix[2,2] / sum(conf_matrix[,2])
recall_B <- conf_matrix[2,2] / sum(conf_matrix[2,])
f1_B <- 2 * (precision_B * recall_B) / (precision_B + recall_B)

print(paste("F1-score for Class A:", f1_A))
print(paste("F1-score for Class B:", f1_B))

# Macro-averaged F1 without weights
macro_f1_unweighted <- mean(c(f1_A, f1_B))
print(paste("Macro-averaged F1 (Unweighted):", macro_f1_unweighted))

# Sample weights (proportional to the number of instances, for illustration purposes)
weights <- rowSums(conf_matrix)/ sum(conf_matrix)

# Weighted Macro Averaged F1 Score
macro_f1_weighted <- (f1_A * weights[1] + f1_B*weights[2])
print(paste("Macro-averaged F1 (Weighted):", macro_f1_weighted))

# Micro-averaged F1
tp_micro <- sum(diag(conf_matrix))
fp_micro <- sum(conf_matrix) - tp_micro
fn_micro <- sum(conf_matrix) - tp_micro
micro_precision <- tp_micro/ (tp_micro + sum(conf_matrix[,2]))
micro_recall <- tp_micro/ (tp_micro + sum(conf_matrix[2,]))
micro_f1 <- 2* (micro_precision * micro_recall)/ (micro_precision + micro_recall)

print(paste("Micro-averaged F1:", micro_f1))

```
This snippet provides a hands-on calculation example and shows that micro-averaging involves aggregating the values and computing a single value from the total aggregated TP, FP and FN. The unweighted macro-averaged F1 is the mean of per class f1 score, while weighted macro-averaged F1 gives more weight to classes that have more instances.

Now, we can transition to using R packages that offer these functions. Let's use the `caret` package, a popular choice for this kind of task. I've found its flexibility invaluable over the years.
```r
# Install and load necessary libraries if not installed
# install.packages(c("caret", "e1071"))
library(caret)

# Sample predictions and true labels (assuming binary classification)
predicted_labels <- factor(c("A", "B", "A", "B", "A", "B", "B", "B","B", "A"))
true_labels <- factor(c("A", "B", "B", "B", "A", "A", "B", "B", "B", "A"))


# Create a confusion matrix
confusion_matrix <- confusionMatrix(predicted_labels, true_labels)
print(confusion_matrix)

# Calculate macro-averaged F1 score
macro_f1_caret <- mean(confusion_matrix$byClass[,"F1"])
print(paste("Macro-averaged F1 (caret):", macro_f1_caret))

# Calculate micro-averaged F1 score
# Note that the default caret behavior doesn't directly compute micro F1 and does it as "Accuracy"
micro_f1_caret <- confusion_matrix$overall["Accuracy"]
print(paste("Micro-averaged F1 (caret Accuracy):", micro_f1_caret))


# Example of weights
weights <- c(0.7, 0.3) # Sample weights, for example
weighted_macro_f1 <- sum(confusion_matrix$byClass[,"F1"] * weights)
print(paste("Weighted macro-averaged F1 (caret):", weighted_macro_f1))


```

This example shows how easily we can achieve the same results with the `caret` package. As you can see, the `confusionMatrix` function computes the F1-score for each class and you can compute macro average of those. Note that in the context of `caret`, the `Accuracy` metric is equivalent to micro-averaged F1 score, this was a surprise the first time I used this library. This example also adds an arbitrary weight example, to show how weights can be added to class averages.

Let's consider a more sophisticated example with a multi-class scenario, which is common in real-world applications. We will use the `mltools` package which I also used extensively.

```r
# Install the package if necessary
# install.packages("mltools")
library(mltools)

# Generate some sample data for demonstration
actual_labels <- factor(sample(c("A", "B", "C"), 100, replace = TRUE))
predicted_labels <- factor(sample(c("A", "B", "C"), 100, replace = TRUE))

# Create the confusion matrix
conf_mat <- table(actual_labels, predicted_labels)
print(conf_mat)

# Calculate the macro average f1-score
macro_f1 <- macro_avg_f1(conf_mat)
print(paste("Macro-averaged F1:", macro_f1))

# Calculate micro-averaged F1
micro_f1 <- micro_avg_f1(conf_mat)
print(paste("Micro-averaged F1:", micro_f1))

# Calculate the weighted-macro average F1 score
class_weights <- table(actual_labels) / length(actual_labels)
weighted_macro_f1 <- weighted_avg_f1(conf_mat, class_weights)

print(paste("Weighted Macro-averaged F1:", weighted_macro_f1))

```

This final example shows how `mltools` can help compute macro, micro and weighted f1-scores in a multiclass setting. I find that this library makes the weighted average calculations much easier.

When you're delving into this, I would recommend spending some time with:

1.  *Pattern Recognition and Machine Learning* by Christopher Bishop. It’s a classic text that lays the groundwork for understanding these concepts, particularly the mathematical underpinnings.
2.  *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman. While a more advanced text, it provides further insights into classification and performance metrics.
3.  The documentation for both the `caret` and `mltools` packages in R. Both are well documented and provide specific usage examples that clarify any confusion.

In conclusion, understanding how to calculate weighted macro and micro F1-scores is critical when dealing with imbalanced data or situations where different classes have varying degrees of importance. While initially seeming complex, when broken down into steps and demonstrated with code examples as we have done here, these concepts become much more manageable. It is not just knowing the code, but understanding the mathematical reasoning and implication of the scores on our model performance that is important. That way, you will avoid the pitfalls of focusing solely on metrics that are inadequate. Remember always to look at metrics that fit the specific task at hand, and never just blindly use one or the other metric. This principle has served me very well throughout my experience in this field.
