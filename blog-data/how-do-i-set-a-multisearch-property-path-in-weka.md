---
title: "How do I set a MultiSearch property path in WEKA?"
date: "2024-12-23"
id: "how-do-i-set-a-multisearch-property-path-in-weka"
---

Alright, let's dive into the intricacies of configuring property paths within WEKA's `MultiSearch` component. This is something I've spent quite a bit of time on, particularly back when I was optimizing ensemble methods for a rather complex time-series forecasting project. The challenge usually isn't the *idea* of a multi-search, but nailing down the precise syntax for those nested property paths. It can certainly feel a bit cryptic at first.

The crucial thing to understand is that `MultiSearch`, as its name suggests, isn’t searching *data*; instead, it’s searching through the parameter space of other WEKA objects – typically classifiers or filters. These objects can have their own internal properties, and navigating through those nested structures requires a very specific path syntax. Think of it like traversing a file system where each component is a folder and each property is a file within those folders. You need the full path to the target property, not just the file name.

Essentially, the path structure looks like this: `[object identifier].[property name]` if the target property belongs to the object being directly explored by the `MultiSearch`. However, if that object, in turn, has its own internal objects with their own properties, we nest the identifiers, which might lead to something like `[object identifier].[nested object identifier].[property name]`. It's a bit like a chain of references, and each step needs to be precise.

Let me clarify this with some working code snippets. We'll tackle a few common scenarios I've encountered.

**Example 1: Tuning a simple classifier’s parameters**

Let's start with a basic example, adjusting parameters of a `weka.classifiers.trees.J48` classifier using a `MultiSearch` object. In this scenario, the J48 classifier is directly passed to MultiSearch, so the paths are straightforward. I’ll show you how to configure that:

```java
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.meta.MultiSearch;

public class MultiSearchExample1 {

  public static void main(String[] args) throws Exception {
    // Load sample dataset (replace with your actual dataset)
    DataSource source = new DataSource("path/to/your/dataset.arff"); // Replace with path to your data
    Instances data = source.getDataSet();
    if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);

    // Convert numeric class to nominal (if required by classifier)
    NumericToNominal nominalConverter = new NumericToNominal();
    nominalConverter.setInputFormat(data);
    nominalConverter.setAttributeIndices("" + (data.classIndex() + 1));
    data = Filter.useFilter(data, nominalConverter);


    // Initialize J48 classifier
    J48 j48 = new J48();

    // Initialize MultiSearch object
    MultiSearch multiSearch = new MultiSearch();
    multiSearch.setClassifier(j48); // Set the classifier to be used with MultiSearch
    multiSearch.setSearchParameters(" -E 10 -sampleSize 100"); //  Set parameters for multi search
    multiSearch.addPropertyPath(" -C ", "C", "-M ", "M" ); // configure the property paths we are searching over.
    multiSearch.setVerbose(true);

    // Perform the search
     multiSearch.buildClassifier(data);

    // Evaluate the best found classifier
     Evaluation eval = new Evaluation(data);
     eval.evaluateModel(multiSearch,data);
     System.out.println(eval.toSummaryString("\nResults\n======\n", true));
     System.out.println("Best classifier found:"+multiSearch.getBestClassifier());

  }
}
```

In this example, we are directly setting the properties of the `J48` classifier, which we have set as the target for the `MultiSearch`. As you can see, I've specified paths such as "`-C`", and "`M`". This translates to the command-line arguments that the J48 uses, and then also maps it to the Java object's method call.

**Example 2: Targeting nested filters**

Now, let’s consider a more complex example: tuning properties of a filter that is used *inside* a classifier. Say, we are using a `weka.classifiers.meta.FilteredClassifier`, which first applies a filter before classifying. We want to tune the parameter of that embedded filter. This is where the nested paths come into play.

```java
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Normalize;
import weka.meta.MultiSearch;

public class MultiSearchExample2 {

  public static void main(String[] args) throws Exception {
     // Load sample dataset (replace with your actual dataset)
      DataSource source = new DataSource("path/to/your/dataset.arff"); // Replace with path to your data
      Instances data = source.getDataSet();
      if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);

     // Convert numeric class to nominal (if required by classifier)
     NumericToNominal nominalConverter = new NumericToNominal();
     nominalConverter.setInputFormat(data);
     nominalConverter.setAttributeIndices("" + (data.classIndex() + 1));
     data = Filter.useFilter(data, nominalConverter);


    // Initialize a FilteredClassifier
    FilteredClassifier filteredClassifier = new FilteredClassifier();

    // Initialize the Normalize filter
    Normalize normalize = new Normalize();
    filteredClassifier.setFilter(normalize);


    // Initialize the J48 classifier
    J48 j48 = new J48();
    filteredClassifier.setClassifier(j48);

    // Initialize MultiSearch object
    MultiSearch multiSearch = new MultiSearch();
    multiSearch.setClassifier(filteredClassifier); // Set the filtered classifier
    multiSearch.setSearchParameters(" -E 10 -sampleSize 100");
    multiSearch.addPropertyPath("filter.scale", "scale"); // configure the property path we are searching over
    multiSearch.setVerbose(true);

    // Perform the search
     multiSearch.buildClassifier(data);

    // Evaluate the best found classifier
     Evaluation eval = new Evaluation(data);
     eval.evaluateModel(multiSearch,data);
     System.out.println(eval.toSummaryString("\nResults\n======\n", true));
     System.out.println("Best classifier found:"+multiSearch.getBestClassifier());


  }
}
```

Here, the property path is `filter.scale`. The "filter" is not a direct property of MultiSearch. Instead, it corresponds to the identifier of the filter within the FilteredClassifier (the getter method of that property). So, `filter.scale` means "go to the filter object within the `FilteredClassifier` and adjust its `scale` property. Again, these are based on the command-line arguments that the underlying filter uses and maps to the Java methods associated with it.

**Example 3: Complex nested path with multiple internal objects**

Let's look at something truly complex - imagine a scenario where your classifier *itself* uses internal meta-learners, that have yet another layer of objects within them. In this conceptual example, the exact components may not exist in the standard WEKA, but the principle remains the same:

```java
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.MultiScheme;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Normalize;
import weka.meta.MultiSearch;

public class MultiSearchExample3 {
    public static void main(String[] args) throws Exception {
      // Load sample dataset (replace with your actual dataset)
        DataSource source = new DataSource("path/to/your/dataset.arff"); // Replace with path to your data
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);

        // Convert numeric class to nominal (if required by classifier)
        NumericToNominal nominalConverter = new NumericToNominal();
        nominalConverter.setInputFormat(data);
        nominalConverter.setAttributeIndices("" + (data.classIndex() + 1));
        data = Filter.useFilter(data, nominalConverter);


        // Initialize MultiScheme (Imagine it has an internal J48 under the name "baseLearner")
        MultiScheme multiScheme = new MultiScheme();
        J48 baseLearner = new J48();
        multiScheme.setClassifier(baseLearner);


        // Initialize FilteredClassifier
        FilteredClassifier filteredClassifier = new FilteredClassifier();
        Normalize normalize = new Normalize();
        filteredClassifier.setFilter(normalize);
        filteredClassifier.setClassifier(multiScheme);



        // Initialize MultiSearch
        MultiSearch multiSearch = new MultiSearch();
        multiSearch.setClassifier(filteredClassifier);
        multiSearch.setSearchParameters(" -E 10 -sampleSize 100");
        multiSearch.addPropertyPath("classifier.baseLearner.C", "C","classifier.baseLearner.M", "M", "filter.scale", "scale" );
        multiSearch.setVerbose(true);


        // Perform the search
         multiSearch.buildClassifier(data);

        // Evaluate the best found classifier
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(multiSearch,data);
        System.out.println(eval.toSummaryString("\nResults\n======\n", true));
        System.out.println("Best classifier found:"+multiSearch.getBestClassifier());

    }
}

```
In this case, we have a FilteredClassifier containing a MultiScheme classifier, which further internally contains J48.  So, if you want to tune the parameters of the internal J48, the correct property path is going to be `classifier.baseLearner.C` and `classifier.baseLearner.M` - reflecting the nesting. Also, we have kept the `filter.scale` parameter too. This illustrates that you can tune all the parameters nested arbitrarily deeply within.

To truly grasp the inner workings, consulting the source code of the specific WEKA classifiers and filters involved is essential. Resources like the WEKA documentation and associated academic papers, particularly those focusing on WEKA internals, are invaluable. I also recommend, specifically, *Data Mining: Practical Machine Learning Tools and Techniques* by Ian H. Witten, Eibe Frank, Mark A. Hall, and Christopher J. Pal, as a good grounding in the underlying concepts.

This might seem a bit abstract at first, but with practice, and careful examination of the relevant object hierarchies, those cryptic property paths become significantly less daunting. Remember, the keys here are the object identifiers and the precise property names. Trace the object structure, and you'll find your way.
