---
title: "How do I use templates with plain java blocks in JAPE grammars?"
date: "2024-12-15"
id: "how-do-i-use-templates-with-plain-java-blocks-in-jape-grammars"
---

so, you're tackling the beast that is jape and wanting to mix plain java code with those sweet, sweet template rules. i've been there, my friend, and it can get hairy pretty quickly if you're not careful. it's not exactly intuitive, is it?

i remember my first real project using gate and jape; it was a mess. i was trying to build a named entity recognizer for medical texts. i had these super complex annotation types, and i needed to do some serious string manipulation that was way beyond what jape rules could handle alone. i started trying to hack together multiple rules and features, it became a deeply nested maze. i quickly understood that using java was the way to go. so, i dove into the docs (which at the time felt like they were written in ancient sanskrit), and after much trial and error, i figured it out. i'm going to lay out the basics for you, using examples that are a little less convoluted than my medical text fiasco.

the core concept is that you can embed java code within a jape rule using a `java:` block. these blocks allow you to execute arbitrary java, and more importantly, interact with annotations and features during processing. you will also use templates. templates are your way to extract information from annotations after it is modified by java code and create a new annotation. the general format is:

    rule: rule_name
    (
        ({your_annotation_type.feature1=="some_value"})
        :match
    )
    -->
    {
    java:
      // your java code block goes here
    }:output
    :output.my_new_annotation_type = {template_string};

let's break this down:

*   `rule: rule_name`: pretty self-explanatory. you give your rule a unique name.
*   `({your_annotation_type.feature1=="some_value"}) :match`: this is your pattern matching. you select annotations of a certain type with specific feature values. the `:match` is how the rule can refer to these annotations.
*   `-->`: that’s where the fun starts, it separates the pattern from what we do with it.
*   `{ java: ... }:output`: this is your java block that is executed when there is a match. the `:output` allows us to refer to the java-produced annotation.
*   `:output.my_new_annotation_type = {template_string};`: here we use a template that create a new annotation called 'my\_new\_annotation\_type' in the same span the 'output' annotation created, using a template string.

let's look at a simple example. suppose you want to recognize sequences of digits and then convert those digits to an integer and create an annotation with it. here is the jape code:

```jape
Phase: SimpleExample
  
Rule: NumberRecognition
(
    {Token.category == "CD"}:number
):match
-->
{
java:
    gate.AnnotationSet annots = bindings.get("number");
    gate.Annotation annotation = annots.iterator().next();
    String text = gate.Utils.stringFor(document, annotation);
    try {
        int number = Integer.parseInt(text);
        gate.FeatureMap features = Factory.newFeatureMap();
        features.put("integer_value", number);
        outputAnnotation = Factory.newAnnotation(annotation.getStartNode(),
            annotation.getEndNode(),
            "IntegerAnnotation",
            features
        );
        // put it back
        outputAS.add(outputAnnotation);

     } catch (NumberFormatException e) {
        // if is not a number.
     }
}:output

:output.IntegerAnnotation = {integer_value};

```

in this snippet:

*   the rule `numberrecognition` matches all tokens that are categorized as "cd", like numbers.
*   in the `java:` block we get the matching annotation called `number`, we get its text content and convert to an integer. then we create a new feature map with the integer and finally we create an annotation of type `integerannotation` with the feature map we created. we also add the annotation to output annotation set.
*   finally using the output label we create annotation of type `integerannotation` using a jape template.

note the use of `bindings.get("number")` to access the annotations that the pattern matched, this allows you to manipulate them. you can also access the gate document using `document`. in our case we need the text associated to the annotations. `gate.utils.stringfor()` gets that for us.

now, let's say we want to do something a little more involved, like handling multiple tokens. i once had to normalize dates by parsing different textual formats. that is when i learned to use the binding and templates in combination. here's how you would handle something similar, but a little simpler:

```jape
Phase: MultiTokenExample
  
Rule: MultiWordMatching
(
    ({Token.string == "hello"}):hello
    ({Token.string == "world"}):world
):match
-->
{
java:
  gate.AnnotationSet hello_annots = bindings.get("hello");
  gate.AnnotationSet world_annots = bindings.get("world");

    if (hello_annots.size() > 0 && world_annots.size() > 0) {
      gate.Annotation hello_annot = hello_annots.iterator().next();
      gate.Annotation world_annot = world_annots.iterator().next();
      long start = hello_annot.getStartNode().getOffset();
      long end = world_annot.getEndNode().getOffset();
    
        gate.FeatureMap features = Factory.newFeatureMap();
      features.put("concat_string", "hello world");
        
     outputAnnotation = Factory.newAnnotation(start, end,
            "CombinedStringAnnotation",
          features
        );
    
      outputAS.add(outputAnnotation);
    }

}:output
:output.CombinedStringAnnotation = {concat_string};
```

in this example:

*   the rule `multiwordmatching` matches sequences of "hello" and "world" words.
*   in the `java:` block we retrieve both matching annotations using the bindings api, if they exist we create the output annotation with all the spans of the two annotations, we create a feature map with the concatanated string "hello world".
*   we create and add the output annotation and then we create an annotation of type `combinedstringannotation` using the template.

templates can use jape notation as well. you can reference annotations with `:annotation_name`, for example in the previous example, instead of concatenating in the java block you can write in the template part:

```jape
:output.CombinedStringAnnotation = {hello.string} {world.string};
```

here is another example where we extract tokens in order and use them in the java block and also on a jape template to create new annotations:

```jape
Phase: SequenceExample
  
Rule: SequenceRecognition
(
    ({Token.string == "first"}:first)
    ({Token.string == "second"}:second)
    ({Token.string == "third"}:third)
):match
-->
{
java:
    gate.AnnotationSet first_annots = bindings.get("first");
    gate.AnnotationSet second_annots = bindings.get("second");
    gate.AnnotationSet third_annots = bindings.get("third");


   if (first_annots.size() > 0 && second_annots.size() > 0 && third_annots.size() > 0) {
        gate.Annotation first_annot = first_annots.iterator().next();
    gate.Annotation second_annot = second_annots.iterator().next();
        gate.Annotation third_annot = third_annots.iterator().next();
        long start = first_annot.getStartNode().getOffset();
      long end = third_annot.getEndNode().getOffset();
         String combined = gate.Utils.stringFor(document,first_annot) + " " + gate.Utils.stringFor(document,second_annot)+ " " + gate.Utils.stringFor(document,third_annot);
         gate.FeatureMap features = Factory.newFeatureMap();
        features.put("combined_string", combined);
       outputAnnotation = Factory.newAnnotation(start, end,
            "SequenceAnnotation",
             features
        );
        outputAS.add(outputAnnotation);
     }
  
}:output
:output.SequenceAnnotation = "{first.string} {second.string} {third.string}";
```

in this last example we:

*   the rule `sequencerecognition` matches sequences of "first", "second" and "third" words.
*   in the `java:` block we retrieve all the annotations, if they exist, get the text associated to each annotation and create a new output annotation.
*   we add the new annotation to output annotation set, and create an annotation with a jape template.

a good practice when writing these rules is to log what is going on, this may help you debug. you can use `gate.util.gate.gate.util.Out.prln("your message here")` to print messages to the gate message console. i did this so many times, that the logging window ended up with a permanent scrollbar (a joke).

now, i'd recommend diving deeper into some resources that i found helpful when learning this. the "gate developer's guide" is a must-have. it's the main documentation for gate. you will also benefit by reading the chapter on jape from "natural language processing with gate". it gives you a step-by-step guide with lots of examples on how to use jape. and last but not least, check the gate source code. i know this sounds insane, but seeing how other components of gate use jape with java really gave me some light on how to use the framework and its api. it is an invaluable resource.

these examples should get you going. remember, jape and java, in combination, is a powerful tool. keep experimenting and you'll conquer those complex information extraction tasks in no time. just remember to log things so you can understand what is going on with your rules and don't be afraid to consult the documentation, it is all there.
