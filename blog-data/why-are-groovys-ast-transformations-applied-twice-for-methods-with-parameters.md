---
title: "Why are Groovy's AST transformations applied twice for methods with parameters?"
date: "2024-12-23"
id: "why-are-groovys-ast-transformations-applied-twice-for-methods-with-parameters"
---

Alright, let's unpack this groovy behavior. It's a quirk that's tripped up quite a few developers, and I've certainly spent my fair share of time debugging its nuances. The apparent double application of abstract syntax tree (AST) transformations for methods with parameters in Groovy is not a bug, but rather an intended consequence of how the Groovy compiler handles method signatures and its internal phases of compilation. To understand this, we need to delve into Groovy's compilation pipeline and look closely at how method ASTs are initially constructed and subsequently modified.

In essence, when the Groovy compiler encounters a method definition with parameters, it generates a method AST that includes parameter information. This initial AST is, in a way, a placeholder. It contains the bare minimum structure needed to represent the method and its parameters. Crucially, at this stage, the parameter types might not have been fully resolved or processed. This is the first point of entry for AST transformations. Those transformations can then operate on this preliminary AST.

After the initial AST transformation phase, Groovy then proceeds with further type resolution and bytecode generation. This phase requires a complete and accurate representation of the method’s signature, including fully resolved parameter types. Groovy handles this by essentially creating a *new* AST node, a more complete version of the initial one that incorporates this crucial type information. This new AST node is not a replacement but a refactoring or an augmented version of the initial node created in the first phase. Crucially, it is this refactored node that triggers the application of AST transformations a *second* time. This second pass ensures that AST transformations have access to the complete method signature, incorporating concrete parameter type information, enabling transforms that require fully resolved parameter types to operate correctly.

To put it another way, think of the compilation in two parts: a setup phase to build the structure and then a refinement phase where the details are filled in. AST transformations might need to operate on different levels of detail within these phases.

I remember having a particularly frustrating experience with this a few years back while working on a custom logging framework using AST transformations. The transformation was designed to add logging statements at the beginning and end of methods. My initial approach worked perfectly for methods without parameters. However, for methods with parameters, I was seeing my logging statements appear twice. It took a while, and several debug sessions with the Groovy compiler source code, to pinpoint the cause: the transformations were being applied on that initial 'skeleton' AST, and then again to the refined, fully typed version during that second compilation phase.

To illustrate this, let’s consider a simplified example. First, consider a basic AST transformation that merely adds a println statement at the start of a method:

```groovy
import org.codehaus.groovy.ast.*
import org.codehaus.groovy.control.*
import org.codehaus.groovy.transform.*
import static org.codehaus.groovy.ast.tools.GeneralUtils.*


@GroovyASTTransformation(phase = CompilePhase.SEMANTIC_ANALYSIS)
class MySimpleTransformation implements ASTTransformation {
    void visit(ASTNode[] nodes, SourceUnit sourceUnit) {
        nodes.each { node ->
            if(node instanceof ClassNode){
                node.methods.each{method->
                    method.code = block(
                            callX(constX('java.lang.System',false),
                                    'println',
                                    args(constX("Entering Method: "+method.name, true))),
                            method.code)
                }
            }

        }
    }
}
```
Now, imagine this class:
```groovy
@MySimpleTransformation
class MyClass {
    def myMethod(){
        println "doing something..."
    }
    def myMethodWithParam(String name){
        println "doing something with $name"
    }
}
```
If you compile and run this code. myMethod will call the println once, before "doing something..", but `myMethodWithParam` will print the "Entering Method" twice, before printing the string "doing something with $name". The reason for this is because, this method with a parameter goes through two different versions of the method AST during compilation phase, each invoking the AST transformation.

Now, let's look at a transformation example that attempts to analyze parameter types more deeply.

```groovy
import org.codehaus.groovy.ast.*
import org.codehaus.groovy.control.*
import org.codehaus.groovy.transform.*
import static org.codehaus.groovy.ast.tools.GeneralUtils.*

@GroovyASTTransformation(phase = CompilePhase.SEMANTIC_ANALYSIS)
class MyParameterTypeTransformation implements ASTTransformation {
    void visit(ASTNode[] nodes, SourceUnit sourceUnit) {
        nodes.each { node ->
            if (node instanceof ClassNode) {
                node.methods.each { method ->
                    method.parameters?.each { param ->
                       def paramType = param.type
                       if (paramType != null){
                            method.code = block(
                                    callX(constX('java.lang.System', false),
                                        'println',
                                        args(constX("Parameter: "+param.name+" is of type:"+ paramType.name, true))),
                                method.code
                                )
                       }
                    }
                }
            }
        }
    }
}
```

Applying this transformation to the previous class:
```groovy
@MyParameterTypeTransformation
class MyClass {
    def myMethod(){
        println "doing something..."
    }
    def myMethodWithParam(String name){
        println "doing something with $name"
    }
}

```
This will now correctly print out the parameter name and type within the method before "doing something with $name". However, for the second phase, the type information is now concrete instead of just the placeholder, meaning the transformation will correctly print out parameter information at the time when the type resolution is performed, but only for methods with parameters.

Now, for a more advanced transformation that requires specific type information and shows how this is important for some more complex operations:

```groovy
import org.codehaus.groovy.ast.*
import org.codehaus.groovy.control.*
import org.codehaus.groovy.transform.*
import static org.codehaus.groovy.ast.tools.GeneralUtils.*

@GroovyASTTransformation(phase = CompilePhase.SEMANTIC_ANALYSIS)
class MyTypeSpecificTransformation implements ASTTransformation {
    void visit(ASTNode[] nodes, SourceUnit sourceUnit) {
        nodes.each { node ->
            if (node instanceof ClassNode) {
                node.methods.each { method ->
                    method.parameters?.each { param ->
                        if (param.type.name == 'String') {
                              method.code = block(
                                    callX(constX('java.lang.System', false),
                                        'println',
                                        args(constX("String parameter: "+param.name+" detected", true))),
                                method.code
                                )
                        }
                    }
                }
            }
        }
    }
}
```
And running this with the class again:
```groovy
@MyTypeSpecificTransformation
class MyClass {
    def myMethod(){
        println "doing something..."
    }
    def myMethodWithParam(String name){
        println "doing something with $name"
    }
    def myMethodWithOtherParam(Integer count){
        println "doing something with $count"
    }
}
```
In this case, `myMethod` will not trigger the output. However, `myMethodWithParam` will, but only at the second execution when the String type has been fully processed by the compiler. While `myMethodWithOtherParam` will not trigger the output since it does not have a String parameter.

The key takeaway here is that AST transformations are often designed to operate on a fully resolved AST. If you need to capture the initial state of a method AST with unresolved types, you would use earlier phases. If you need information about concrete types, then you should expect your transformations to happen on the refined AST.

This nuanced behavior has its trade-offs, but it provides a flexible mechanism for handling transformations, catering to a wide range of scenarios. It can be challenging to grasp initially, but once you understand this two-pass process, it becomes much easier to write robust and predictable AST transformations in Groovy.

For further deep dives into Groovy's compilation process, I would highly recommend the book "Groovy in Action" by Dierk Koenig et al. Its detailed explanation of the compiler phases is invaluable. Additionally, reviewing the Groovy compiler source code itself (available on GitHub) can also reveal insights into the internal mechanisms. Be sure to look specifically at the `org.codehaus.groovy.control` package and related classes, including the various phases of compilation.
