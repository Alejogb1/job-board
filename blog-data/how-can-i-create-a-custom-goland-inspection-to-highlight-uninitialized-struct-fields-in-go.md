---
title: "How can I create a custom GoLand inspection to highlight uninitialized struct fields in Go?"
date: "2024-12-23"
id: "how-can-i-create-a-custom-goland-inspection-to-highlight-uninitialized-struct-fields-in-go"
---

 I remember a particularly thorny project a few years back where uninitialized struct fields were silently wreaking havoc, leading to some unpredictable behavior that took far too long to track down. A custom GoLand inspection to highlight these issues early would have been invaluable. So, while it's not built-in, creating one isn’t too complex, but requires a grasp of Go’s abstract syntax tree (AST) and the GoLand plugin API.

First, let’s clarify what we mean by “uninitialized struct fields.” In Go, if you declare a struct variable without explicitly assigning values to its fields, those fields are initialized with their zero values (e.g., `0` for integers, `""` for strings, `false` for booleans, and `nil` for pointers, slices, maps, functions, and channels). This isn't always a problem, of course. Sometimes, the zero value is perfectly acceptable and even intended. The issue we’re targeting is when a field *should* have a specific, non-zero value but is left to its default, leading to logical errors.

GoLand’s plugin architecture allows us to write custom inspections by leveraging its IntelliJ platform. We’ll primarily be dealing with two key components: the PSI (Program Structure Interface) that represents the source code as a tree structure and the inspection API, which allows us to define checks and highlight issues.

Let’s get into the practicalities. Our inspection needs to do the following:

1. **Identify struct literals:** We need to find places in the code where a struct is being instantiated using literal syntax (e.g., `MyStruct{field1: "value1", field2: 123}`).

2. **Get the struct type:** From the literal, we need to know the type of the struct being created.

3. **Inspect fields:** Once we know the type, we can look at all the fields defined in that struct.

4. **Check for initialized fields:** For each field in our struct definition, we check if it was explicitly initialized in the literal. If it is not, we mark it as uninitialized.

5. **Report the problem:** Finally, if any fields are found to be uninitialized we will highlight it as an inspection problem.

Let's dive into some example code snippets.

**Snippet 1: Struct Literal Identification and Field Extraction**

```java
import com.goide.psi.*;
import com.intellij.codeInspection.*;
import com.intellij.psi.PsiElement;
import com.intellij.psi.PsiElementVisitor;
import com.intellij.util.containers.ContainerUtil;
import org.jetbrains.annotations.NotNull;

import java.util.List;


public class UninitializedStructFieldInspection extends LocalInspectionTool {

    @NotNull
    @Override
    public PsiElementVisitor buildVisitor(@NotNull ProblemsHolder holder, boolean isOnTheFly) {
        return new PsiElementVisitor() {
             @Override
            public void visitElement(@NotNull PsiElement element) {
                if (element instanceof GoStructLiteral) {
                    GoStructLiteral structLiteral = (GoStructLiteral) element;
                    GoTypeReferenceExpression typeReference = structLiteral.getGoTypeReference();
                    if (typeReference == null) return; // Handle cases where type is not explicit
                    GoTypeDeclaration structDeclaration = typeReference.resolveType();

                    if (!(structDeclaration instanceof GoTypeDeclarationSpec)) return;
                    GoTypeDeclarationSpec typeDeclarationSpec = (GoTypeDeclarationSpec) structDeclaration;
                    GoTypeSpec typeSpec = typeDeclarationSpec.getTypeSpec();
                    if (!(typeSpec instanceof GoStructType)) return;
                    GoStructType structType = (GoStructType) typeSpec;
                    if(structType.getStructBody() == null) return;

                     checkFields(structLiteral, structType, holder);
                }
             }
         };
    }


    private void checkFields(@NotNull GoStructLiteral literal, GoStructType structType, @NotNull ProblemsHolder holder) {
        if (structType.getStructBody() == null) return;

        List<String> initializedFields = ContainerUtil.mapNotNull(literal.getCompositeElementList(),
                e -> { if (e instanceof GoElementKey)
                        return e.getText();
                else
                        return null;});
        
       structType.getStructBody().getFieldDeclarationList().forEach(fieldDeclaration -> {
            if (fieldDeclaration.getFieldDefinitionList().isEmpty()) return;
            String fieldName = fieldDeclaration.getFieldDefinitionList().get(0).getText();
            if (!initializedFields.contains(fieldName))
            {
                holder.registerProblem(literal, "Uninitialized field: '" + fieldName + "'", ProblemHighlightType.GENERIC_ERROR_OR_WARNING);
            }
        });
    }
}
```
This code snippet focuses on identifying `GoStructLiteral` elements (our struct literals), resolving the underlying type of the struct (using `resolveType`), and retrieving the `GoStructType` information, which contains details about its fields. The `checkFields` method would be called from within this code block.

**Snippet 2: Checking Initialized Fields**

```java
private void checkFields(@NotNull GoStructLiteral literal, GoStructType structType, @NotNull ProblemsHolder holder) {
        if (structType.getStructBody() == null) return;

        List<String> initializedFields = ContainerUtil.mapNotNull(literal.getCompositeElementList(),
                e -> { if (e instanceof GoElementKey)
                        return e.getText();
                else
                        return null;});
        
       structType.getStructBody().getFieldDeclarationList().forEach(fieldDeclaration -> {
            if (fieldDeclaration.getFieldDefinitionList().isEmpty()) return;
            String fieldName = fieldDeclaration.getFieldDefinitionList().get(0).getText();
            if (!initializedFields.contains(fieldName))
            {
                holder.registerProblem(literal, "Uninitialized field: '" + fieldName + "'", ProblemHighlightType.GENERIC_ERROR_OR_WARNING);
            }
        });
    }
```
This snippet goes inside the `buildVisitor` method. Here we are getting the actual fields which have been initialised by the literal from the `GoStructLiteral` component using the `GoElementKey`. Then, we take each field definition from the `GoStructType` and ensure that each of them have been initialised, and if not then register it to the `ProblemsHolder`.

**Snippet 3: Registering a Problem**

```java
 holder.registerProblem(literal, "Uninitialized field: '" + fieldName + "'", ProblemHighlightType.GENERIC_ERROR_OR_WARNING);
```
This snippet from within `checkFields` demonstrates how to mark a specific location within the code with a problem. `holder` is a `ProblemsHolder` object that provides functionality to mark a piece of code as a problem to the user. `ProblemHighlightType.GENERIC_ERROR_OR_WARNING` allows us to specify the type of the problem, such as warning or error.

Building this plugin requires setting up an IntelliJ plugin development environment and having the Go plugin's API readily available. Go through the official IntelliJ plugin documentation; it is the most reliable and accurate source of information for setting this up. Additionally, the `Structure and Interpretation of Computer Programs` (Abelson, Sussman & Sussman) is highly valuable for grasping the foundational concepts of language structure and representation, which underpins understanding ASTs and PSI. Finally, the Go language specification will be crucial for thoroughly understanding the language semantics we're attempting to analyze. I recommend going through the specification when you are attempting to cover additional language features.

In practical terms, I've found that testing these kinds of inspections can be tricky. Write extensive unit tests to cover different types of structs, different scenarios (e.g., embedded structs), and edge cases. Debugging inspections often requires stepping through the code within the plugin while running a GoLand instance in debug mode – which initially felt a little meta, but you quickly get the hang of it.

Ultimately, this inspection provides a tangible benefit: making hidden bugs visible and promoting more explicit and correct initialization practices. The upfront work of building a custom inspection can easily pay for itself by eliminating time spent debugging and promoting cleaner code. Creating a custom inspection is a valuable exercise in truly understanding how code analysis tools work. It’s a skill that has served me well many times over.
